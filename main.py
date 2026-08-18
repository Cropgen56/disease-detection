import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


# Add the directory containing main.py to system path for robust imports
sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from inference import ModelManager

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backend")


# Disease Data (loaded once at startup, after logger is ready)

_DATA_DIR = Path(__file__).resolve().parent / "data"

def _load_json(path: Path) -> dict:
    """Load a JSON file; return empty dict and log a warning on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Data file not found: {path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {path}: {e}")
        return {}

DISEASE_DATA_EN: dict = _load_json(_DATA_DIR / "crop_diseases.json")
DISEASE_DATA_HI: dict = _load_json(_DATA_DIR / "crop_diseases_hindi.json")

# Initialize FastAPI App
app = FastAPI(
    title="Crop Disease Classification Backend",
    description="Backend API for crop classification (Phase 1) and disease detection (Phase 2).",
    version="1.2.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate Global Model Manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Attempt to load Stage 0, Phase 1 and Phase 2 models at startup."""
    logger.info("Initializing models on startup...")

    s0_ok = model_manager.load_stage0()
    if s0_ok:
        logger.info("Stage 0 (YOLOE leaf gate) loaded successfully.")
    else:
        logger.warning(
            "Stage 0 model files not found or failed to load. "
            "Expected in 'weights/': yoloe-11s-seg.pt, leaf_vocab_embeddings.pt, "
            "vocab_config.json, stage0_config.json. "
            "Leaf gating will be SKIPPED until Stage 0 is available."
        )

    p1_ok = model_manager.load_phase1()
    if p1_ok:
        logger.info("Phase 1 models loaded successfully.")
    else:
        logger.warning("Phase 1 model weights not found or failed to load. "
                       "Place files in 'weights/' to enable Phase 1.")

    p2_ok = model_manager.load_phase2()
    if p2_ok:
        logger.info("Phase 2 models loaded successfully.")
    else:
        logger.warning("Phase 2 model weights not found or failed to load. "
                       "Place files in 'weights/' to enable Phase 2.")


# Request and Response Schemas

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Details regarding the failure case.")

class CropPrediction(BaseModel):
    crop: str = Field(..., description="Normalized class name of the crop.")
    display_name: str = Field(..., description="Human-readable formatted crop name.")
    confidence: float = Field(..., description="Confidence probability (0 to 1).")

class CropClassificationResponse(BaseModel):
    predicted_crop: str = Field(..., description="Top predicted crop.")
    display_name: str = Field(..., description="Display name of the top predicted crop.")
    confidence: float = Field(..., description="Confidence for the top predicted crop.")
    all_predictions: List[CropPrediction] = Field(..., description="Top predictions with confidence levels.")

class DiseasePrediction(BaseModel):
    disease: str = Field(..., description="Human-readable formatted disease name.")
    raw_label: str = Field(..., description="Raw dataset disease label.")
    confidence: float = Field(..., description="Confidence probability (0 to 1).")

class DiseaseDetectionResponse(BaseModel):
    crop: str = Field(..., description="Target crop.")
    predicted_disease: str = Field(..., description="Top predicted disease.")
    raw_disease_label: str = Field(..., description="Raw dataset disease label for the top prediction.")
    confidence: float = Field(..., description="Confidence for the top predicted disease.")
    confidence_level: str = Field(..., description="Interpretability category (High, Moderate, Low).")
    top_k_predictions: List[DiseasePrediction] = Field(..., description="Top predictions sorted by confidence.")

class CropMetadata(BaseModel):
    crop: str = Field(..., description="Normalized crop identifier.")
    valid_diseases: List[str] = Field(..., description="Canonical disease names registered for this crop.")

class MetadataResponse(BaseModel):
    # Phase 1 — all classifiable crops
    all_crops_count: int = Field(..., description="Total crops Phase 1 can classify.")
    all_crops: List[str] = Field(..., description="All crop identifiers Phase 1 was trained on (33 crops).")
    all_crops_display: dict = Field(..., description="Map of all crop identifiers to human-readable display names.")
    # Phase 2 — subset with disease detection
    disease_detection_crops_count: int = Field(..., description="Number of crops that also have Phase 2 disease detection.")
    disease_detection_crops: List[str] = Field(..., description="Crop identifiers that support disease detection.")
    # Model load status
    stage0_active: bool = Field(..., description="Indicates if Stage 0 YOLOE leaf gate is loaded.")
    phase1_active: bool = Field(..., description="Indicates if Phase 1 crop classifier is loaded.")
    phase1_crop_count: int = Field(..., description="Number of crop classes the Phase 1 model was trained on.")
    phase2_active: bool = Field(..., description="Indicates if Phase 2 disease detector is loaded.")
    total_disease_classes: int = Field(..., description="Total unique disease classes in the Phase 2 model (across all crops).")

# symptoms and control
class SymptomsControlResponse(BaseModel):
    crop: str = Field(..., description="Normalized crop identifier.")
    disease_en: str = Field(..., description="English display name of the disease.")
    disease_hi: Optional[str] = Field(None, description="Hindi name of the disease (None when lang=en).")
    lang: str = Field(..., description="Language of the returned symptoms/control content ('en' or 'hi').")
    symptoms: List[str] = Field(..., description="List of symptom descriptions in the requested language.")
    control: List[str] = Field(..., description="List of control/treatment measures in the requested language.")


class Stage0GateResponse(BaseModel):
    status: str = Field(..., description="Gate decision: 'leaf', 'rejected', or 'no_object_detected'.")
    message: str = Field(..., description="Human-readable explanation of the gate decision.")
    detected_object: Optional[str] = Field(None, description="What was detected when status is 'rejected'.")
    confidence: Optional[float] = Field(None, description="Detection confidence (leaf or rejected object).")
    box: Optional[List[int]] = Field(None, description="Padded bounding box [x1, y1, x2, y2] when status is 'leaf'.")


# Helpers
CROP_DISPLAY_MAP = {
    'apple': 'Apple',
    'bean': 'Bean',
    'bell_pepper': 'Bell Pepper',
    'blackgram_greengram': 'Blackgram / Greengram',
    'cherry': 'Cherry',
    'chilli': 'Chilli',
    'coconut': 'Coconut',
    'coffee': 'Coffee',
    'corn': 'Corn / Maize',
    'cotton': 'Cotton',
    'dragon_fruit': 'Dragon Fruit',
    'eggplant': 'Eggplant / Brinjal',
    'grape': 'Grape',
    'groundnut': 'Groundnut',
    'jute': 'Jute',
    'lemon': 'Lemon',
    'mango': 'Mango',
    'okra': 'Okra / Bhindi',
    'onion': 'Onion',
    'orange': 'Orange',
    'paddy': 'Paddy / Rice',
    'peach': 'Peach',
    'pineapple': 'Pineapple',
    'potato': 'Potato',
    'pumpkin': 'Pumpkin',
    'raspberry': 'Raspberry',
    'snake_gourd': 'Snake Gourd',
    'soybean': 'Soybean',
    'strawberry': 'Strawberry',
    'sugarcane': 'Sugarcane',
    'tea': 'Tea',
    'tomato': 'Tomato',
    'wheat': 'Wheat',
}

def format_crop_display(crop_name: str) -> str:
    """Formats crop name to human-readable format."""
    return CROP_DISPLAY_MAP.get(crop_name.lower().strip(), crop_name.replace("_", " ").title())

def format_disease_display(disease_label: str) -> str:
    """Formats raw disease label (e.g. 'tomato__yellow_leaf_curl_virus') to human-readable (e.g. 'Yellow Leaf Curl Virus')."""
    if "__" in disease_label:
        _, suffix = disease_label.split("__", 1)
    else:
        suffix = disease_label
    return suffix.replace("_", " ").title()

def determine_confidence_level(prob: float) -> str:
    """Categorizes confidence level matching plot colors in Colab."""
    if prob >= 0.90:
        return "High confidence"
    elif prob >= 0.60:
        return "Moderate"
    return "Low — treat with caution"

def load_image(file_bytes: bytes) -> Image.Image:
    """Safely decodes raw file bytes to PIL Image."""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Image decode failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image or could not be decoded."
        )


# API Routes


def _save_upload_to_temp(file_bytes: bytes, suffix: str = ".jpg") -> str:
    """Write raw upload bytes to a NamedTemporaryFile and return its path.
    The caller is responsible for deleting the file after use.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
    except Exception:
        os.unlink(path)
        raise
    return path


@app.get("/")
def read_root():
    return {
        "status": "online",
        "stage0_active": model_manager.is_stage0_loaded(),
        "phase1_active": model_manager.is_phase1_loaded(),
        "phase2_active": model_manager.is_phase2_loaded()
    }


@app.post(
    "/api/v1/stage0-gate",
    response_model=Stage0GateResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Stage 0 model not loaded."},
        400: {"model": ErrorResponse, "description": "Invalid image file."},
    },
    tags=["Stage 0"],
)
async def stage0_gate_endpoint(
    file: UploadFile = File(..., description="Upload image to test the leaf gate."),
):
    """
    **Stage 0 Endpoint: Leaf Gate (standalone test)**

    Runs YOLOE open-vocabulary detection to decide whether the image contains a
    leaf. Useful for integration testing and debugging Stage 0 independently.

    Returns:
    - `leaf` — image contains a detectable leaf; includes bounding box and confidence.
    - `rejected` — non-leaf object detected with sufficient confidence.
    - `no_object_detected` — nothing detected above the gate threshold.

    > **Note:** ~89% leaf recall / ~1.1% false-positive rate (Open Images
    > negative set). True production FP rate on farmer submissions is unverified.
    """
    if not model_manager.is_stage0_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stage 0 model is not loaded. Ensure yoloe-11s-seg.pt, "
                   "leaf_vocab_embeddings.pt, vocab_config.json, and "
                   "stage0_config.json are present in the weights/ folder.",
        )

    contents = await file.read()
    # Validate image before passing to YOLOE
    load_image(contents)  # raises 400 on decode failure

    tmp_path = _save_upload_to_temp(contents)
    try:
        gate = model_manager.run_stage0(tmp_path)
    except Exception as e:
        logger.exception("Stage 0 inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stage 0 inference failed: {e}",
        )
    finally:
        os.unlink(tmp_path)

    if gate["status"] == "no_object_detected":
        return Stage0GateResponse(
            status="no_object_detected",
            message="No object detected above the gate threshold. Please retake the photo.",
        )
    if gate["status"] == "rejected":
        return Stage0GateResponse(
            status="rejected",
            message=f"Detected '{gate['detected_object']}', not a leaf. Please retake the photo.",
            detected_object=gate["detected_object"],
            confidence=gate["confidence"],
        )
    # status == "leaf"
    return Stage0GateResponse(
        status="leaf",
        message="Leaf detected.",
        confidence=gate["confidence"],
        box=gate["box"],
    )


@app.post(
    "/api/v1/classify-crop",
    response_model=CropClassificationResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Phase 1 model weights not loaded."},
        422: {"model": ErrorResponse, "description": "Stage 0 gate rejected: not a leaf image."},
        400: {"model": ErrorResponse, "description": "Invalid image file uploaded."}
    }
)
async def classify_crop_endpoint(
    file: UploadFile = File(..., description="Upload leaf image file."),
    direct_resize: bool = Query(
        False,
        description="If true, bypasses CenterCrop and resizes the entire image directly to 240x240."
    )
):
    """
    **Phase 1 Endpoint: Crop Classification**

    Classifies the uploaded image into one of the 33 crop classes.
    When Stage 0 (YOLOE leaf gate) is loaded, the image is screened first: non-leaf
    images are rejected with HTTP 422 before reaching Stage 1 inference. If Stage 0 is
    unavailable, the gate is skipped and classification proceeds directly.

    Requires Phase 1 weight file (`stage1_dinov2_30class.pt` or
    `efficientnet_b1_crop_mini.pt`) to be present in the `weights/` directory.
    """
    if not model_manager.is_phase1_loaded():
        # Re-check in case user dropped weights while server was running
        if not model_manager.load_phase1():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Phase 1 Crop Classification model is not loaded. Ensure weights and class_to_idx_phase1.json are in the weights/ folder."
            )

    contents = await file.read()

    # ── Stage 0 Leaf Gate ──────────────────────────────────────────────────
    # Runs YOLOE at conf=0.01; resolve_call() applies the 0.08 threshold.
    # Gate is skipped gracefully if Stage 0 failed to load at startup.
    image: Image.Image
    if model_manager.is_stage0_loaded():
        # Validate image first (raises 400 on bad bytes)
        load_image(contents)
        tmp_path = _save_upload_to_temp(contents)
        try:
            gate = model_manager.run_stage0(tmp_path)
        except Exception as e:
            logger.exception("Stage 0 inference error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Stage 0 inference failed: {e}",
            )
        finally:
            os.unlink(tmp_path)

        if gate["status"] == "no_object_detected":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No leaf detected. Please retake the photo.",
            )
        if gate["status"] == "rejected":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Detected '{gate['detected_object']}' (confidence {gate['confidence']:.2%}), "
                       "not a leaf. Please retake the photo.",
            )
        # status == "leaf" — use the padded crop for Stage 1
        image = Image.fromarray(gate["crop"])  # gate['crop'] is RGB numpy array
        logger.info(f"Stage 0 passed (leaf, conf={gate['confidence']}, box={gate['box']})")
    else:
        logger.debug("Stage 0 not loaded — skipping leaf gate.")
        image = load_image(contents)
    # ── End Stage 0 ────────────────────────────────────────────────────────

    try:
        predictions = model_manager.predict_crop(image, direct_resize=direct_resize, top_k=5)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference process failed: {str(e)}"
        )

    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model returned no classification results."
        )

    top_crop, top_conf = predictions[0]
    all_preds_formatted = [
        CropPrediction(
            crop=c,
            display_name=format_crop_display(c),
            confidence=conf
        ) for c, conf in predictions
    ]

    return CropClassificationResponse(
        predicted_crop=top_crop,
        display_name=format_crop_display(top_crop),
        confidence=top_conf,
        all_predictions=all_preds_formatted
    )



@app.post(
    "/api/v1/detect-disease",
    response_model=DiseaseDetectionResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Phase 2 model weights not loaded."},
        400: {"model": ErrorResponse, "description": "Unsupported crop or invalid inputs."}
    }
)
async def detect_disease_endpoint(
    crop: str = Query(..., description="Crop class name (e.g. 'tomato', 'potato')."),
    file: UploadFile = File(..., description="Upload leaf image file."),
    direct_resize: bool = Query(
        False, 
        description="If true, bypasses CenterCrop and resizes the entire image directly to 224x224."
    )
):
    """
    **Phase 2 Endpoint: Disease Detection**
    
    Detects the disease for the specified crop by running a masked inference over
    the 100+ disease model outputs.
    Requires Phase 2 weight file (`efficientnet_b0_disease_mini.pt`) and maps to be present in `weights/` directory.
    """
    if not model_manager.is_phase2_loaded():
        # Re-check in case user dropped weights while server was running
        if not model_manager.load_phase2():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Phase 2 Disease Detection model is not loaded. Ensure weights/efficientnet_b0_disease_mini.pt and weights/crop_disease_idx_map.json are in place."
            )

    crop_clean = crop.lower().strip()
    supported_crops = model_manager.get_supported_crops()
    if crop_clean not in supported_crops:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Crop '{crop}' is not supported. Supported crops: {supported_crops}"
        )

    contents = await file.read()
    image = load_image(contents)

    try:
        predictions = model_manager.predict_disease(image, crop_clean, direct_resize=direct_resize, top_k=5)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference process failed: {str(e)}"
        )

    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model returned no disease diagnosis."
        )

    top_disease_raw, top_conf = predictions[0]
    top_disease_display = format_disease_display(top_disease_raw)
    
    top_k_formatted = [
        DiseasePrediction(
            disease=format_disease_display(raw_lbl),
            raw_label=raw_lbl,
            confidence=conf
        ) for raw_lbl, conf in predictions
    ]

    return DiseaseDetectionResponse(
        crop=crop_clean,
        predicted_disease=top_disease_display,
        raw_disease_label=top_disease_raw,
        confidence=top_conf,
        confidence_level=determine_confidence_level(top_conf),
        top_k_predictions=top_k_formatted
    )


@app.get("/api/v1/metadata", response_model=MetadataResponse, tags=["Metadata"])
async def get_metadata():
    """
    **Metadata Endpoint**

    Returns:
    - **all_crops** — all 33 crops Phase 1 can classify (crop name + display name)
    - **disease_detection_crops** — subset of crops that also have Phase 2 disease detection
    - Model load statuses for Stage 0, Phase 1, and Phase 2
    - Total disease class count across all Phase 2 crops
    """
    all_crops = model_manager.get_phase1_crops()
    disease_crops = model_manager.get_supported_crops()
    return MetadataResponse(
        all_crops_count=len(all_crops),
        all_crops=all_crops,
        all_crops_display={c: format_crop_display(c) for c in all_crops},
        disease_detection_crops_count=len(disease_crops),
        disease_detection_crops=disease_crops,
        stage0_active=model_manager.is_stage0_loaded(),
        phase1_active=model_manager.is_phase1_loaded(),
        phase1_crop_count=model_manager.get_phase1_crop_count(),
        phase2_active=model_manager.is_phase2_loaded(),
        total_disease_classes=model_manager.get_total_disease_count(),
    )

@app.get(
     "/api/v1/symptoms-control",
    response_model=SymptomsControlResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Crop or disease not found in the data files."},
        400: {"model": ErrorResponse, "description": "Invalid lang parameter."},
        503: {"model": ErrorResponse, "description": "Disease data files are not loaded."},
    }
)
async def get_symptoms_control(
    crop: str = Query(..., description="Crop name (e.g. 'tomato', 'potato')."),
    disease: str = Query(..., description="English display name of the disease (e.g. 'Early Blight')."),
    lang: str = Query("en", description="Response language: 'en' (default) or 'hi'."),
):
    """
    **Symptoms & Control Endpoint**
    Returns symptom descriptions and control/treatment measures for a given
    crop–disease pair, in English or Hindi.
    - `crop` — normalized crop key (same identifiers used by /classify-crop and /detect-disease)
    - `disease` — English display name of the disease as returned by /detect-disease (e.g. 'Early Blight')
    - `lang` — 'en' for English (default), 'hi' for Hindi
    """

    # Validate lang

    lang = lang.lower().strip()
    if lang not in {"en", "hi"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid lang parameter. currently only 'en' or 'hi' is supported."
        )
        
    # check data
    if not DISEASE_DATA_EN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "English disease data not loaded."
        )
    if lang=="hi" and not DISEASE_DATA_HI:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "Hindi disease data not loaded"
            
        )

    crop_clean = crop.lower().strip()
    disease_clean = disease.strip()

    # lookup crop in english data
    crop_entries_en = DISEASE_DATA_EN.get(crop_clean)
    if crop_entries_en is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
             detail=f"Crop '{crop_clean}' not found in disease data. "
                    f"Supported crops: {list(DISEASE_DATA_EN.keys())}"
                    
        )
    # find disease index in english data (case-insensitive)
    matched_index: Optional[int] = None
    matched_en_entry: Optional[dict] = None
    for i, entry in enumerate(crop_entries_en):
        if entry["disease"].lower() == disease_clean.lower():
            matched_index = i
            matched_en_entry = entry
            break

    if matched_index is None:
        available = [e["disease"] for e in crop_entries_en]
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = f"Disease '{disease_clean}' not found for crop '{crop_clean}'. "
                     f"Available diseases: {available}"
        )
    
    # Serve English
    if lang == "en":
        return SymptomsControlResponse(
            crop=crop_clean,
            disease_en=matched_en_entry["disease"],
            disease_hi=None,
            lang="en",
            symptoms=matched_en_entry["symptoms"],
            control=matched_en_entry["control"],
        )
    # Serve Hindi (same index, Hindi JSON)
    crop_entries_hi = DISEASE_DATA_HI.get(crop_clean, [])
    if matched_index >= len(crop_entries_hi):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Hindi data is out of sync for crop '{crop_clean}' at index {matched_index}."
        )
    hi_entry = crop_entries_hi[matched_index]
    return SymptomsControlResponse(
        crop=crop_clean,
        disease_en=matched_en_entry["disease"],
        disease_hi=hi_entry["disease"],
        lang="hi",
        symptoms=hi_entry["symptoms"],
        control=hi_entry["control"],
    )
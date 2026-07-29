import io
import logging
import sys
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
    """Attempt to load both Phase 1 and Phase 2 models at startup."""
    logger.info("Initializing models on startup...")
    
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
    supported_crops_count: int = Field(..., description="Number of supported crops.")
    supported_crops: List[str] = Field(..., description="List of supported crop identifiers.")
    phase1_active: bool = Field(..., description="Indicates if Phase 1 crop model is loaded.")
    phase2_active: bool = Field(..., description="Indicates if Phase 2 disease model is loaded.")


# Helpers

CROP_DISPLAY_MAP = {
    'apple': 'Apple', 'bean': 'Bean', 'bell_pepper': 'Bell Pepper',
    'blackgram': 'Blackgram', 'blueberry': 'Blueberry', 'cherry': 'Cherry',
    'chilli': 'Chilli', 'coconut': 'Coconut', 'coffee': 'Coffee',
    'corn': 'Corn / Maize', 'cotton': 'Cotton', 'dragon_fruit': 'Dragon Fruit',
    'grape': 'Grape', 'groundnut': 'Groundnut', 'jute': 'Jute',
    'lemon': 'Lemon', 'mango': 'Mango', 'onion': 'Onion', 'orange': 'Orange',
    'paddy': 'Paddy / Rice', 'peach': 'Peach', 'pineapple': 'Pineapple',
    'potato': 'Potato', 'raspberry': 'Raspberry', 'snake_gourd': 'Snake Gourd',
    'soybean': 'Soybean', 'squash': 'Squash', 'strawberry': 'Strawberry',
    'sugarcane': 'Sugarcane', 'tea': 'Tea', 'tomato': 'Tomato', 'wheat': 'Wheat',
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

@app.get("/")
def read_root():
    return {
        "status": "online",
        "phase1_active": model_manager.is_phase1_loaded(),
        "phase2_active": model_manager.is_phase2_loaded()
    }

@app.post(
    "/api/v1/classify-crop",
    response_model=CropClassificationResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Phase 1 model weights not loaded."},
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
    
    Classifies the uploaded image into one of the 32 crop classes.
    Requires Phase 1 weight file (`efficientnet_b1_crop_mini.pt` or `yolo11n_crop_mini.pt`) to be present in the `weights/` directory.
    """
    if not model_manager.is_phase1_loaded():
        # Re-check in case user dropped weights while server was running
        if not model_manager.load_phase1():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Phase 1 Crop Classification model is not loaded. Ensure weights and class_to_idx_phase1.json are in the weights/ folder."
            )

    contents = await file.read()
    image = load_image(contents)

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


@app.get("/api/v1/metadata", response_model=MetadataResponse)
async def get_metadata():
    """
    **Metadata Endpoint**
    
    Returns lists of supported crops, active phase statuses, and general configurations.
    """
    return MetadataResponse(
        supported_crops_count=len(model_manager.get_supported_crops()),
        supported_crops=model_manager.get_supported_crops(),
        phase1_active=model_manager.is_phase1_loaded(),
        phase2_active=model_manager.is_phase2_loaded()
    )

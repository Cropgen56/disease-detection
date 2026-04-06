"""
Crop Disease Detection Service
Based on ResNet9 trained on PlantVillage dataset (38 classes, 99.2% accuracy)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import torch
import io
from PIL import Image

from app.model import load_model, predict, get_valid_plants, resolve_plant_prefix
from app.schemas import PredictionResponse, HealthResponse


# Lifespan: load model once at startup 
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield
    # cleanup (if needed) goes here


# App
app = FastAPI(
    title="Crop Disease Detection API",
    description="Detects plant diseases from leaf images using ResNet9 trained on PlantVillage (38 classes).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes 
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return {
        "status": "ok",
        "model": "ResNet9",
        "num_classes": 38,
        "input_size": "256x256",
    }


@app.get("/plants", tags=["Info"])
def list_plants():
    """
    Returns the list of plant names accepted by the `plant_name` parameter
    in /predict. Pass any of these strings to enable constrained prediction.
    """
    return {"supported_plants": get_valid_plants()}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_disease(
    file: UploadFile = File(...),
    plant_name: Optional[str] = Query(
        default=None,
        description=(
            "Optional. Name of the plant in the image (e.g. 'tomato', 'potato', 'corn'). "
            "When provided, prediction is constrained to that plant's disease classes only, "
            "improving accuracy. Call GET /plants for the full list of accepted names."
        ),
    ),
):
    """
    Upload a leaf image (JPG, PNG, WEBP) and get back the predicted disease,
    confidence score, and top-5 alternatives.

    The endpoint handles all preprocessing internally:
    - Converts to RGB (handles RGBA/grayscale inputs)
    - Resizes to 256×256
    - Normalises pixel values to [0, 1]
    """
    # Validate plant_name if provided 
    if plant_name and not resolve_plant_prefix(plant_name):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown plant '{plant_name}'. "
                f"Call GET /plants for the list of accepted plant names."
            ),
        )

    # Validate file type 
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Send JPEG, PNG, or WEBP.",
        )

    # Read raw bytes 
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # open image 
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Make sure the file is a valid image.")

    # Run inference (preprocessing happens inside predict())
    result = predict(image, app.state.model, plant_name=plant_name)
    return result
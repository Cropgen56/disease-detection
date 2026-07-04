# Crop Disease Detection Backend API

This repository contains a full-fledged FastAPI backend implementation for crop classification (Phase 1) and disease detection (Phase 2) based on the PyTorch and YOLOv11 notebooks.

---

## Project Structure

```text
disease_detection_v2/
├── weights/                           # Model weights and JSON mapping configurations
│   ├── efficientnet_b0_disease_mini.pt # Phase 2 model weights
│   ├── crop_disease_idx_map.json      # Phase 2 crop -> disease index mask
│   ├── class_to_idx_phase1.json       # Phase 1 crop index mapping (Renamed)
│   └── efficientnet_b1_crop_mini.pt   # Phase 1 crop classifier weights (Or yolo11n_crop_mini.pt)
├── main.py                            # FastAPI app, endpoints, and startup events
├── inference.py                       # Inference logic (preprocessing, model loaders, predictions)
├── requirements.txt                   # Dependency checklist
└── README.md                          # This documentation file
```

---

## Installation & Setup

1. **Install Dependencies**:
   It is recommended to run this in a virtual environment (conda, venv, or poetry):
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Model Weights**:
   Download the following files from your Google Drive path `CropClassifier_v2/models/` and place them inside the `weights/` directory:

   * **Phase 2 (Disease Detector)**:
     - Checkpoint file: `stage2_disease/efficientnet_b0_disease_mini.pt`
     - Mappings file: `stage2_disease/crop_disease_idx_map.json`

   * **Phase 1 (Crop Classifier)**:
     - Mappings file: `stage1_crop/class_to_idx.json` (Save it to `weights/class_to_idx_phase1.json` to prevent conflict).
     - Checkpoint file: Either `stage1_crop/efficientnet_b1_crop_mini.pt` OR `stage1_crop/yolo11n_crop_mini/weights/best.pt` (placed as `weights/yolo11n_crop_mini.pt` or `weights/best.pt`).

---

## Running the API Server

Start the development server using Uvicorn:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Once running, the server will output logging indicating which models were successfully loaded from the `weights/` folder.

---

## API Endpoints

### 1. Interactive Swagger Docs
Access the interactive API documentation at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 2. Phase 1: Classify Crop (`POST /api/v1/classify-crop`)
Used to predict what crop type is in the image.
- **Request Parameters**:
  - `file`: Upload file (multipart/form-data)
  - `direct_resize` (optional): `true` or `false` (default `false`). If `true`, the entire image is resized to 240x240 directly (retaining boundaries). If `false` (default), the center crop matches training transforms.
- **Response**:
  ```json
  {
    "predicted_crop": "tomato",
    "display_name": "Tomato",
    "confidence": 0.985,
    "all_predictions": [
      {"crop": "tomato", "display_name": "Tomato", "confidence": 0.985},
      {"crop": "potato", "display_name": "Potato", "confidence": 0.012}
    ]
  }
  ```

### 3. Phase 2: Detect Disease (`POST /api/v1/detect-disease`)
Used to predict the disease of a specific crop using index masking (restricts predictions only to valid diseases for that crop).
- **Request Parameters**:
  - `crop`: Query parameter (e.g. `crop=tomato`)
  - `file`: Upload file (multipart/form-data)
  - `direct_resize` (optional): `true` or `false` (default `false`). If `true`, the image is resized to 224x224 directly.
- **Response**:
  ```json
  {
    "crop": "tomato",
    "predicted_disease": "Yellow Leaf Curl Virus",
    "raw_disease_label": "tomato__yellow_leaf_curl_virus",
    "confidence": 0.932,
    "confidence_level": "High confidence",
    "top_k_predictions": [
      {"disease": "Yellow Leaf Curl Virus", "raw_label": "tomato__yellow_leaf_curl_virus", "confidence": 0.932},
      {"disease": "Healthy", "raw_label": "tomato__healthy", "confidence": 0.041}
    ]
  }
  ```

### 4. Database Metadata (`GET /api/v1/metadata`)
Lists supported crops, registration info, and active model flags.
- **Response**:
  ```json
  {
    "supported_crops_count": 18,
    "supported_crops": ["apple", "bean", "bell_pepper", ...],
    "phase1_active": true,
    "phase2_active": true
  }
  ```

---

## Preprocessing: Crop vs. Resize Details

* **`direct_resize=false` (Default CenterCrop)**: Resizes input to 256x256 and crops the center (240x240 for Phase 1, 224x224 for Phase 2). Best for matching training data distributions and leaf textures.
* **`direct_resize=true`**: Bypasses the crop step and resizes the entire image directly to the target shape (240x240 for Phase 1, 224x224 for Phase 2). Useful if the target disease symptoms are located at the edges of the image, although aspect ratio distortion may occur.

## 1. Directory Structure

To keep the system simple and easy to maintain, we used a flat, modular directory structure:

```text
disease_detection_v2/
├── weights/                          # Stores ML models and class maps
│   ├── class_to_idx_phase1.json      # Phase 1 crop name -> index mapping
│   ├── efficientnet_b1_crop_mini.pt  # Phase 1 model weights (EfficientNet-B1)
│   ├── crop_disease_idx_map.json     # Phase 2 crop name -> valid disease index list
│   └── efficientnet_b0_disease_mini.pt # Phase 2 model weights (EfficientNet-B0)
├── main.py                           # FastAPI configuration and endpoints routing
├── inference.py                      # Model loading, image transforms, and prediction logic
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation and execution instructions
```

---

## 2. Startup Flow

When you run `uvicorn main:app --reload`:

```mermaid
graph TD
    A[Start FastAPI App] --> B[Startup Event Triggered]
    B --> C[Initialize ModelManager]
    C --> D[Load Phase 1 Mappings & Checkpoint]
    D -->|Success| E[Phase 1 Active = True]
    D -->|Weights Missing| F[Print Warning: P1 Disabled]
    C --> G[Load Phase 2 Mappings & Checkpoint]
    G -->|Success| H[Phase 2 Active = True]
    G -->|Weights Missing| I[Print Warning: P2 Disabled]
    E --> J[Ready to Accept Connections]
    F --> J
    H --> J
    I --> J
```

* **Graceful Degradation**: If model weights are missing, the server does **not** crash. It starts up normally and logs a warning. If a client attempts to call an endpoint whose weights aren't loaded, they receive a clean `HTTP 503 Service Unavailable` response with setup instructions.

---

## 3. Phase 1: Crop Classification Flow

Called via `POST /api/v1/classify-crop`:

```mermaid
sequenceDiagram
    participant User
    participant MainAPI as main.py
    participant Infer as inference.py
    participant Model1 as EfficientNet-B1

    User->>MainAPI: POST /api/v1/classify-crop (image, direct_resize=false/true)
    MainAPI->>MainAPI: Check if Phase 1 weights are active
    alt Phase 1 Not Active
        MainAPI-->>User: HTTP 503 (Model not loaded)
    end
    MainAPI->>MainAPI: Decode uploaded file to PIL Image
    MainAPI->>Infer: predict_crop(image, direct_resize, top_k=5)
    
    alt direct_resize = false (Default)
        Infer->>Infer: Resize image to 256x256 -> CenterCrop to 240x240
    else direct_resize = true
        Infer->>Infer: Resize entire image directly to 240x240 (no crop)
    end
    
    Infer->>Infer: Normalize inputs to [-1, 1] range
    Infer->>Model1: Feed transformed tensor
    Model1-->>Infer: Raw logit outputs (32 classes)
    Infer->>Infer: Apply Softmax to get probabilities (%)
    Infer->>MainAPI: Return top-5 crop names + confidence percentages
    MainAPI->>MainAPI: Format response (human-friendly names)
    MainAPI-->>User: JSON Response (e.g. Predicted Crop: Tomato, 98%)
```

---

## 4. Phase 2: Disease Detection Flow (Masked Inference)

Called via `POST /api/v1/detect-disease`:

```mermaid
sequenceDiagram
    participant User
    participant MainAPI as main.py
    participant Infer as inference.py
    participant Model2 as EfficientNet-B0

    User->>MainAPI: POST /api/v1/detect-disease (image, crop='tomato', direct_resize=false/true)
    MainAPI->>MainAPI: Check if Phase 2 weights are active
    alt Phase 2 Not Active
        MainAPI-->>User: HTTP 503 (Model not loaded)
    end
    
    MainAPI->>MainAPI: Validate if crop is supported (check keys in crop_disease_idx_map.json)
    alt Crop not supported
        MainAPI-->>User: HTTP 400 (Bad Request: lists valid crops)
    end
    
    MainAPI->>MainAPI: Decode uploaded file to PIL Image
    MainAPI->>Infer: predict_disease(image, crop='tomato', direct_resize, top_k=5)
    
    alt direct_resize = false (Default)
        Infer->>Infer: Resize image to 256x256 -> CenterCrop to 224x224
    else direct_resize = true
        Infer->>Infer: Resize entire image directly to 224x224 (no crop)
    end
    Infer->>Infer: Normalize inputs to [-1, 1] range
    
    Infer->>Model2: Feed transformed tensor
    Model2-->>Infer: Raw logit outputs (100+ disease classes across all crops)
    
    Note over Infer: Index Masking:<br/>1. Look up valid tomato disease indices from crop_disease_idx_map.json.<br/>2. Set all other disease class logits to negative infinity (-inf).
    
    Infer->>Infer: Apply Softmax on masked logits (0% for other crops' diseases)
    Infer->>MainAPI: Return top disease labels + confidence percentages
    MainAPI->>MainAPI: Parse display name (e.g. 'tomato__early_blight' -> 'Early Blight')
    MainAPI->>MainAPI: Categorize confidence level (High, Moderate, Low)
    MainAPI-->>User: JSON Response (e.g. Disease: Early Blight, 93%, High Confidence)
```
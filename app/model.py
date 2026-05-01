import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Supported plant names (what the user can pass in)
# Keys are lowercase, normalised names the frontend/user can send.
# Values are the exact prefix strings used in DISEASE_CLASSES.
PLANT_NAME_MAP = {
    "apple":      "Apple",
    "blueberry":  "Blueberry",
    "cherry":     "Cherry_(including_sour)",
    "corn":       "Corn_(maize)",
    "maize":      "Corn_(maize)",
    "grape":      "Grape",
    "orange":     "Orange",
    "peach":      "Peach",
    "pepper":     "Pepper,_bell",
    "bell pepper":"Pepper,_bell",
    "potato":     "Potato",
    "raspberry":  "Raspberry",
    "soybean":    "Soybean",
    "squash":     "Squash",
    "strawberry": "Strawberry",
    "tomato":     "Tomato",
}

#  Class labels 
# These are the 38 folder names from the PlantVillage augmented dataset.
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(DISEASE_CLASSES)  # 38

# Preprocessing transform 
# The notebook uses ONLY transforms.ToTensor() for both train and inference.
# ToTensor() does two things: HxWxC uint8 → CxHxW float32, then divides by 255.
# The model was trained on 256×256 images (confirmed from torchsummary output).
# We replicate that exactly here.
INFERENCE_TRANSFORM = T.Compose([
    T.Resize((256, 256)),   # match training image size
    T.ToTensor(),           # → [0, 1] float32 CxHxW  (NO ImageNet normalisation)
])


# Model architecture 


def ConvBlock(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels: int, num_diseases: int):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)   # 128 x 64 x 64
        self.res1  = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)  # 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # 512 x  4 x  4
        self.res2  = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)


# Model loading 

def load_model(weights_path: str = "models/plant-disease-model-complete.pth") -> ResNet9:
    """
    Build the ResNet9 and load weights from disk.
    Falls back to random weights with a warning if the .pth file is not found
    (useful for smoke-testing the API without trained weights).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResNet9(in_channels=3, num_diseases=NUM_CLASSES)

    if os.path.exists(weights_path):
        # The checkpoint was saved via torch.save(model, ...) in a Kaggle notebook
        # where ResNet9 lived in __main__.  When pickle deserializes it, it looks for
        # ResNet9 in __main__ — but inside uvicorn's worker that module is __mp_main__
        # and the class is nowhere to be found.  Temporarily injecting our class into
        # sys.modules['__main__'] makes pickle resolve it correctly.
        sys.modules["__main__"].ResNet9 = ResNet9

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # Support both checkpoint formats:
        #   (a) full model object  → extract its state_dict
        #   (b) plain state_dict   → use directly
        if isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        print(f"[disease-service] ✅  Loaded weights from '{weights_path}' on {device}")
    else:
        print(
            f"[disease-service] ⚠️   Weights file '{weights_path}' not found. "
            "Running with random weights — predictions will be meaningless.\n"
            "Train the model using the Kaggle notebook and save as 'models/plant-disease-model-complete.pth'."
        )

    model.to(device)
    model.eval()
    return model


# Inference 

def preprocess(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image (any mode/size from the user) into the tensor the
    model expects: float32 RGB [0, 1] with shape (1, 3, 256, 256).

    Handles:
    - RGBA / palette images  → converted to RGB
    - Grayscale              → converted to RGB
    - Any size               → resized to 256×256
    - Any pixel range        → normalised to [0, 1] by ToTensor()
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = INFERENCE_TRANSFORM(image)   # (3, 256, 256)  float32 [0, 1]
    return tensor.unsqueeze(0)            # (1, 3, 256, 256)


def resolve_plant_prefix(plant_name: str) -> str | None:
    """
    Normalise a user-supplied plant name to its DISEASE_CLASSES prefix.
    Returns None if the name is not recognised.

    e.g.  "Tomato" → "Tomato"
          "maize"  → "Corn_(maize)"
          "banana" → None
    """
    return PLANT_NAME_MAP.get(plant_name.strip().lower())


def get_valid_plants() -> list[str]:
    """Return the list of accepted plant names for API docs / validation."""
    return sorted(set(PLANT_NAME_MAP.keys()))


@torch.no_grad()
def predict(image: Image.Image, model: ResNet9, plant_name: str | None = None) -> dict:
    """
    Run inference and return structured prediction dict.

    Args:
        image:      PIL Image from the user (any mode/size — preprocessed internally).
        model:      Loaded ResNet9.
        plant_name: Optional plant name supplied by the user (e.g. "tomato").
                    When provided, only that plant's disease classes are
                    considered, masking all others to -inf before softmax.
                    This prevents the model from predicting e.g. "Apple scab"
                    on a tomato leaf image.

    Returns:
        {
          "predicted_class":  "Tomato___Early_blight",
          "confidence":        0.9823,
          "plant":            "Tomato",
          "disease":          "Early blight",
          "is_healthy":       False,
          "plant_filter_used": True,
          "top5": [
              {"class": "Tomato___Early_blight",  "confidence": 0.9823},
              ...
          ]
        }
    """
    device = next(model.parameters()).device
    tensor = preprocess(image).to(device)

    logits = model(tensor)          # (1, 38)
    probs  = F.softmax(logits, dim=1)[0]   # (38,) — full distribution

    plant_filter_used = False
    allowed_indices   = list(range(NUM_CLASSES))  # default: all classes

    # Apply plant constraint if a valid plant name was given 
    if plant_name:
        prefix = resolve_plant_prefix(plant_name)
        if prefix:
            allowed_indices = [
                i for i, cls in enumerate(DISEASE_CLASSES)
                if cls.startswith(prefix)
            ]
            # Mask logits: set non-allowed classes to -inf, recompute softmax
            # so confidences are relative within the plant's own disease classes.
            mask = torch.full((NUM_CLASSES,), float("-inf"), device=device)
            mask[allowed_indices] = logits[0][allowed_indices]
            probs = F.softmax(mask.unsqueeze(0), dim=1)[0]
            plant_filter_used = True

    # Top-1 
    top1_idx   = probs.argmax().item()
    top1_class = DISEASE_CLASSES[top1_idx]
    top1_conf  = probs[top1_idx].item()

    # Top-5  (within allowed classes only) 
    k = min(5, len(allowed_indices))
    top5_vals, top5_idxs = torch.topk(probs, k=k)
    top5 = [
        {"class": DISEASE_CLASSES[i], "confidence": round(v, 4)}
        for i, v in zip(top5_idxs.tolist(), top5_vals.tolist())
        if v > float("-inf")
    ]

    # Parse plant / disease 
    parts   = top1_class.split("___", maxsplit=1)
    plant   = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "unknown"

    return {
        "predicted_class":   top1_class,
        "confidence":        round(top1_conf, 4),
        "plant":             plant,
        "disease":           disease,
        "is_healthy":        "healthy" in disease.lower(),
        "plant_filter_used": plant_filter_used,
        "top5":              top5,
    }

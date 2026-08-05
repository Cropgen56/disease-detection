import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import timm
from ultralytics import YOLO, YOLOE

logger = logging.getLogger(__name__)

# Device Configuration: Support MPS (Apple Silicon), CUDA (NVIDIA), and CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# DINOv2 uses upsample_bicubic2d which is not fully implemented on MPS.
# Force CPU for DINOv2 on Apple Silicon to avoid NotImplementedError.
DINOV2_DEVICE = torch.device("cpu") if DEVICE.type == "mps" else DEVICE

logger.info(f"Using device: {DEVICE}")
if DINOV2_DEVICE != DEVICE:
    logger.info(f"DINOv2 will run on: {DINOV2_DEVICE} (MPS fallback for unsupported ops)")

# Configuration constants matching training notebooks
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

DINOV2_NORM_MEAN = (0.485, 0.456, 0.406)
DINOV2_NORM_STD = (0.229, 0.224, 0.225)

# --- Preprocessing Transforms ---

def get_transforms(img_size: int, direct_resize: bool = False) -> A.Compose:
    """
    Builds the inference transform pipeline.
    
    - CenterCrop: Resizes to 256x256, then center crops to img_size.
    - DirectResize: Resizes directly to img_size.
    """
    if direct_resize:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_CUBIC),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ])


def get_dinov2_transforms(img_size: int = 238, direct_resize: bool = False) -> A.Compose:
    """
    Builds the inference transform pipeline for DINOv2.
    """
    if direct_resize:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=DINOV2_NORM_MEAN, std=DINOV2_NORM_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=img_size, interpolation=cv2.INTER_CUBIC),
            A.CenterCrop(height=img_size, width=img_size, pad_if_needed=True, border_mode=cv2.BORDER_REFLECT),
            A.Normalize(mean=DINOV2_NORM_MEAN, std=DINOV2_NORM_STD),
            ToTensorV2(),
        ])


class DinoV2Classifier(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = 'dinov2_vits14'):
        super().__init__()
        # Load from torch hub
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


# --- Model Management State ---

class ModelManager:
    def __init__(self, weights_dir: Path = Path("weights")):
        self.weights_dir = weights_dir

        # Stage 0: Leaf Gate (YOLOE open-vocabulary)
        self.stage0_model: Optional[YOLOE] = None
        self.stage0_config: Optional[dict] = None
        self._stage0_loaded: bool = False

        # Phase 1: Crop Classification
        self.phase1_model: Optional[nn.Module] = None
        self.phase1_yolo: Optional[YOLO] = None
        self.phase1_class_to_idx: Optional[Dict[str, int]] = None
        self.phase1_idx_to_class: Optional[Dict[int, str]] = None

        # Phase 2: Disease Detection
        self.phase2_model: Optional[nn.Module] = None
        self.phase2_class_to_idx: Optional[Dict[str, int]] = None
        self.phase2_idx_to_class: Optional[Dict[int, str]] = None
        self.crop_disease_idx_map: Optional[Dict[str, List[int]]] = None

    def is_stage0_loaded(self) -> bool:
        return self._stage0_loaded

    def is_phase1_loaded(self) -> bool:
        return (self.phase1_model is not None or self.phase1_yolo is not None) and self.phase1_idx_to_class is not None

    def is_phase2_loaded(self) -> bool:
        return self.phase2_model is not None and self.phase2_idx_to_class is not None and self.crop_disease_idx_map is not None

    def load_stage0(self) -> bool:
        """
        Load YOLOE model, pre-computed vocab embeddings, and stage0 config once at startup.

        Per the notebook instructions, set_classes() is called once here so it
        is NOT repeated per-request — it is relatively expensive.
        """
        model_path = self.weights_dir / "yoloe-11s-seg.pt"
        embeddings_path = self.weights_dir / "leaf_vocab_embeddings.pt"
        vocab_config_path = self.weights_dir / "vocab_config.json"
        stage0_config_path = self.weights_dir / "stage0_config.json"

        missing = [p for p in (model_path, embeddings_path, vocab_config_path, stage0_config_path) if not p.exists()]
        if missing:
            logger.warning(f"Stage 0 files not found: {[str(p) for p in missing]}")
            return False

        try:
            logger.info(f"Loading Stage 0 YOLOE model from {model_path}")
            model = YOLOE(str(model_path))

            with open(vocab_config_path) as f:
                vocab = json.load(f)["vocab"]

            # Load pre-computed text embeddings — avoids running get_text_pe() on every start
            text_pe = torch.load(str(embeddings_path), map_location="cpu")
            model.set_classes(vocab, text_pe)

            with open(stage0_config_path) as f:
                stage0_config = json.load(f)

            self.stage0_model = model
            self.stage0_config = stage0_config
            self._stage0_loaded = True
            logger.info("Stage 0 YOLOE leaf gate loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Stage 0 model: {e}")
            self._stage0_loaded = False
            return False

    def run_stage0(self, image_path: str) -> dict:
        """
        Run the Stage 0 leaf gate on a saved image file.

        Returns one of:
          {"status": "leaf", "crop": np.ndarray, "confidence": float, "box": [x1,y1,x2,y2]}
          {"status": "rejected", "detected_object": str, "confidence": float}
          {"status": "no_object_detected"}

        Raises RuntimeError if Stage 0 is not loaded.
        """
        if not self._stage0_loaded:
            raise RuntimeError("Stage 0 model is not loaded.")
        # Import here to avoid circular import at module level
        from stage0 import stage0_leaf_gate
        return stage0_leaf_gate(image_path, self.stage0_model, self.stage0_config)

    def load_phase1(self) -> bool:
        """Loads Phase 1 crop classifier model weights and mappings."""
        dinov2_path = self.weights_dir / "stage1_dinov2_30class.pt"
        effnet_path = self.weights_dir / "efficientnet_b1_crop_mini.pt"
        yolo_path = self.weights_dir / "yolo11n_crop_mini.pt"  # or user's best.pt renamed

        # Try DINOv2 first as the primary stage 1 model
        if dinov2_path.exists():
            try:
                logger.info(f"Loading Phase 1 DINOv2 model from {dinov2_path}")
                # Load to CPU first regardless; DINOv2 will be moved to DINOV2_DEVICE
                ckpt = torch.load(dinov2_path, map_location="cpu")
                
                if "class_to_idx" in ckpt:
                    self.phase1_class_to_idx = ckpt["class_to_idx"]
                else:
                    logger.error("No class_to_idx found inside the Phase 1 DINOv2 checkpoint file!")
                    return False
                
                self.phase1_idx_to_class = {int(v): k for k, v in self.phase1_class_to_idx.items()}
                
                # Overwrite class_to_idx_phase1.json so it matches the loaded model
                try:
                    with open(self.weights_dir / "class_to_idx_phase1.json", "w") as f:
                        json.dump(self.phase1_class_to_idx, f, indent=2)
                    logger.info("Successfully updated class_to_idx_phase1.json with checkpoint mapping.")
                except Exception as ex:
                    logger.warning(f"Could not update class_to_idx_phase1.json file: {ex}")

                num_classes = len(self.phase1_class_to_idx)
                model = DinoV2Classifier(num_classes=num_classes)
                
                state_dict = ckpt.get("model_state", ckpt)
                model.load_state_dict(state_dict)
                # Use DINOV2_DEVICE (CPU on MPS) — upsample_bicubic2d not supported on MPS
                model = model.to(DINOV2_DEVICE).eval()
                
                self.phase1_model = model
                self.phase1_yolo = None
                logger.info("Successfully loaded Phase 1 DINOv2 model")
                return True
            except Exception as e:
                logger.error(f"Failed to load Phase 1 DINOv2 model: {e}")

        # Fallback 1: EfficientNet-B1 Classifier
        # 1. Load class mapping for fallback models
        crop_class_map_path = self.weights_dir / "class_to_idx_phase1.json"
        if not crop_class_map_path.exists():
            logger.warning(f"Phase 1 class mapping missing at {crop_class_map_path}")
            return False
            
        try:
            with open(crop_class_map_path, "r") as f:
                self.phase1_class_to_idx = json.load(f)
            self.phase1_idx_to_class = {int(v) if str(v).isdigit() else v: k for k, v in self.phase1_class_to_idx.items()}
            # Ensure indices are integer keys
            self.phase1_idx_to_class = {int(k): v for k, v in self.phase1_idx_to_class.items()}
        except Exception as e:
            logger.error(f"Failed to parse Phase 1 class map: {e}")
            return False

        if effnet_path.exists():
            try:
                logger.info(f"Loading Phase 1 EfficientNet-B1 model from {effnet_path}")
                num_classes = len(self.phase1_class_to_idx)
                model = timm.create_model("efficientnet_b1", pretrained=False, num_classes=num_classes, drop_rate=0.0)
                
                ckpt = torch.load(effnet_path, map_location=DEVICE)
                state_dict = ckpt.get("model_state", ckpt)
                model.load_state_dict(state_dict)
                model = model.to(DEVICE).eval()
                
                self.phase1_model = model
                self.phase1_yolo = None
                logger.info("Successfully loaded Phase 1 EfficientNet-B1 model")
                return True
            except Exception as e:
                logger.error(f"Failed to load Phase 1 EfficientNet model: {e}")
                
        if yolo_path.exists() or (self.weights_dir / "best.pt").exists():
            actual_yolo_path = yolo_path if yolo_path.exists() else (self.weights_dir / "best.pt")
            try:
                logger.info(f"Loading Phase 1 YOLOv11 classifier from {actual_yolo_path}")
                # ultralytics handles model loading and device selection internally
                self.phase1_yolo = YOLO(str(actual_yolo_path))
                self.phase1_model = None
                # YOLO holds its own names dictionary, but we align with class_to_idx
                logger.info("Successfully loaded Phase 1 YOLOv11 classifier")
                return True
            except Exception as e:
                logger.error(f"Failed to load Phase 1 YOLO model: {e}")

        logger.warning("No Phase 1 model weights (.pt) found in weights/ directory.")
        return False

    def load_phase2(self) -> bool:
        """Loads Phase 2 disease detector model weights and mappings."""
        # 1. Load crop-disease mapping JSONs
        idx_map_path = self.weights_dir / "crop_disease_idx_map.json"
        
        if not idx_map_path.exists():
            logger.warning(f"Phase 2 crop-disease mapping missing at {idx_map_path}")
            return False
            
        try:
            with open(idx_map_path, "r") as f:
                # Convert keys to string and lists to list of integers
                self.crop_disease_idx_map = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse crop_disease_idx_map: {e}")
            return False

        # 2. Check and load model weights
        disease_pt_path = self.weights_dir / "efficientnet_b0_disease_mini.pt"
        if not disease_pt_path.exists():
            logger.warning(f"Phase 2 model checkpoint missing at {disease_pt_path}")
            return False

        try:
            logger.info(f"Loading Phase 2 EfficientNet-B0 model from {disease_pt_path}")
            ckpt = torch.load(disease_pt_path, map_location=DEVICE)
            
            # Load class map bundled in the checkpoint
            self.phase2_class_to_idx = ckpt.get("class_to_idx")
            if not self.phase2_class_to_idx:
                logger.error("No class_to_idx found inside the Phase 2 checkpoint file!")
                return False
                
            self.phase2_idx_to_class = {int(v): k for k, v in self.phase2_class_to_idx.items()}
            num_classes = len(self.phase2_class_to_idx)
            
            model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes, drop_rate=0.0)
            model.load_state_dict(ckpt["model_state"])
            model = model.to(DEVICE).eval()
            
            self.phase2_model = model
            logger.info(f"Successfully loaded Phase 2 model with {num_classes} classes")
            return True
        except Exception as e:
            logger.error(f"Failed to load Phase 2 model: {e}")
            return False

    def predict_crop(self, img_pil: Image.Image, direct_resize: bool = False, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predicts the crop class of the given image."""
        if not self.is_phase1_loaded():
            raise RuntimeError("Phase 1 model weights or metadata are not loaded.")

        # Scenario 1: PyTorch nn.Module Classifier (EfficientNet-B1 or DINOv2)
        if self.phase1_model is not None:
            if isinstance(self.phase1_model, DinoV2Classifier):
                transform = get_dinov2_transforms(img_size=238, direct_resize=direct_resize)
                # DINOv2 runs on DINOV2_DEVICE (CPU on MPS)
                infer_device = DINOV2_DEVICE
            else:
                transform = get_transforms(img_size=240, direct_resize=direct_resize)
                infer_device = DEVICE
            img_arr = np.array(img_pil.convert("RGB"))
            tensor = transform(image=img_arr)["image"].unsqueeze(0).to(infer_device)
            
            with torch.no_grad():
                logits = self.phase1_model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu()
                
            top_probs, top_idxs = probs.topk(min(top_k, len(probs)))
            results = []
            for idx, prob in zip(top_idxs, top_probs):
                idx_val = idx.item()
                crop_name = self.phase1_idx_to_class.get(idx_val, f"unknown_{idx_val}")
                results.append((crop_name, prob.item()))
            return results

        # Scenario 2: YOLOv11 Classifier
        elif self.phase1_yolo is not None:
            # YOLO internally handles loading PIL image and preprocessing
            results = self.phase1_yolo(img_pil, verbose=False)[0]
            probs = results.probs
            
            # YOLO sorted alphabetically, we can extract it directly
            top_indices = probs.top5[:top_k]
            top_confs = probs.top5conf[:top_k].tolist()
            
            # Retrieve names from YOLO names mapping or align with class_to_idx
            yolo_names = self.phase1_yolo.names
            
            output = []
            for idx, conf in zip(top_indices, top_confs):
                idx_val = int(idx)
                crop_name = yolo_names.get(idx_val, f"unknown_{idx_val}")
                output.append((crop_name, conf))
            return output

        raise RuntimeError("No loaded Phase 1 model structure was identified.")

    def predict_disease(self, img_pil: Image.Image, crop_name: str, direct_resize: bool = False, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predicts the disease of the crop from valid disease classes, applying index masking."""
        if not self.is_phase2_loaded():
            raise RuntimeError("Phase 2 model weights or metadata are not loaded.")
            
        crop_clean = crop_name.lower().strip()
        if crop_clean not in self.crop_disease_idx_map:
            raise ValueError(f"Crop '{crop_name}' is not supported in the disease database.")
            
        valid_idxs = self.crop_disease_idx_map[crop_clean]
        if not valid_idxs:
            raise ValueError(f"No valid diseases registered for crop '{crop_name}'.")

        # 1. Preprocess the image (EfficientNet-B0 uses IMG_SIZE = 224)
        transform = get_transforms(img_size=224, direct_resize=direct_resize)
        img_arr = np.array(img_pil.convert("RGB"))
        tensor = transform(image=img_arr)["image"].unsqueeze(0).to(DEVICE)

        # 2. Run forward pass
        with torch.no_grad():
            logits = self.phase2_model(tensor)[0]

        # 3. Mask out all classes not valid for this crop (set to -infinity)
        valid_idxs_t = torch.tensor(valid_idxs, device=DEVICE)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits[valid_idxs_t] = logits[valid_idxs_t]

        # 4. Compute Softmax and get Top K predictions
        probs = torch.softmax(masked_logits, dim=0).cpu()
        top_probs, top_idxs = probs.topk(min(top_k, len(valid_idxs)))

        # 5. Format results
        results = []
        for idx, prob in zip(top_idxs, top_probs):
            idx_val = idx.item()
            disease_label = self.phase2_idx_to_class.get(idx_val, f"unknown_{idx_val}")
            results.append((disease_label, prob.item()))
            
        return results

    def get_supported_crops(self) -> List[str]:
        """Returns the list of supported crops for Phase 2."""
        if self.crop_disease_idx_map is not None:
            return sorted(list(self.crop_disease_idx_map.keys()))
        return []

    def get_diseases_for_crop(self, crop_name: str) -> List[str]:
        """Returns the list of valid diseases for a given crop name."""
        if not self.is_phase2_loaded():
            return []
        
        crop_clean = crop_name.lower().strip()
        valid_idxs = self.crop_disease_idx_map.get(crop_clean, [])
        
        diseases = []
        for idx in valid_idxs:
            d_name = self.phase2_idx_to_class.get(idx)
            if d_name:
                diseases.append(d_name)
        return sorted(diseases)

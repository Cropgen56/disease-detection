
import numpy as np
from PIL import Image



def resolve_call(detections, leaf_terms, threshold):
    """Leaf wins if ANY leaf-term detection clears the threshold, regardless of rank."""
    above_threshold = [d for d in detections if d['confidence'] >= threshold]
    if not above_threshold:
        return None  # no_object_detected

    leaf_matches = [d for d in above_threshold if d['class_name'] in leaf_terms]
    if leaf_matches:
        return max(leaf_matches, key=lambda d: d['confidence'])

    return max(above_threshold, key=lambda d: d['confidence'])



def stage0_leaf_gate(image_path, model, config):
    """
    Runs Stage 0 leaf detection on an image.

    Returns one of:
      {"status": "leaf", "crop": np.ndarray, "confidence": float, "box": [x1,y1,x2,y2]}
      {"status": "rejected", "detected_object": str, "confidence": float}
      {"status": "no_object_detected"}
    """
    conf_threshold = config['conf_threshold']
    leaf_terms = set(config['leaf_terms'])
    min_area_ratio = config['min_box_area_ratio']
    padding_pct = config['crop_padding_pct']

    img = np.array(Image.open(image_path).convert('RGB'))
    h, w = img.shape[:2]
    img_area = h * w

    results = model.predict(image_path, conf=0.01, verbose=False)  # low conf, filter via resolve_call
    r = results[0]

    if len(r.boxes) == 0:
        return {"status": "no_object_detected"}

    candidates = []
    for box in r.boxes:
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        xyxy = box.xyxy.cpu().numpy()[0].tolist()
        x1, y1, x2, y2 = xyxy
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / img_area

        if area_ratio < min_area_ratio:
            continue

        candidates.append({
            'class_name': model.names[cls_id],
            'confidence': conf,
            'box': xyxy,
        })

    if not candidates:
        return {"status": "no_object_detected"}

    top = resolve_call(candidates, leaf_terms, conf_threshold)

    if top is None:
        return {"status": "no_object_detected"}

    if top['class_name'] not in leaf_terms:
        return {
            "status": "rejected",
            "detected_object": top['class_name'],
            "confidence": round(top['confidence'], 4),
        }

    x1, y1, x2, y2 = top['box']
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = box_w * padding_pct, box_h * padding_pct

    x1p = max(0, int(x1 - pad_x))
    y1p = max(0, int(y1 - pad_y))
    x2p = min(w, int(x2 + pad_x))
    y2p = min(h, int(y2 + pad_y))

    crop = img[y1p:y2p, x1p:x2p]

    return {
        "status": "leaf",
        "crop": crop,
        "confidence": round(top['confidence'], 4),
        "box": [x1p, y1p, x2p, y2p],
    }

"""
diagnose.py — Full pipeline diagnostic script
Usage:
    python diagnose.py <image_path> [--lang en|hi] [--base-url http://localhost:8000]

Runs all three API calls in sequence:
  1. /api/v1/classify-crop    → detect which crop
  2. /api/v1/detect-disease   → detect which disease
  3. /api/v1/symptoms-control → fetch symptoms & control measures
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# ── ANSI colours (auto-disabled if not a TTY) ────────────────────────────────
def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

BOLD   = "\033[1m"   if _supports_colour() else ""
GREEN  = "\033[92m"  if _supports_colour() else ""
YELLOW = "\033[93m"  if _supports_colour() else ""
CYAN   = "\033[96m"  if _supports_colour() else ""
RED    = "\033[91m"  if _supports_colour() else ""
DIM    = "\033[2m"   if _supports_colour() else ""
RESET  = "\033[0m"   if _supports_colour() else ""


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post_multipart(url: str, image_path: Path) -> dict:
    """POST image as multipart/form-data and return parsed JSON body."""
    boundary = "----PythonDiagnoseBoundary"
    file_bytes = image_path.read_bytes()
    mime_type = (
        "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    )

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _get(url: str, params: dict) -> dict:
    """GET with query params and return parsed JSON body."""
    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full_url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _handle_http_error(step: str, e: urllib.error.HTTPError) -> None:
    """Print a readable error and exit."""
    try:
        body = json.loads(e.read().decode())
        detail = body.get("detail", "(no detail)")
    except Exception:
        detail = "(could not parse error body)"
    print(f"\n{RED}✗ {step} failed — HTTP {e.code}: {detail}{RESET}")
    sys.exit(1)


# ── Pretty printers ───────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 50}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 50}{RESET}")


def _print_kv(key: str, value: str, colour: str = "") -> None:
    print(f"  {DIM}{key:<20}{RESET}{colour}{value}{RESET}")


def _print_list(items: list[str]) -> None:
    for item in items:
        print(f"    {DIM}•{RESET} {item}")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_classify_crop(base_url: str, image_path: Path) -> str:
    """Returns the top predicted crop name (e.g. 'tomato')."""
    _print_header("Step 1 — Crop Classification")
    print(f"  {DIM}Sending image to {base_url}/api/v1/classify-crop …{RESET}")

    try:
        result = _post_multipart(f"{base_url}/api/v1/classify-crop", image_path)
    except urllib.error.HTTPError as e:
        _handle_http_error("Crop classification", e)

    crop = result["predicted_crop"]
    display = result["display_name"]
    confidence = result["confidence"]

    _print_kv("Crop detected:", f"{display}  ({crop})", GREEN)
    _print_kv("Confidence:", f"{confidence * 100:.1f}%", GREEN)

    print(f"\n  {DIM}Top predictions:{RESET}")
    for pred in result.get("all_predictions", []):
        bar = "█" * int(pred["confidence"] * 20)
        print(
            f"    {pred['display_name']:<25} {bar:<20} {pred['confidence']*100:.1f}%"
        )

    return crop


def step2_detect_disease(base_url: str, image_path: Path, crop: str) -> str:
    """Returns the top predicted disease display name (e.g. 'Early Blight')."""
    _print_header("Step 2 — Disease Detection")
    print(f"  {DIM}Sending image to {base_url}/api/v1/detect-disease?crop={crop} …{RESET}")

    url = f"{base_url}/api/v1/detect-disease?crop={urllib.parse.quote(crop)}"
    try:
        result = _post_multipart(url, image_path)
    except urllib.error.HTTPError as e:
        _handle_http_error("Disease detection", e)

    disease = result["predicted_disease"]
    confidence = result["confidence"]
    confidence_level = result["confidence_level"]

    _print_kv("Disease:", disease, YELLOW)
    _print_kv("Confidence:", f"{confidence * 100:.1f}%  [{confidence_level}]", YELLOW)

    print(f"\n  {DIM}Top predictions:{RESET}")
    for pred in result.get("top_k_predictions", []):
        bar = "█" * int(pred["confidence"] * 20)
        print(
            f"    {pred['disease']:<35} {bar:<20} {pred['confidence']*100:.1f}%"
        )

    return disease


def step3_symptoms_control(
    base_url: str, crop: str, disease: str, lang: str
) -> None:
    """Fetches and prints symptoms & control measures."""
    _print_header(
        f"Step 3 — Symptoms & Control  [lang={lang}]"
    )
    print(
        f"  {DIM}Fetching from {base_url}/api/v1/symptoms-control …{RESET}"
    )

    try:
        result = _get(
            f"{base_url}/api/v1/symptoms-control",
            {"crop": crop, "disease": disease, "lang": lang},
        )
    except urllib.error.HTTPError as e:
        _handle_http_error("Symptoms & control lookup", e)

    disease_label = result.get("disease_hi") or result.get("disease_en", disease)
    print(f"\n  {BOLD}Disease:{RESET} {disease_label}\n")

    print(f"  {BOLD}Symptoms:{RESET}")
    _print_list(result.get("symptoms", []))

    print(f"\n  {BOLD}Control Measures:{RESET}")
    _print_list(result.get("control", []))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline: classify crop → detect disease → symptoms & control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python diagnose.py test_images/tomato_leaf.jpg\n"
            "  python diagnose.py test_images/leaf.png --lang hi\n"
            "  python diagnose.py test_images/leaf.jpg --base-url http://192.168.1.5:8000\n"
        ),
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the leaf image (jpg/png).",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "hi"],
        default="en",
        help="Language for symptoms & control output (default: en).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running FastAPI server (default: http://localhost:8000).",
    )

    args = parser.parse_args()

    image_path: Path = args.image
    if not image_path.exists():
        print(f"{RED}Error: image file not found: {image_path}{RESET}")
        sys.exit(1)
    if not image_path.is_file():
        print(f"{RED}Error: path is not a file: {image_path}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}Diagnosing image:{RESET} {image_path}")
    print(f"{BOLD}Server:{RESET}          {args.base_url}")
    print(f"{BOLD}Language:{RESET}        {args.lang}")

    # Step 1: Crop classification
    crop = step1_classify_crop(args.base_url, image_path)

    # Step 2: Disease detection
    disease = step2_detect_disease(args.base_url, image_path, crop)

    # Step 3: Symptoms & control
    step3_symptoms_control(args.base_url, crop, disease, args.lang)

    print(f"\n{BOLD}{GREEN}✓ Diagnosis complete.{RESET}\n")


if __name__ == "__main__":
    main()

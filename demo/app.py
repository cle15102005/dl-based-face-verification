"""
Web-based face verification demo using Flask.

Runs a local server accessible from any browser (including Windows browser
when the server is running in WSL). Webcam is accessed via browser API.

Usage:
    pip install flask
    python demo/app.py
    Open http://localhost:5000 in browser
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from facenet_pytorch import MTCNN

from src.models.backbone import InceptionResnetV1Backbone
from src.models.siamese import SiameseNetwork
from src.utils.paths import CHECKPOINTS_DIR, METRICS_DIR

app = Flask(__name__, template_folder="templates")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MTCNN — same settings as mtcnn_preprocess.py so webcam crops match training data
_mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=False,
    device=_device,
)

# Loaded models cache: { "arcface": {...}, "baseline": {...} }
_models = {}
_enrolled_embedding = None


def load_one_model(checkpoint_name: str):
    """Load a single model by checkpoint name and cache it."""
    if checkpoint_name in _models:
        return _models[checkpoint_name]

    config_path = PROJECT_ROOT / "configs" / f"{checkpoint_name}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    loss_type = config.get("loss_type", "arcface")
    backbone = InceptionResnetV1Backbone(
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
        pretrained=None,
    )
    if loss_type == "contrastive":
        model = SiameseNetwork(backbone).to(_device)
        use_siamese = True
    else:
        model = backbone.to(_device)
        use_siamese = False

    ckpt = CHECKPOINTS_DIR / f"{checkpoint_name}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=_device), strict=False)
    model.eval()

    # Use EER threshold from LFW evaluation directly
    threshold = 0.5
    csv_path = METRICS_DIR / "comparison_table.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        row = df[df["checkpoint"].str.contains(checkpoint_name)]
        if not row.empty:
            threshold = float(row.iloc[0]["eer_threshold"])

    input_size = config.get("input_size", 160)
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    entry = {
        "model": model,
        "use_siamese": use_siamese,
        "transform": transform,
        "threshold": threshold,
        "loss_type": loss_type,
    }
    _models[checkpoint_name] = entry
    print(f"[INFO] Loaded {checkpoint_name}.pt | loss={loss_type} | threshold={threshold:.4f}")
    return entry


def load_all_models():
    """Pre-load all available checkpoints."""
    for name in ("arcface", "baseline"):
        ckpt = CHECKPOINTS_DIR / f"{name}.pt"
        if ckpt.exists():
            try:
                load_one_model(name)
            except Exception as e:
                print(f"[WARN] Could not load {name}: {e}")


def decode_image(data_url: str) -> Image.Image:
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def detect_and_crop(img: Image.Image):
    """
    Run MTCNN on the image.
    Returns a cropped PIL face image (160px) if a human face is found, else None.
    """
    face_tensor = _mtcnn(img)   # returns (C, H, W) tensor [0,255] or None
    if face_tensor is None:
        return None
    # Convert tensor [0,255] → PIL image
    face_np = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
    return Image.fromarray(face_np)


def pil_to_b64(img: Image.Image) -> str:
    """Encode a PIL image to a base64 JPEG data-URL."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def get_embedding(img: Image.Image, entry: dict):
    """
    Detect face with MTCNN, then extract embedding.
    Returns (embedding_tensor, face_pil, None) on success,
            (None, None, error_str) if no face detected.
    """
    face_img = detect_and_crop(img)
    if face_img is None:
        return None, None, "no_face"

    tensor = entry["transform"](face_img).unsqueeze(0).to(_device)
    with torch.no_grad():
        if entry["use_siamese"]:
            emb = entry["model"].backbone(tensor)
        else:
            emb = entry["model"](tensor)
    return emb, face_img, None


@app.route("/")
def index():
    available = [n for n in ("arcface", "baseline") if (CHECKPOINTS_DIR / f"{n}.pt").exists()]
    return render_template("index.html", available_models=available, device=str(_device))


@app.route("/model_info", methods=["GET"])
def model_info():
    name = request.args.get("model", "arcface")
    try:
        entry = load_one_model(name)
        return jsonify({"threshold": entry["threshold"], "loss_type": entry["loss_type"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/enroll", methods=["POST"])
def enroll():
    global _enrolled_embedding
    data = request.json
    images = data.get("images") or ([data.get("image")] if data.get("image") else [])
    model_name = data.get("model", "arcface")
    if not images:
        return jsonify({"error": "No image provided"}), 400

    entry = load_one_model(model_name)
    embeddings = []
    face_crops_b64 = []
    for img_data in images:
        img = decode_image(img_data)
        emb, face_img, err = get_embedding(img, entry)
        if err == "no_face":
            continue
        embeddings.append(emb)
        face_crops_b64.append(pil_to_b64(face_img))

    if not embeddings:
        return jsonify({"error": "No human face detected in any capture. Please try again."}), 400

    avg_emb = torch.stack(embeddings).mean(dim=0)
    avg_emb = F.normalize(avg_emb, p=2, dim=1)
    _enrolled_embedding = (avg_emb, model_name)
    return jsonify({"status": "enrolled", "captures": len(embeddings), "face_crops": face_crops_b64})


@app.route("/verify", methods=["POST"])
def verify():
    global _enrolled_embedding
    if _enrolled_embedding is None:
        return jsonify({"error": "No enrolled face. Please enroll first."}), 400
    data = request.json
    img_data = data.get("image")
    model_name = data.get("model", "arcface")
    if not img_data:
        return jsonify({"error": "No image provided"}), 400

    enrolled_emb, enrolled_model = _enrolled_embedding
    if enrolled_model != model_name:
        return jsonify({"error": "Model changed since enrollment. Please re-enroll."}), 400

    entry = load_one_model(model_name)
    img = decode_image(img_data)
    probe_emb, probe_face, err = get_embedding(img, entry)
    if err == "no_face":
        return jsonify({"error": "No human face detected. Please try again."}), 400

    client_threshold = data.get("threshold")
    threshold = float(client_threshold) if client_threshold is not None else entry["threshold"]

    score = F.cosine_similarity(enrolled_emb, probe_emb).item()
    match = score >= threshold

    return jsonify({
        "score": round(score, 4),
        "threshold": round(threshold, 4),
        "match": match,
        "face_crop": pil_to_b64(probe_face),
    })


@app.route("/reset", methods=["POST"])
def reset():
    global _enrolled_embedding
    _enrolled_embedding = None
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    load_all_models()
    print(f"[INFO] Open http://localhost:{args.port} in your browser")
    app.run(host="0.0.0.0", port=args.port, debug=False)

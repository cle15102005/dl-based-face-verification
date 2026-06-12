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

from src.models.backbone import InceptionResnetV1Backbone
from src.models.siamese import SiameseNetwork
from src.utils.paths import CHECKPOINTS_DIR, METRICS_DIR

app = Flask(__name__, template_folder="templates")

# Global model state
_model = None
_use_siamese = False
_transform = None
_threshold = 0.5
_model_name = ""
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per-session enrolled embedding
_enrolled_embedding = None


def load_model():
    global _model, _use_siamese, _transform, _threshold, _model_name

    csv_path = METRICS_DIR / "comparison_table.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        best = df.loc[df["eer"].idxmin()]
        checkpoint_name = Path(best["checkpoint"]).stem
        loss_type = best["loss_type"]
        _threshold = float(best["eer_threshold"])
    else:
        checkpoint_name = "arcface"
        loss_type = "arcface"
        _threshold = 0.5

    config_path = PROJECT_ROOT / "configs" / f"{checkpoint_name}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    backbone = InceptionResnetV1Backbone(
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
        pretrained=None,
    )
    if loss_type == "contrastive":
        _model = SiameseNetwork(backbone).to(_device)
        _use_siamese = True
    else:
        _model = backbone.to(_device)
        _use_siamese = False

    ckpt = CHECKPOINTS_DIR / f"{checkpoint_name}.pt"
    _model.load_state_dict(torch.load(ckpt, map_location=_device), strict=False)
    _model.eval()
    _model_name = f"{checkpoint_name}.pt"

    input_size = config.get("input_size", 160)
    _transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"[INFO] Model loaded: {_model_name} | threshold={_threshold:.4f} | device={_device}")


def decode_image(data_url: str) -> Image.Image:
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def get_embedding(img: Image.Image) -> torch.Tensor:
    tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        if _use_siamese:
            return _model.backbone(tensor)
        return _model(tensor)


@app.route("/")
def index():
    return render_template("index.html", model_name=_model_name, threshold=_threshold)


@app.route("/enroll", methods=["POST"])
def enroll():
    global _enrolled_embedding
    data = request.json.get("image")
    if not data:
        return jsonify({"error": "No image provided"}), 400
    img = decode_image(data)
    _enrolled_embedding = get_embedding(img)
    return jsonify({"status": "enrolled"})


@app.route("/verify", methods=["POST"])
def verify():
    global _enrolled_embedding
    if _enrolled_embedding is None:
        return jsonify({"error": "No enrolled face. Please enroll first."}), 400
    data = request.json.get("image")
    if not data:
        return jsonify({"error": "No image provided"}), 400

    img = decode_image(data)
    probe_emb = get_embedding(img)
    score = F.cosine_similarity(_enrolled_embedding, probe_emb).item()
    match = score >= _threshold

    return jsonify({
        "score": round(score, 4),
        "threshold": round(_threshold, 4),
        "match": match,
    })


@app.route("/reset", methods=["POST"])
def reset():
    global _enrolled_embedding
    _enrolled_embedding = None
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    load_model()
    print("[INFO] Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False)

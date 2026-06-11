"""
Live webcam face verification demo.

Student exam authentication scenario:
  1. Enroll: capture registered student face -> extract embedding.
  2. Verify: capture exam face -> extract embedding.
  3. Decision: cosine similarity vs EER threshold -> MATCH or REJECT.

Loads the best model (lowest EER) from comparison_table.csv automatically.
Falls back to arcface_s.pt if evaluation has not been run yet.

Usage:
    python demo/verify.py
    python demo/verify.py --threshold 0.45

Controls: SPACE = capture frame, ESC = cancel
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import EfficientNetV2Backbone
from src.models.siamese import SiameseNetwork
from src.utils.paths import CHECKPOINTS_DIR, METRICS_DIR, VARIANT_INPUT_SIZES


def load_best_model(device):
    """
    Load the best-performing model for the demo.

    Steps:
      1. Read comparison_table.csv and select lowest EER model.
      2. Load YAML config for architecture hyperparameters.
      3. Build and load model weights.

    Fallback: arcface_s.pt with threshold=0.5 if comparison_table.csv is missing.
    """
    csv_path = METRICS_DIR / "comparison_table.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        best_row = df.loc[df["eer"].idxmin()]
        checkpoint = Path(best_row["checkpoint"])
        loss_type = best_row["loss_type"]
        variant = best_row["variant"]
        threshold = float(best_row["eer_threshold"])
    else:
        checkpoint = CHECKPOINTS_DIR / "arcface_s.pt"
        loss_type = "arcface"
        variant = "s"
        threshold = 0.5
        print("[WARN] comparison_table.csv not found. Falling back to arcface_s.pt with threshold=0.5")

    config_path = PROJECT_ROOT / "configs" / f"{checkpoint.stem}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_size = VARIANT_INPUT_SIZES[variant]
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    backbone = EfficientNetV2Backbone(
        variant=variant,
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
    )
    if loss_type == "contrastive":
        model = SiameseNetwork(backbone).to(device)
    else:
        model = backbone.to(device)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model, loss_type, transform, threshold, checkpoint.name


def capture_face(cap, window_name, transform, device, model, loss_type):
    """
    Capture a face from webcam and extract its embedding.

    SPACE = capture current frame, ESC = cancel.
    """
    print(f"[INFO] Press SPACE to capture for '{window_name}', ESC to cancel.")
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read from webcam.")

        display = frame.copy()
        cv2.putText(display, window_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Face Verification Demo", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None
        if key == 32:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                if loss_type == "contrastive":
                    embedding = model.backbone(tensor)
                else:
                    embedding = model(tensor)
            return embedding


def cosine_similarity(emb_a, emb_b):
    return torch.nn.functional.cosine_similarity(emb_a, emb_b).item()


def run_demo(threshold_override=None):
    """
    Run enrollment + verification demo.

    Steps:
      1. Load best model and EER threshold.
      2. Enroll reference face from webcam.
      3. Capture probe face from webcam.
      4. Compare cosine similarity vs threshold -> MATCH or REJECT.
      5. Display result on screen.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loss_type, transform, threshold, model_name = load_best_model(device)
    if threshold_override is not None:
        threshold = threshold_override

    print(f"[INFO] Loaded model: {model_name} | Threshold: {threshold:.4f}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        print("[STEP 1] Enroll registered student face.")
        enrolled = capture_face(cap, "ENROLL - Press SPACE", transform, device, model, loss_type)
        if enrolled is None:
            print("[INFO] Enrollment cancelled.")
            return

        print("[STEP 2] Capture exam verification face.")
        probe = capture_face(cap, "VERIFY - Press SPACE", transform, device, model, loss_type)
        if probe is None:
            print("[INFO] Verification cancelled.")
            return

        score = cosine_similarity(enrolled, probe)
        decision = "MATCH" if score >= threshold else "REJECT"
        color = (0, 200, 0) if decision == "MATCH" else (0, 0, 255)

        print(f"\n{'=' * 40}")
        print(f"  Similarity : {score:.4f}")
        print(f"  Threshold  : {threshold:.4f}")
        print(f"  Decision   : {decision}")
        print(f"{'=' * 40}\n")

        result_frame = np.zeros((200, 640, 3), dtype=np.uint8)
        cv2.putText(result_frame, f"Score: {score:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Threshold: {threshold:.3f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result_frame, decision, (420, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow("Face Verification Demo", result_frame)
        cv2.waitKey(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live webcam face verification demo.")
    parser.add_argument("--threshold", type=float, default=None, help="Override similarity threshold.")
    args = parser.parse_args()
    run_demo(threshold_override=args.threshold)

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_utils
from tqdm import tqdm

from src.models.unet_plus_plus import UNetPlusPlus


def predict(model, image, size, device):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    x = image.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def postprocess(prob, threshold, min_area):
    binary = (prob >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    mask = np.zeros_like(binary)

    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == label] = 1

    return mask


def load_model(checkpoint, features, device):
    model = UNetPlusPlus(features=tuple(features)).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)

    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--size_a", type=int, required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--size_b", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--output", default="/kaggle/working/outputs/submissioncsv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(args.data_root)
    test_dir = root / "test" / "test_images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = cfg["model"]["features"]

    model_a = load_model(args.checkpoint_a, features, device)
    model_b = load_model(args.checkpoint_b, features, device)

    rows = []

    for path in tqdm(sorted(test_dir.glob("*.jpeg")), desc="Predicting"):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        prob_a = predict(model_a, image, args.size_a, device)
        prob_b = predict(model_b, image, args.size_b, device)

        prob_a = cv2.resize(prob_a, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_b = cv2.resize(prob_b, (w, h), interpolation=cv2.INTER_LINEAR)

        prob = (prob_a + prob_b) / 2.0
        binary = postprocess(prob, args.threshold, args.min_area)

        n, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)

        component_id = 0
        for label in range(1, n):
            component_id += 1
            component = (labels == label).astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(component))
            counts = rle["counts"]

            if isinstance(counts, bytes):
                counts = counts.decode("utf-8")

            rows.append({
                "filament_id": f"{path.stem}_{component_id}",
                "segmentation_rle": counts,
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        rows,
        columns=["filament_id", "segmentation_rle"]
    ).to_csv(output, index=False)

    print(f"Saved {len(rows)} predicted filaments to {output}")


if __name__ == "__main__":
    main()
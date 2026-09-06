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


def predict_tta(model, image, size, device):
    probs = [predict(model, image, size, device)]
    transforms = [
        lambda x: np.fliplr(x).copy(),
        lambda x: np.flipud(x).copy(),
        lambda x: np.flipud(np.fliplr(x)).copy(),
    ]
    for i, transform in enumerate(transforms):
        prob = predict(model, transform(image), size, device)
        if i == 0:
            prob = np.fliplr(prob)
        elif i == 1:
            prob = np.flipud(prob)
        else:
            prob = np.flipud(np.fliplr(prob))
        probs.append(prob)
    return np.mean(probs, axis=0)


def postprocess(prob, threshold, min_area):
    binary = (prob >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    # 8 means 8-connectivity
    mask = np.zeros_like(binary)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == label] = 1
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--min_area", type=int, default=50)
    parser.add_argument("--output", default="/kaggle/working/outputs/submission.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(args.data_root)
    test_dir = root / "test" / "test_images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg["data"]["image_size"]

    model = UNetPlusPlus(features=tuple(cfg["model"]["features"])).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    for path in tqdm(sorted(test_dir.glob("*.jpeg")), desc="Predicting"):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        prob = predict_tta(model, image, size, device)
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
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
            rows.append({"filament_id": f"{path.stem}_{component_id}", "segmentation_rle": counts})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(output, index=False)
    print(f"Saved {len(rows)} predicted filaments to {output}")


if __name__ == "__main__":
    main()
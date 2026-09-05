"""Evaluate threshold and minimum-area post-processing."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.data.dataset import load_records, make_mask
from src.models.unet_plus_plus import UNetPlusPlus


def predict_image(model, path, image_size, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)

    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    x = image.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def postprocess(prob, threshold, min_area):
    binary = (prob >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    mask = np.zeros_like(binary)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == label] = 1

    return mask


def calc_metrics(pred, target):
    inter = np.logical_and(pred, target).sum()
    pred_area = pred.sum()
    target_area = target.sum()

    dice = (2 * inter + 1e-7) / (pred_area + target_area + 1e-7)
    union = pred_area + target_area - inter
    iou = (inter + 1e-7) / (union + 1e-7)

    return dice, iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--min_areas", nargs="+", type=int, default=[50, 100, 200, 500])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["data"]["image_size"]
    features = tuple(cfg["model"]["features"])

    _, val_records = load_records(
        args.data_root,
        cfg["data"]["val_ratio"],
        cfg["seed"],
    )

    model = UNetPlusPlus(features=features).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)

    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    results = []

    for threshold in args.thresholds:
        for min_area in args.min_areas:
            dices, ious = [], []

            for record in tqdm(val_records, leave=False):
                prob = predict_image(model, record["image_path"], image_size, device)
                pred = postprocess(prob, threshold, min_area)
                target = make_mask(record["annotations"], record["height"], record["width"])

                dice, iou = calc_metrics(pred, target)
                dices.append(dice)
                ious.append(iou)

            results.append({
                "threshold": threshold,
                "min_area": min_area,
                "dice": np.mean(dices),
                "iou": np.mean(ious),
            })

    results.sort(key=lambda x: x["dice"], reverse=True)

    print("\nPost-processing results:")
    for result in results[:10]:
        print(
            f"Threshold={result['threshold']:.2f} | "
            f"MinArea={result['min_area']:4d} | "
            f"Dice={result['dice']:.4f} | "
            f"IoU={result['iou']:.4f}"
        )

    best = results[0]
    print("\nBest parameters:")
    print(f"Threshold: {best['threshold']}")
    print(f"Min Area:  {best['min_area']}")
    print(f"Dice:      {best['dice']:.4f}")
    print(f"IoU:       {best['iou']:.4f}")


if __name__ == "__main__":
    main()
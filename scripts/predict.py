"""Predict the Kaggle test set and create submission.csv."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_utils
from tqdm import tqdm

from src.scripts.train import build_model

def predict_image(model, path, size, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    # unsqueeze batch dimension [1, 3, size, size]
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--checkpoint', default='/kaggle/working/outputs/best_unet.pth')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--output', default='/kaggle/working/outputs/submission.csv')
    parser.add_argument("--model", default=None, help="Segmentation model to use.", )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    root = Path(args.data_root or cfg['data']['root'])
    if not root.exists():
        raise FileNotFoundError(f'Data root not found: {root}')
    test_dir = root / 'test' / 'test_images'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model or cfg["model"].get("name", "unet")
    model = build_model(model_name, cfg['model']['features']).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    for path in tqdm(sorted(test_dir.glob('*.jpeg'))):
        prob = predict_image(model, path, cfg['data']['image_size'], device)
        binary = (prob >= args.threshold).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        # filter small components
        component_id = 0
        for label in range(1, n):
            if stats[label, cv2.CC_STAT_AREA] < args.min_area:
                continue
            component_id += 1
            component = (labels == label).astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(component))
            # component mask -> COCO RLE(Run-Length Encoding, 游程编码)
            counts = rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
            rows.append({'filament_id': f'{path.stem}_{component_id}', 'segmentation_rle': counts})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle']).to_csv(out, index=False)
    print(f'Saved {len(rows)} predicted filaments to {out}')

if __name__ == '__main__':
    main()

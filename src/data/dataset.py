"""MAGFiLO 2026 dataset loader.

The Kaggle data is COCO-style:
train/train_images/*.jpeg
train/MAGFiLO_1.0_Annotations_kaggle2026_train.json

We build one binary filament mask per image by combining all COCO
annotations belonging to the same image file.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


class MagfiloDataset(Dataset):
    def __init__(self, records, image_size=512, train=False):
        self.records = records
        self.image_size = image_size
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = make_mask(record["annotations"], record["height"], record["width"])

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        # mask 使用 cv2.INTER_NEAREST : 因为 mask 是 0/1，使用双线性插值可能产生小数

        if self.train:
            image, mask = augment(image, mask)

        image = image.astype(np.float32) / 255.0   # [0,1]
        image = (image - 0.5) / 0.5                # [-1,1]
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        # OpenCV/NumPy: HWC -> PyTorch CNN: CHW
        mask = torch.from_numpy(mask[None].astype(np.float32))
        # eg: image [3, 512, 512], mask [1, 512, 512]
        return image, mask


def load_records(root, val_ratio=0.15, seed=42):
    root = Path(root)
    image_dir = root / "train" / "train_images"
    annotation_file = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training images not found: {image_dir}")
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation JSON not found: {annotation_file}")

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_by_id = {item["id"]: item for item in coco["images"]}
    annotations_by_file = {}
    for ann in coco["annotations"]:
        image_info = image_by_id.get(ann["image_id"])
        if image_info is None:
            continue
        name = Path(image_info["file_name"]).name
        annotations_by_file.setdefault(name, []).append(ann)

    records = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        anns = annotations_by_file.get(path.name, [])
        if not anns:
            continue
        # Use the actual image size. The JSON is normally 2048x2048.
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        h, w = image.shape
        records.append({"image_path": str(path), "annotations": anns, "height": h, "width": w})

    if not records:
        raise RuntimeError("No labeled training images were matched with the COCO JSON.")

    random.Random(seed).shuffle(records)
    val_size = max(1, int(len(records) * val_ratio))
    return records[val_size:], records[:val_size]


def make_mask(annotations, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        segmentation = ann.get("segmentation")
        if not segmentation:
            continue
        if isinstance(segmentation, list):
            for polygon in segmentation:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if len(points) >= 3:
                    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        elif isinstance(segmentation, dict):
            if mask_utils is None:
                raise ImportError("pycocotools is required for RLE annotations")
            decoded = mask_utils.decode(segmentation)
            if decoded.ndim == 3:
                decoded = decoded.any(axis=2)
            mask[decoded.astype(bool)] = 1
    return mask


def augment(image, mask):
    '''
        image 与 mask 同步增强
    '''
    if random.random() < 0.5: # 水平翻转
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
    if random.random() < 0.5: # 垂直翻转
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
    k = random.randint(0, 3)
    if k: # 随机旋转
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()
    return image, mask


def make_loaders(root, image_size, batch_size, num_workers, val_ratio, seed):
    train_records, val_records = load_records(root, val_ratio, seed)
    train_ds = MagfiloDataset(train_records, image_size, train=True)
    val_ds = MagfiloDataset(val_records, image_size, train=False)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, shuffle=False, **common),
        len(train_ds), len(val_ds),
    )

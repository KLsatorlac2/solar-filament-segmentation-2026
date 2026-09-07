import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_records, MagfiloDataset
from src.data.sam_dataset import SAMDataset
from src.models.sam import SAM


def dice_score(pred, target):
    pred = pred.bool()
    target = target.bool()

    intersection = (pred & target).sum().item()

    return (
        2 * intersection + 1e-6
    ) / (
        pred.sum().item() +
        target.sum().item() +
        1e-6
    )


def iou_score(pred, target):
    pred = pred.bool()
    target = target.bool()

    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--image_size", type=int, default=1024)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, val_records = load_records(
        args.data_root,
        val_ratio=0.15,
        seed=42,
    )

    val_base = MagfiloDataset(
        val_records,
        image_size=args.image_size,
        train=False,
    )

    val_dataset = SAMDataset(
        val_base,
        args.image_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = SAM(
        checkpoint=args.sam_checkpoint,
        model_type="vit_b",
        freeze_encoder=True,
    ).to(device)

    decoder_state = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.sam.mask_decoder.load_state_dict(
        decoder_state
    )

    model.sam.image_encoder.eval()
    model.sam.prompt_encoder.eval()
    model.sam.mask_decoder.eval()

    dice_scores = []
    iou_scores = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device).float()

            points = batch["point_coords"].to(device)
            labels = batch["point_labels"].to(device)

            pred_masks, _ = model(
                images,
                points,
                labels,
            )

            probs = torch.sigmoid(pred_masks)

            pred = probs > 0.5

            dice = dice_score(
                pred[0, 0],
                masks[0, 0],
            )

            iou = iou_score(
                pred[0, 0],
                masks[0, 0],
            )

            dice_scores.append(dice)
            iou_scores.append(iou)

    print()
    print("=" * 50)
    print("SAM VALIDATION RESULTS")
    print("=" * 50)

    print(
        f"Samples: {len(dice_scores)}"
    )

    print(
        f"Mean Dice: {np.mean(dice_scores):.4f}"
    )

    print(
        f"Mean IoU:  {np.mean(iou_scores):.4f}"
    )

    print(
        f"Median Dice: {np.median(dice_scores):.4f}"
    )

    print(
        f"Median IoU:  {np.median(iou_scores):.4f}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()
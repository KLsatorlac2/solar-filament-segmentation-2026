import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import load_records, MagfiloDataset
from src.data.sam_dataset import SAMDataset
from src.models.sam import SAM


def dice_loss(pred, target):
    probs = torch.sigmoid(pred)

    intersection = (probs * target).sum(dim=(1, 2, 3))
    denominator = (
        probs.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
    )

    dice = (
        (2 * intersection + 1e-6)
        / (denominator + 1e-6)
    )

    return 1 - dice.mean()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument(
        "--output",
        default="outputs/best_sam.pth"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True,
    )

    train_records, val_records = load_records(
        args.data_root,
        val_ratio=0.15,
        seed=42,
    )

    train_base = MagfiloDataset(
        train_records,
        image_size=args.image_size,
        train=True,
    )

    train_dataset = SAMDataset(
        train_base,
        args.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = SAM(
        checkpoint=args.checkpoint,
        model_type="vit_b",
        freeze_encoder=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.sam.mask_decoder.parameters(),
        lr=args.lr,
    )

    best_loss = float("inf")

    for epoch in range(args.epochs):

        model.sam.image_encoder.eval()
        model.sam.prompt_encoder.eval()
        model.sam.mask_decoder.train()

        total_loss = 0.0

        for batch in train_loader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            ).float()

            points = batch["point_coords"].to(
                device,
                non_blocking=True,
            )

            labels = batch["point_labels"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred_masks, _ = model(
                images,
                points,
                labels,
            )

            pred_masks = pred_masks.float()

            bce = F.binary_cross_entropy_with_logits(
                pred_masks,
                masks,
            )

            dice = dice_loss(
                pred_masks,
                masks,
            )

            loss = bce + dice

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"Loss: {avg_loss:.4f}"
        )

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                model.sam.mask_decoder.state_dict(),
                args.output,
            )

            print(
                f"Saved: {args.output}"
            )


if __name__ == "__main__":
    main()
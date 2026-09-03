"""Train the U-Net baseline on the MAGFiLO 2026 dataset."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.data.dataset import make_loaders
from src.models.unet import UNet
from src.utils.losses import dice_score, iou_score, get_loss_function
from src.utils.utils import seed_everything

def run_epoch(model, loader, optimizer, device, loss_fn, train=True):
    model.train(train)
    total_loss = total_dice = total_iou = 0.0
    for images, masks in tqdm(loader, leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        total_dice += dice_score(logits, masks).item()
        total_iou += iou_score(logits, masks).item()
    n = max(1, len(loader))
    return total_loss / n, total_dice / n, total_iou / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')  # Path to the config file
    parser.add_argument('--data_root', default=None)  # Optional: Path to the root of the dataset
    parser.add_argument('--output_dir', default=None)  # Optional: Path to the output directory
    parser.add_argument(
        "--loss",
        default=None,
        choices=["bce_dice", "focal", "focal_dice", "tversky"],
        help="Loss function. Overrides training.loss in the config.",
    )
    args = parser.parse_args()  # Parse the arguments

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    seed_everything(cfg['seed'])

    data_root = args.data_root or cfg['data']['root']
    if not data_root:
        raise ValueError('Please provide --data_root or set data.root in the config.')
    out = Path(args.output_dir or cfg['output']['dir'])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Data:   {data_root}')
    print(f'Output: {out}')

    loss_name = args.loss or cfg["training"].get("loss", "bce_dice")
    loss_fn = get_loss_function(loss_name)
    print(f"Loss:   {loss_name}")

    train_loader, val_loader, n_train, n_val = make_loaders(
        data_root, cfg['data']['image_size'], cfg['training']['batch_size'],
        cfg['training']['num_workers'], cfg['data']['val_ratio'], cfg['seed'])
    print(f'Train images: {n_train} | Val images: {n_val}')

    model = UNet(tuple(cfg['model']['features'])).to(device)
    # 多 GPU 训练优化（Kaggle GPU T4 * 2）
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                  weight_decay=cfg['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_dice, bad_epochs, history = -1.0, 0, []

    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, optimizer, device, loss_fn, True)
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, optimizer, device, loss_fn, False)
        scheduler.step(val_dice)

        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
               'train_dice': train_dice, 'val_dice': val_dice,
               'train_iou': train_iou, 'val_iou': val_iou}
        history.append(row)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f'Epoch {epoch:03d} | loss {train_loss:.4f}/{val_loss:.4f} | '
              f'Dice {train_dice:.4f}/{val_dice:.4f} | IoU {train_iou:.4f}/{val_iou:.4f}'
              f' | LR {current_lr:.4f}')

        if val_dice > best_dice:
            best_dice, bad_epochs = val_dice, 0
            torch.save(model.state_dict(), out / 'best_unet.pth')
        else:
            bad_epochs += 1
            if bad_epochs >= cfg['training']['patience']:
                print('Early stopping.')
                break

    with open(out / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f'Best validation Dice: {best_dice:.4f}')

if __name__ == '__main__':
    main()

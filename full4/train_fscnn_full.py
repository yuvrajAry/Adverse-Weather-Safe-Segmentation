#!/usr/bin/env python3
"""
Full Fast-SCNN training on IDD-AW with clear terminal logs.
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

from app.config import AppConfig
from app.data.iddaw_dataset import IDDAWDataset
from app.models import build_model


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        targ_c = target == c
        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


def main():
    device = torch.device('cpu')
    model_name = 'fast_scnn'
    fusion_mode = 'early'
    width, height = AppConfig.DEFAULT_INFER_SIZE
    batch_size = 4
    epochs = 10
    lr = 1e-3
    workers = 0
    out_dir = 'checkpoints/fast_scnn_early_full'

    print('IDD-AW Fast-SCNN Full Training')
    print('=' * 60)
    print(f'Dataset: {AppConfig.DATASET_ROOT}')
    print(f'Device: {device}')
    print(f'Image size: {width}x{height}')
    print(f'Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}')

    # Datasets
    print('\nLoading datasets...')
    train_set = IDDAWDataset(
        root=AppConfig.DATASET_ROOT,
        split='train',
        size=(width, height),
        fusion_mode=fusion_mode,
        num_classes=AppConfig.NUM_CLASSES,
    )
    val_set = IDDAWDataset(
        root=AppConfig.DATASET_ROOT,
        split='val',
        size=(width, height),
        fusion_mode=fusion_mode,
        num_classes=AppConfig.NUM_CLASSES,
    )
    print(f'Train samples: {len(train_set)}')
    print(f'Val samples:   {len(val_set)}')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Model
    print('\nBuilding model...')
    model = build_model(model_name, fusion_mode, AppConfig.NUM_CLASSES, None).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer / Loss
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(out_dir, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        # Train
        model.train()
        running_loss = 0.0
        running_miou = 0.0
        n_batches = 0
        for b_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model.forward_logits(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                miou = compute_miou(pred, y, AppConfig.NUM_CLASSES)
            running_loss += loss.item()
            running_miou += miou
            n_batches += 1

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  [Train] Batch {b_idx+1}/{len(train_loader)}  loss={loss.item():.4f}  miou={miou:.4f}")

        train_loss = running_loss / max(1, n_batches)
        train_miou = running_miou / max(1, n_batches)
        print(f"  => Train: loss={train_loss:.4f}  miou={train_miou:.4f}")

        # Val
        model.eval()
        val_running_loss = 0.0
        val_running_miou = 0.0
        val_batches = 0
        with torch.no_grad():
            for b_idx, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                logits = model.forward_logits(x)
                loss = criterion(logits, y)
                pred = logits.argmax(dim=1)
                miou = compute_miou(pred, y, AppConfig.NUM_CLASSES)
                val_running_loss += loss.item()
                val_running_miou += miou
                val_batches += 1
                
            
        val_loss = val_running_loss / max(1, val_batches)
        val_miou = val_running_miou / max(1, val_batches)
        print(f"  => Val:   loss={val_loss:.4f}  miou={val_miou:.4f}")

        # Save checkpoints
        ckpt = {
            'epoch': epoch,
            'model': model_name,
            'fusion': fusion_mode,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_miou': train_miou,
            'val_loss': val_loss,
            'val_miou': val_miou,
        }
        latest_path = os.path.join(out_dir, f'{model_name}_{fusion_mode}_latest.pt')
        torch.save(ckpt, latest_path)

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(out_dir, f'{model_name}_{fusion_mode}_best.pt')
            torch.save(ckpt, best_path)
            print(f"  ✔ New best mIoU: {best_miou:.4f}")

    print('\nTraining completed.')
    print(f'Best mIoU: {best_miou:.4f}')
    print(f'Checkpoints in: {out_dir}')


if __name__ == '__main__':
    main()

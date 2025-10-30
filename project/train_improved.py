"""
Improved training script with learning rate scheduling and class weights
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from .dataset import IDDAWDataset
from .labels import LabelMapper, SafetyGroups
from .models import build_model
from .metrics import fast_hist, compute_miou, compute_safe_miou


def compute_class_weights(loader: DataLoader, num_classes: int, ignore_index: int, device: str) -> torch.Tensor:
    """Compute class weights based on inverse class frequency"""
    print(f"  Computing class weights for {num_classes} classes (ignore_index={ignore_index})...")
    
    # Initialize with small value to avoid division by zero
    class_counts = torch.ones(num_classes, dtype=torch.long, device=device)
    
    for batch in tqdm(loader, desc="  Processing batch"):
        mask = batch["mask"].to(device)
        # Flatten and count each class (excluding ignore_index)
        valid_mask = (mask >= 0) & (mask < num_classes)  # Only count valid class indices
        classes = mask[valid_mask].long()
        if len(classes) > 0:
            counts = torch.bincount(classes, minlength=num_classes)
            class_counts += counts
    
    # Convert to float for division
    class_counts = class_counts.float()
    
    # Calculate median frequency balancing
    median_freq = torch.median(class_counts[class_counts > 0])
    weights = median_freq / (class_counts + 1e-6)  # Add small epsilon
    
    # Set weight for ignore_index to 0
    if ignore_index < num_classes:
        weights[ignore_index] = 0.0
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    print("  Class distribution (pixels):", class_counts.cpu().numpy().astype(int))
    print("  Class weights (normalized):", [f"{w:.4f}" for w in weights.cpu().numpy()])
    
    return weights.to(device)


def create_dataloaders(splits_dir: str, image_size: int, batch_size: int, num_workers: int, modality: str):
    lbl = LabelMapper.default()
    fusion = modality
    train_ds = IDDAWDataset(f"{splits_dir}/train.csv", mode="train", label_mapper=lbl, augment=True, fusion=fusion, image_size=(image_size, image_size))
    val_ds = IDDAWDataset(f"{splits_dir}/val.csv", mode="val", label_mapper=lbl, augment=False, fusion=fusion, image_size=(image_size, image_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, lbl


def evaluate(model: nn.Module, loader: DataLoader, lbl: LabelMapper, device: torch.device, variant: str):
    model.eval()
    conf = torch.zeros((len(lbl.id_to_class), len(lbl.id_to_class)), dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in loader:
            if variant.startswith("mid"):
                x_rgb = batch["image_rgb"].to(device)
                x_nir = batch["image_nir"].to(device)
                logits = model(x_rgb, x_nir)
            else:
                x = batch["image"].to(device)
                logits = model(x)
            pred = torch.argmax(logits, dim=1)
            target = batch["mask"].to(device)
            conf += fast_hist(pred, target, num_classes=len(lbl.id_to_class), ignore_index=lbl.ignore_index)
    iou, mean_iou = compute_miou(conf.float())
    safety = compute_safe_miou(conf.float(), lbl.id_to_class, SafetyGroups.default().group_to_classes)
    return float(mean_iou), {i: float(iou[i]) for i in range(len(iou))}, safety


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, lbl = create_dataloaders(args.splits_dir, args.image_size, args.batch_size, args.workers, args.modality)
    
    print("=" * 70)
    print("IDDAW Training - Improved Version")
    print("=" * 70)
    print(f"Device: {device}", flush=True)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}", flush=True)
    print(f"Modality: {args.modality} | Backbone: {args.backbone}", flush=True)
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}", flush=True)
    print(f"Image size: {args.image_size} | AMP: {args.amp}", flush=True)
    print("=" * 70)
    
    # Build model
    variant = f"{args.modality}_{args.backbone}" if args.modality != "mid" else "mid_mbv3"
    model = build_model(variant, num_classes=len(lbl.id_to_class)).to(device)
    
    # Compute class weights if enabled
    class_weights = None
    if args.class_weights:
        class_weights = compute_class_weights(train_loader, len(lbl.id_to_class), lbl.ignore_index, device)
        criterion = nn.CrossEntropyLoss(ignore_index=lbl.ignore_index, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=lbl.ignore_index)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        print(f"Using CosineAnnealingLR scheduler", flush=True)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.1
        )
        print(f"Using StepLR scheduler", flush=True)
    else:
        scheduler = None
        print(f"No LR scheduler", flush=True)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_miou = float("-inf")
    if args.resume:
        checkpoint_path = f"{args.out_dir}/last_{variant}.pt"
        if Path(checkpoint_path).exists():
            print(f"Resuming from {checkpoint_path}", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            
            if "optimizer" in checkpoint and checkpoint["optimizer"]:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scaler" in checkpoint and checkpoint["scaler"]:
                scaler.load_state_dict(checkpoint["scaler"])
            if scheduler and "scheduler" in checkpoint and checkpoint["scheduler"]:
                scheduler.load_state_dict(checkpoint["scheduler"])
            
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_miou = checkpoint.get("miou", float("-inf"))
            print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.3f}", flush=True)
        else:
            print(f"No checkpoint found, starting from scratch", flush=True)

    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch+1}/{args.epochs}", flush=True)
        if scheduler:
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}", flush=True)
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            if args.modality == "mid":
                x_rgb = batch["image_rgb"].to(device)
                x_nir = batch["image_nir"].to(device)
                target = batch["mask"].to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(x_rgb, x_nir)
                    loss = criterion(logits, target)
            else:
                x = batch["image"].to(device)
                target = batch["mask"].to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(x)
                    loss = criterion(logits, target)
            
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Progress printing
            if (i + 1) % max(1, len(train_loader) // 10) == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  [{i+1}/{len(train_loader)}] loss: {avg_loss:.4f}", flush=True)
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}", flush=True)
        
        # Validation
        print("Evaluating...", flush=True)
        miou, class_ious, safety = evaluate(model, val_loader, lbl, device, variant)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Save checkpoints
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        
        # Always save last
        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "miou": miou,
            "safety": safety,
            "loss": avg_epoch_loss
        }
        if scheduler:
            checkpoint_data["scheduler"] = scheduler.state_dict()
        
        torch.save(checkpoint_data, f"{args.out_dir}/last_{variant}.pt")
        
        # Save best
        if miou >= best_miou:
            best_miou = miou
            torch.save({
                "model": model.state_dict(),
                "miou": miou,
                "safety": safety,
                "epoch": epoch
            }, f"{args.out_dir}/best_{variant}.pt")
            print(f"✓ New best model saved! mIoU: {miou:.3f}", flush=True)
        
        print(f"Epoch {epoch+1}: mIoU={miou:.3f}, safe_mIoU={safety.get('safe_mIoU',0):.3f}, best={best_miou:.3f}", flush=True)
        
        # Print per-class IoU for important classes
        if args.verbose:
            print("Per-class IoU:", flush=True)
            for class_id, iou_val in class_ious.items():
                if class_id < len(lbl.id_to_class):
                    class_name = lbl.id_to_class[class_id]
                    print(f"  {class_name}: {iou_val:.3f}", flush=True)
    
    print("=" * 70)
    print(f"Training complete! Best mIoU: {best_miou:.3f}")
    print(f"Checkpoints saved in {args.out_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="IDDAW Improved Training")
    parser.add_argument("--splits_dir", type=str, default="project/splits")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="project/ckpts")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--modality", type=str, default="mid", choices=["rgb", "nir", "early4", "mid"])
    parser.add_argument("--backbone", type=str, default="mbv3", choices=["mbv3", "fastscnn"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--class_weights", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--verbose", action="store_true", help="Print per-class IoU")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive training script for IDD-AW dataset with multiple models and fusion strategies.
Trains Fast-SCNN and MobileNetV3 models with Early and Mid-level fusion.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Import local modules
from app.config import AppConfig
from app.data.iddaw_dataset import IDDAWDataset
from app.models import build_model


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Compute mean Intersection over Union (mIoU) metric."""
    # pred, target: [N,H,W] int
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
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute pixel-wise accuracy."""
    pred = pred.view(-1)
    target = target.view(-1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total if total > 0 else 0.0


def train_model(model_name: str, fusion_mode: str, args):
    """Train a single model configuration."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} with {fusion_mode} fusion")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    try:
        train_set = IDDAWDataset(
            root=AppConfig.DATASET_ROOT, 
            split='train', 
            size=(args.width, args.height), 
            fusion_mode=fusion_mode, 
            num_classes=AppConfig.NUM_CLASSES
        )
        val_set = IDDAWDataset(
            root=AppConfig.DATASET_ROOT, 
            split='val', 
            size=(args.width, args.height), 
            fusion_mode=fusion_mode, 
            num_classes=AppConfig.NUM_CLASSES
        )
        print(f"Training samples: {len(train_set)}")
        print(f"Validation samples: {len(val_set)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True
    )
    
    # Create model
    model = build_model(model_name, fusion_mode, AppConfig.NUM_CLASSES, None).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training state
    best_miou = 0.0
    train_history = []
    
    # Create output directory
    model_dir = os.path.join(args.out_dir, f"{model_name}_{fusion_mode}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Saving checkpoints to: {model_dir}")
    
    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        running_pixel_acc = 0.0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for batch_idx, (x, y) in enumerate(train_pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Forward pass
            logits = model.forward_logits(x)
            loss = criterion(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                pixel_acc = compute_pixel_accuracy(pred, y)
                running_pixel_acc += pixel_acc
                running_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pixel_acc': f'{pixel_acc:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        train_loss = running_loss / num_batches
        train_pixel_acc = running_pixel_acc / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        val_pixel_acc = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]")
            for x, y in val_pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                logits = model.forward_logits(x)
                loss = criterion(logits, y)
                pred = logits.argmax(dim=1)
                
                val_loss += loss.item()
                val_miou += compute_miou(pred, y, AppConfig.NUM_CLASSES)
                val_pixel_acc += compute_pixel_accuracy(pred, y)
                num_val_batches += 1
                
                val_pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_miou': f'{compute_miou(pred, y, AppConfig.NUM_CLASSES):.4f}'
                })
        
        val_loss /= num_val_batches
        val_miou /= num_val_batches
        val_pixel_acc /= num_val_batches
        
        # Learning rate scheduling
        scheduler.step(val_miou)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model_name,
            'fusion': fusion_mode,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_pixel_acc': train_pixel_acc,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'val_pixel_acc': val_pixel_acc,
            'best_miou': best_miou,
            'args': vars(args)
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(model_dir, f"{model_name}_{fusion_mode}_latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(model_dir, f"{model_name}_{fusion_mode}_best.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved! mIoU: {val_miou:.4f}")
        
        # Log training history
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_pixel_acc': train_pixel_acc,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'val_pixel_acc': val_pixel_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Print epoch summary
        print(f"Epoch {epoch:3d}: "
              f"train_loss={train_loss:.4f} train_acc={train_pixel_acc:.4f} "
              f"val_loss={val_loss:.4f} val_miou={val_miou:.4f} val_acc={val_pixel_acc:.4f} "
              f"best_miou={best_miou:.4f}")
    
    # Save training history
    history_path = os.path.join(model_dir, f"{model_name}_{fusion_mode}_history.json")
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"\nTraining completed for {model_name} + {fusion_mode}")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Results saved in: {model_dir}")
    
    return {
        'model': model_name,
        'fusion': fusion_mode,
        'best_miou': best_miou,
        'final_epoch': args.epochs,
        'model_dir': model_dir
    }


def main():
    parser = argparse.ArgumentParser(description='Train IDD-AW models')
    parser.add_argument('--model', default='all', 
                       choices=['fast_scnn', 'mobilenetv3_lite', 'all'],
                       help='Model to train')
    parser.add_argument('--fusion', default='all',
                       choices=['early', 'mid', 'all'],
                       help='Fusion strategy')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out-dir', default='checkpoints')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print("IDD-AW Comprehensive Training")
    print("=" * 50)
    print(f"Dataset root: {AppConfig.DATASET_ROOT}")
    print(f"Output directory: {args.out_dir}")
    print(f"Device: {'CPU' if args.cpu else 'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    
    # Determine which models and fusions to train
    models = ['fast_scnn', 'mobilenetv3_lite'] if args.model == 'all' else [args.model]
    fusions = ['early', 'mid'] if args.fusion == 'all' else [args.fusion]
    
    results = []
    start_time = datetime.now()
    
    for model_name in models:
        for fusion_mode in fusions:
            try:
                result = train_model(model_name, fusion_mode, args)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error training {model_name} + {fusion_mode}: {e}")
                continue
    
    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total training time: {duration}")
    print(f"Models trained: {len(results)}")
    
    for result in results:
        print(f"{result['model']} + {result['fusion']}: "
              f"mIoU={result['best_miou']:.4f} "
              f"({result['model_dir']})")
    
    # Save overall results
    summary_path = os.path.join(args.out_dir, 'training_summary.json')
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'args': vars(args),
        'results': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to: {summary_path}")


if __name__ == '__main__':
    main()

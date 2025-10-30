#!/usr/bin/env python3
"""
Efficient training script for all model combinations
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from datetime import datetime

from app.config import AppConfig
from app.data.iddaw_dataset import IDDAWDataset
from app.models import build_model


def compute_miou(pred, target, num_classes):
    """Compute mean IoU"""
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


def train_model(model_name, fusion_mode, epochs=25, batch_size=4):
    """Train a single model configuration"""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} with {fusion_mode.upper()} fusion")
    print(f"{'='*60}")
    
    device = torch.device('cpu')  # CPU training
    print(f"Device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_set = IDDAWDataset(
        root=AppConfig.DATASET_ROOT, 
        split='train', 
        size=(512, 512), 
        fusion_mode=fusion_mode, 
        num_classes=AppConfig.NUM_CLASSES
    )
    val_set = IDDAWDataset(
        root=AppConfig.DATASET_ROOT, 
        split='val', 
        size=(512, 512), 
        fusion_mode=fusion_mode, 
        num_classes=AppConfig.NUM_CLASSES
    )
    
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = build_model(model_name, fusion_mode, AppConfig.NUM_CLASSES, None).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=3, verbose=True
    )
    
    best_miou = 0.0
    train_history = []
    
    # Create output directory
    model_dir = f"checkpoints/{model_name}_{fusion_mode}"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Saving to: {model_dir}")
    print("Starting training...\n")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_miou = 0.0
        num_batches = 0
        
        print(f"Epoch {epoch:2d}/{epochs} [TRAIN]", end=" ")
        
        for batch_idx, (x, y) in enumerate(train_loader):
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
                train_loss += loss.item()
                train_miou += miou
                num_batches += 1
            
            # Progress indicator
            if batch_idx % 50 == 0:
                print(".", end="", flush=True)
        
        train_loss /= num_batches
        train_miou /= num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                logits = model.forward_logits(x)
                loss = criterion(logits, y)
                pred = logits.argmax(dim=1)
                miou = compute_miou(pred, y, AppConfig.NUM_CLASSES)
                
                val_loss += loss.item()
                val_miou += miou
                num_val_batches += 1
        
        val_loss /= num_val_batches
        val_miou /= num_val_batches
        
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
            'train_miou': train_miou,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'best_miou': best_miou,
        }
        
        # Save latest checkpoint
        latest_path = f"{model_dir}/{model_name}_{fusion_mode}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            best_path = f"{model_dir}/{model_name}_{fusion_mode}_best.pt"
            torch.save(checkpoint, best_path)
            print(f" ⭐ NEW BEST!")
        else:
            print()
        
        # Log training history
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_miou': train_miou,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Print epoch summary
        print(f"         Train: loss={train_loss:.4f} mIoU={train_miou:.4f}")
        print(f"         Val:   loss={val_loss:.4f} mIoU={val_miou:.4f} (best: {best_miou:.4f})")
        print(f"         LR:    {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save training history
    history_path = f"{model_dir}/{model_name}_{fusion_mode}_history.json"
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"\n✅ Training completed for {model_name} + {fusion_mode}")
    print(f"   Best validation mIoU: {best_miou:.4f}")
    print(f"   Results saved in: {model_dir}")
    
    return {
        'model': model_name,
        'fusion': fusion_mode,
        'best_miou': best_miou,
        'final_epoch': epochs,
        'model_dir': model_dir
    }


def main():
    print("🚀 IDD-AW Comprehensive Model Training")
    print("=" * 60)
    print(f"Dataset: {AppConfig.DATASET_ROOT}")
    print(f"Classes: {AppConfig.NUM_CLASSES}")
    print(f"Safety classes: {AppConfig.SAFETY_CLASS_IDS}")
    print(f"Device: CPU")
    
    # Training configurations
    models = ['fast_scnn', 'mobilenetv3_lite']
    fusions = ['early', 'mid']
    
    results = []
    start_time = datetime.now()
    
    for model_name in models:
        for fusion_mode in fusions:
            try:
                print(f"\n🎯 Starting {model_name} + {fusion_mode}")
                result = train_model(model_name, fusion_mode, epochs=25, batch_size=4)
                if result:
                    results.append(result)
            except KeyboardInterrupt:
                print(f"\n⏹️  Training interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error training {model_name} + {fusion_mode}: {e}")
                continue
    
    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "🏆" * 20)
    print("TRAINING SUMMARY")
    print("🏆" * 20)
    print(f"Total time: {duration}")
    print(f"Models trained: {len(results)}")
    
    if results:
        print("\nResults:")
        for result in results:
            print(f"  {result['model']} + {result['fusion']}: "
                  f"mIoU = {result['best_miou']:.4f}")
    
    # Save overall results
    summary_path = "checkpoints/training_summary.json"
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'results': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()

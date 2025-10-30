#!/usr/bin/env python3
"""
Working training script for IDD-AW models
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
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


def train_model(model_name, fusion_mode, epochs=10, batch_size=4):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name} with {fusion_mode} fusion")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = build_model(model_name, fusion_mode, AppConfig.NUM_CLASSES, None).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_miou = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_miou = 0.0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for x, y in train_pbar:
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
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'miou': f'{miou:.4f}'
                })
        
        train_loss /= num_batches
        train_miou /= num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]")
            for x, y in val_pbar:
                x = x.to(device)
                y = y.to(device)
                
                logits = model.forward_logits(x)
                loss = criterion(logits, y)
                pred = logits.argmax(dim=1)
                miou = compute_miou(pred, y, AppConfig.NUM_CLASSES)
                
                val_loss += loss.item()
                val_miou += miou
                num_val_batches += 1
                
                val_pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_miou': f'{miou:.4f}'
                })
        
        val_loss /= num_val_batches
        val_miou /= num_val_batches
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model_name,
            'fusion': fusion_mode,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_miou': train_miou,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'best_miou': best_miou,
        }
        
        # Save latest
        latest_path = f"checkpoints/{model_name}_{fusion_mode}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if val_miou > best_miou:
            best_miou = val_miou
            best_path = f"checkpoints/{model_name}_{fusion_mode}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model! mIoU: {val_miou:.4f}")
        
        print(f"Epoch {epoch:3d}: "
              f"train_loss={train_loss:.4f} train_miou={train_miou:.4f} "
              f"val_loss={val_loss:.4f} val_miou={val_miou:.4f} "
              f"best_miou={best_miou:.4f}")
    
    print(f"\nTraining completed for {model_name} + {fusion_mode}")
    print(f"Best validation mIoU: {best_miou:.4f}")
    
    return best_miou


def main():
    print("IDD-AW Model Training")
    print("=" * 50)
    print(f"Dataset root: {AppConfig.DATASET_ROOT}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Train all combinations
    models = ['fast_scnn', 'mobilenetv3_lite']
    fusions = ['early', 'mid']
    
    results = []
    for model_name in models:
        for fusion_mode in fusions:
            try:
                best_miou = train_model(model_name, fusion_mode, epochs=20, batch_size=4)
                results.append({
                    'model': model_name,
                    'fusion': fusion_mode,
                    'best_miou': best_miou
                })
            except Exception as e:
                print(f"Error training {model_name} + {fusion_mode}: {e}")
                continue
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    for result in results:
        print(f"{result['model']} + {result['fusion']}: mIoU = {result['best_miou']:.4f}")


if __name__ == '__main__':
    main()

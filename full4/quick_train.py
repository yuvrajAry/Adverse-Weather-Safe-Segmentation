#!/usr/bin/env python3
"""
Quick training script with verbose output
"""

import os
import sys
sys.path.append('.')

print("Starting quick training...")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

print("Imports successful")

from app.config import AppConfig
from app.data.iddaw_dataset import IDDAWDataset
from app.models import build_model

print("Local imports successful")

def train_fast_scnn_early():
    """Train Fast-SCNN with early fusion"""
    print("\n" + "="*50)
    print("Training Fast-SCNN with Early Fusion")
    print("="*50)
    
    device = torch.device('cpu')  # Force CPU since CUDA not available
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    try:
        train_set = IDDAWDataset(
            root=AppConfig.DATASET_ROOT, 
            split='train', 
            size=(512, 512), 
            fusion_mode='early', 
            num_classes=AppConfig.NUM_CLASSES
        )
        val_set = IDDAWDataset(
            root=AppConfig.DATASET_ROOT, 
            split='val', 
            size=(512, 512), 
            fusion_mode='early', 
            num_classes=AppConfig.NUM_CLASSES
        )
        print(f"Training samples: {len(train_set)}")
        print(f"Validation samples: {len(val_set)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)
    
    # Create model
    print("Creating model...")
    model = build_model('fast_scnn', 'early', AppConfig.NUM_CLASSES, None).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Starting training...")
    
    epochs = 3
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model.forward_logits(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Train] Batch {batch_idx+1}/{len(train_loader)} loss={loss.item():.4f}")
        
        avg_train_loss = train_loss / max(1, num_batches)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                
                logits = model.forward_logits(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / max(1, num_val_batches)
        
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss:   {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': 'fast_scnn',
            'fusion': 'early',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }
        
        os.makedirs('checkpoints/fast_scnn_early_full', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/fast_scnn_early_full/fast_scnn_early_epoch_{epoch}.pt')
        torch.save(checkpoint, 'checkpoints/fast_scnn_early_full/fast_scnn_early_latest.pt')
        print(f"  Saved checkpoints for epoch {epoch}")
    
    print("\nTraining completed!")

if __name__ == '__main__':
    train_fast_scnn_early()

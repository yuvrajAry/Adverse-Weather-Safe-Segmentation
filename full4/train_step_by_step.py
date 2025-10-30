#!/usr/bin/env python3
"""
Step-by-step training to monitor progress
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from app.config import AppConfig
from app.data.iddaw_dataset import IDDAWDataset
from app.models import build_model

def train_single_model():
    """Train one model with detailed logging"""
    print("🚀 Starting Fast-SCNN Early Fusion Training")
    print("=" * 50)
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\n📁 Loading dataset...")
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
    
    print(f"✅ Training samples: {len(train_set)}")
    print(f"✅ Validation samples: {len(val_set)}")
    
    # Create data loaders
    print("\n🔄 Creating data loaders...")
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🧠 Creating model...")
    model = build_model('fast_scnn', 'early', AppConfig.NUM_CLASSES, None).to(device)
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print("✅ Loss function and optimizer ready")
    
    # Create output directory
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n🏃 Starting training...")
    
    # Training loop
    for epoch in range(1, 11):  # 10 epochs
        print(f"\n📈 Epoch {epoch}/10")
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        epoch_start = time.time()
        
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
            
            # Progress update every 100 batches
            if batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                print(f"  Batch {batch_idx:4d}: loss = {loss.item():.4f} "
                      f"(elapsed: {elapsed:.1f}s)")
        
        avg_train_loss = train_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        print(f"  ✅ Train loss: {avg_train_loss:.4f} ({epoch_time:.1f}s)")
        
        # Validation phase
        print("  🔍 Validation...")
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        val_start = time.time()
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                logits = model.forward_logits(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        val_time = time.time() - val_start
        
        print(f"  ✅ Val loss: {avg_val_loss:.4f} ({val_time:.1f}s)")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': 'fast_scnn',
            'fusion': 'early',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'timestamp': time.time()
        }
        
        checkpoint_path = f'checkpoints/fast_scnn_early_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved: {checkpoint_path}")
        
        # Save latest
        latest_path = 'checkpoints/fast_scnn_early_latest.pt'
        torch.save(checkpoint, latest_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Final train loss: {avg_train_loss:.4f}")
    print(f"📊 Final val loss: {avg_val_loss:.4f}")
    
    return avg_train_loss, avg_val_loss

if __name__ == '__main__':
    train_single_model()

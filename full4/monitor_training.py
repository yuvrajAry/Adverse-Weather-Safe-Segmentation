#!/usr/bin/env python3
"""
Monitor training progress and display results
"""

import os
import torch
import json
from datetime import datetime

def check_training_progress():
    """Check current training progress"""
    print("Training Progress Monitor")
    print("=" * 40)
    
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found")
        return
    
    # List all checkpoint files
    pt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not pt_files:
        print("No checkpoint files found")
        return
    
    print(f"Found {len(pt_files)} checkpoint files:")
    
    results = {}
    
    for pt_file in pt_files:
        try:
            checkpoint_path = os.path.join(checkpoint_dir, pt_file)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model = checkpoint.get('model', 'unknown')
            fusion = checkpoint.get('fusion', 'unknown')
            epoch = checkpoint.get('epoch', 0)
            val_miou = checkpoint.get('val_miou', 0.0)
            best_miou = checkpoint.get('best_miou', 0.0)
            
            key = f"{model}_{fusion}"
            if key not in results or val_miou > results[key]['val_miou']:
                results[key] = {
                    'model': model,
                    'fusion': fusion,
                    'epoch': epoch,
                    'val_miou': val_miou,
                    'best_miou': best_miou,
                    'file': pt_file,
                    'size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
                }
            
            print(f"  {pt_file}: epoch {epoch}, val_miou {val_miou:.4f}")
            
        except Exception as e:
            print(f"  {pt_file}: Error loading - {e}")
    
    print("\n" + "=" * 40)
    print("BEST RESULTS BY MODEL+FUSION")
    print("=" * 40)
    
    for key, result in results.items():
        print(f"{result['model']} + {result['fusion']}:")
        print(f"  Epoch: {result['epoch']}")
        print(f"  Validation mIoU: {result['val_miou']:.4f}")
        print(f"  Best mIoU: {result['best_miou']:.4f}")
        print(f"  File: {result['file']} ({result['size_mb']:.1f} MB)")
        print()

def check_dataset_status():
    """Check dataset loading status"""
    print("Dataset Status")
    print("=" * 20)
    
    try:
        import sys
        sys.path.append('.')
        from app.config import AppConfig
        from app.data.iddaw_dataset import IDDAWDataset
        
        root = AppConfig.DATASET_ROOT
        print(f"Dataset root: {root}")
        print(f"Exists: {os.path.exists(root)}")
        
        if os.path.exists(root):
            train_set = IDDAWDataset(root=root, split='train')
            val_set = IDDAWDataset(root=root, split='val')
            
            print(f"Training samples: {len(train_set)}")
            print(f"Validation samples: {len(val_set)}")
            
            # Test loading a sample
            sample = train_set[0]
            print(f"Sample shapes: input {sample[0].shape}, target {sample[1].shape}")
            
    except Exception as e:
        print(f"Error checking dataset: {e}")

if __name__ == '__main__':
    check_dataset_status()
    print()
    check_training_progress()

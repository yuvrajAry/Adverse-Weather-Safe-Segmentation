#!/usr/bin/env python3
"""
Simple training script to test the training process
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.config import AppConfig
from app.data.iddaw_dataset import IDDAWDataset
from app.models import build_model

def main():
    print("Starting simple training test...")
    print(f"Dataset root: {AppConfig.DATASET_ROOT}")
    print(f"Dataset exists: {os.path.exists(AppConfig.DATASET_ROOT)}")
    
    # Test dataset loading
    try:
        print("Loading training dataset...")
        train_set = IDDAWDataset(root=AppConfig.DATASET_ROOT, split='train')
        print(f"Training samples: {len(train_set)}")
        
        print("Loading validation dataset...")
        val_set = IDDAWDataset(root=AppConfig.DATASET_ROOT, split='val')
        print(f"Validation samples: {len(val_set)}")
        
        # Test loading a sample
        print("Loading first sample...")
        sample = train_set[0]
        print(f"Input shape: {sample[0].shape}, Target shape: {sample[1].shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model creation
    try:
        print("Creating model...")
        model = build_model('fast_scnn', 'early', AppConfig.NUM_CLASSES, None)
        print(f"Model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        print("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            input_tensor = sample[0].unsqueeze(0)  # Add batch dimension
            logits = model.forward_logits(input_tensor)
            print(f"Forward pass successful, output shape: {logits.shape}")
            
    except Exception as e:
        print(f"Error with model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("All tests passed! Ready for training.")

if __name__ == '__main__':
    main()

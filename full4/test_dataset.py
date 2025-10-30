#!/usr/bin/env python3
"""
Test script to debug dataset loading issues
"""

import sys
import os
sys.path.append('.')

from app.config import AppConfig
from app.data.iddaw_dataset import IDDAWDataset, _find_samples_split

def test_dataset():
    print("Testing IDD-AW Dataset Loading")
    print("=" * 40)
    
    print(f"Dataset root: {AppConfig.DATASET_ROOT}")
    print(f"Dataset exists: {os.path.exists(AppConfig.DATASET_ROOT)}")
    
    # Test sample finding
    try:
        train_samples = _find_samples_split(AppConfig.DATASET_ROOT, 'train')
        print(f"Found {len(train_samples)} training samples")
        if train_samples:
            print(f"First sample: {train_samples[0]}")
            
        val_samples = _find_samples_split(AppConfig.DATASET_ROOT, 'val')
        print(f"Found {len(val_samples)} validation samples")
        
    except Exception as e:
        print(f"Error finding samples: {e}")
        return False
    
    # Test dataset creation
    try:
        train_set = IDDAWDataset(root=AppConfig.DATASET_ROOT, split='train')
        print(f"Training dataset created: {len(train_set)} samples")
        
        val_set = IDDAWDataset(root=AppConfig.DATASET_ROOT, split='val')
        print(f"Validation dataset created: {len(val_set)} samples")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return False
    
    # Test loading a sample
    try:
        sample = train_set[0]
        print(f"Sample loaded successfully:")
        print(f"  Input shape: {sample[0].shape}")
        print(f"  Target shape: {sample[1].shape}")
        print(f"  Input dtype: {sample[0].dtype}")
        print(f"  Target dtype: {sample[1].dtype}")
        print(f"  Input range: [{sample[0].min():.3f}, {sample[0].max():.3f}]")
        print(f"  Target range: [{sample[1].min()}, {sample[1].max()}]")
        
    except Exception as e:
        print(f"Error loading sample: {e}")
        return False
    
    print("Dataset test completed successfully!")
    return True

if __name__ == '__main__':
    success = test_dataset()
    sys.exit(0 if success else 1)

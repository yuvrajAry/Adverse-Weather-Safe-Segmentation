#!/usr/bin/env python3
"""
Check training progress and display current status
"""

import os
import torch
import json
from datetime import datetime

def check_progress():
    """Check current training progress"""
    print("🔍 Training Progress Monitor")
    print("=" * 50)
    
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("❌ No checkpoints directory found")
        return
    
    # Check for summary file
    summary_path = os.path.join(checkpoint_dir, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        print("📊 Training Summary Found:")
        print(f"   Start time: {summary['start_time']}")
        if 'end_time' in summary:
            print(f"   End time: {summary['end_time']}")
            print(f"   Duration: {summary['duration_seconds']:.0f} seconds")
        print(f"   Results: {len(summary.get('results', []))} models completed")
    
    # Check individual model directories
    model_dirs = [d for d in os.listdir(checkpoint_dir) 
                  if os.path.isdir(os.path.join(checkpoint_dir, d)) and '_' in d]
    
    print(f"\n📁 Found {len(model_dirs)} model directories:")
    
    for model_dir in sorted(model_dirs):
        print(f"\n🎯 {model_dir.upper()}:")
        
        # Check for best checkpoint
        best_path = os.path.join(checkpoint_dir, model_dir, f"{model_dir}_best.pt")
        latest_path = os.path.join(checkpoint_dir, model_dir, f"{model_dir}_latest.pt")
        history_path = os.path.join(checkpoint_dir, model_dir, f"{model_dir}_history.json")
        
        if os.path.exists(best_path):
            try:
                checkpoint = torch.load(best_path, map_location='cpu')
                epoch = checkpoint.get('epoch', 0)
                val_miou = checkpoint.get('val_miou', 0.0)
                print(f"   ✅ Best model: Epoch {epoch}, mIoU {val_miou:.4f}")
            except:
                print(f"   ⚠️  Best checkpoint exists but corrupted")
        
        if os.path.exists(latest_path):
            try:
                checkpoint = torch.load(latest_path, map_location='cpu')
                epoch = checkpoint.get('epoch', 0)
                val_miou = checkpoint.get('val_miou', 0.0)
                print(f"   📈 Latest: Epoch {epoch}, mIoU {val_miou:.4f}")
            except:
                print(f"   ⚠️  Latest checkpoint exists but corrupted")
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                print(f"   📊 Training history: {len(history)} epochs")
                if history:
                    final = history[-1]
                    print(f"   📈 Final: Train mIoU {final['train_miou']:.4f}, "
                          f"Val mIoU {final['val_miou']:.4f}")
            except:
                print(f"   ⚠️  History file exists but corrupted")
        
        if not any([os.path.exists(p) for p in [best_path, latest_path, history_path]]):
            print(f"   ⏳ No checkpoints found yet (training in progress)")
    
    # Check for any loose checkpoint files
    pt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and not os.path.isdir(os.path.join(checkpoint_dir, f))]
    if pt_files:
        print(f"\n📄 Other checkpoint files: {len(pt_files)}")
        for pt_file in pt_files:
            try:
                checkpoint = torch.load(os.path.join(checkpoint_dir, pt_file), map_location='cpu')
                model = checkpoint.get('model', 'unknown')
                fusion = checkpoint.get('fusion', 'unknown')
                epoch = checkpoint.get('epoch', 0)
                val_miou = checkpoint.get('val_miou', 0.0)
                print(f"   {pt_file}: {model}+{fusion}, Epoch {epoch}, mIoU {val_miou:.4f}")
            except:
                print(f"   {pt_file}: Corrupted or invalid")

if __name__ == '__main__':
    check_progress()

# 🚀 Train IDDAW Models on Google Colab (FREE GPU)

## Why Google Colab?

- ✅ **FREE Tesla T4 GPU** (worth $500+)
- ✅ **3-10x faster** than CPU training
- ✅ **No setup** required
- ✅ **Can close laptop** and let it run in cloud

---

## 📦 Step 1: Prepare Your Files

### **Option A: Upload to Google Drive**

1. Compress your project:
   ```powershell
   # In D:\iddaw\pro
   Compress-Archive -Path project,IDDAW -DestinationPath iddaw_project.zip
   ```

2. Upload `iddaw_project.zip` to Google Drive

### **Option B: Use GitHub**

1. Push your code to GitHub
2. Clone in Colab (faster)

---

## 🎯 Step 2: Open Google Colab

1. Go to: https://colab.research.google.com
2. Click **"New Notebook"**
3. **Enable GPU**: 
   - Runtime → Change runtime type → Hardware accelerator → **GPU (T4)**
   - Click **Save**

---

## 💻 Step 3: Run This Code in Colab

### **Cell 1: Setup**

```python
# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### **Cell 2: Install Dependencies**

```python
!pip install -q opencv-python-headless pillow numpy
```

### **Cell 3: Extract Project (if using zip)**

```python
import zipfile
import os

# Extract project
zip_path = '/content/drive/MyDrive/iddaw_project.zip'  # Adjust path
extract_to = '/content/iddaw'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

os.chdir(extract_to)
print(f"Working directory: {os.getcwd()}")
```

### **Cell 4: Train FastSCNN NIR (FASTEST - 30-45 min)**

```python
!python -m project.train_improved \
    --modality nir \
    --backbone fastscnn \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --amp \
    --scheduler cosine \
    --class_weights \
    --out_dir project/ckpts
```

### **Cell 5: Train FastSCNN RGB (1-1.5 hours)**

```python
!python -m project.train_improved \
    --modality rgb \
    --backbone fastscnn \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --amp \
    --scheduler cosine \
    --class_weights \
    --out_dir project/ckpts
```

### **Cell 6: Train Mid-Fusion (BEST - 2-3 hours)**

```python
!python -m project.train_improved \
    --modality mid \
    --backbone mbv3 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --amp \
    --scheduler cosine \
    --class_weights \
    --out_dir project/ckpts
```

### **Cell 7: Download Trained Models**

```python
# Compress checkpoints
!zip -r trained_models.zip project/ckpts/

# Download to your computer
from google.colab import files
files.download('trained_models.zip')
```

---

## ⏱️ Expected Training Times on Colab GPU

| Model | Epochs | Time |
|-------|--------|------|
| NIR + FastSCNN | 50 | ~30-45 min |
| RGB + FastSCNN | 50 | ~1-1.5 hours |
| Early4 + MobileNetV3 | 50 | ~1.5-2 hours |
| Mid-fusion + MobileNetV3 | 50 | ~2-3 hours |

---

## 🎯 Recommended Training Order

### **Fast Track (3-4 hours total)**
1. NIR FastSCNN (45 min)
2. RGB FastSCNN (1.5 hours)
3. Mid-fusion MobileNetV3 (2.5 hours)

### **Quality Track (2-3 hours)**
Just train Mid-fusion MobileNetV3 for best results

---

## 📊 Monitor Training in Colab

You'll see output like:
```
CUDA available: True
GPU: Tesla T4

Epoch 1/50
Learning rate: 0.001000
  [10/100] loss: 1.2345
Epoch 1: mIoU=0.450, safe_mIoU=0.420, best=0.450
✓ New best model saved!
```

---

## 💾 After Training

### **1. Download Models**

Run Cell 7 to download `trained_models.zip`

### **2. Extract to Your Project**

```powershell
# On your local machine
cd D:\iddaw\pro
Expand-Archive -Path trained_models.zip -DestinationPath . -Force
```

### **3. Test New Models**

```powershell
# Start backend with new models
cd D:\iddaw\pro\project
python backend_with_models.py

# Upload images through web interface
# See improved results!
```

---

## 🔥 Pro Tips

### **1. Keep Colab Alive**

Colab disconnects after ~90 minutes of inactivity. To prevent:

```python
# Run this in a cell
import time
from IPython.display import clear_output

while True:
    time.sleep(300)  # Every 5 minutes
    clear_output()
    print("Still training...")
```

### **2. Save Checkpoints to Drive**

```python
# In training command, use Drive path
!python -m project.train_improved \
    --modality mid \
    --epochs 50 \
    --out_dir /content/drive/MyDrive/iddaw_checkpoints \
    --resume
```

This way, if Colab disconnects, you can resume!

### **3. Train Multiple Models**

```python
# Train all models in one go
models = [
    ("nir", "fastscnn"),
    ("rgb", "fastscnn"),
    ("mid", "mbv3")
]

for modality, backbone in models:
    print(f"\n{'='*70}")
    print(f"Training {modality} + {backbone}")
    print(f"{'='*70}\n")
    
    !python -m project.train_improved \
        --modality {modality} \
        --backbone {backbone} \
        --epochs 50 \
        --batch_size 16 \
        --amp \
        --scheduler cosine \
        --class_weights \
        --out_dir project/ckpts
```

---

## 🆘 Troubleshooting

### **"No GPU available"**
- Runtime → Change runtime type → GPU → Save
- Restart runtime

### **"Out of memory"**
Reduce batch size:
```python
--batch_size 8  # or 4
```

### **"Session timeout"**
- Colab free tier: 12 hours max
- Save checkpoints to Drive
- Resume with `--resume` flag

---

## 🎉 Summary

1. **Open Colab** → Enable GPU
2. **Run setup cells** (mount Drive, install packages)
3. **Train models** (start with FastSCNN for speed)
4. **Download models** when done
5. **Use in your backend** for better results

**Total time: 3-4 hours for all models on FREE GPU!** 🚀

Much better than 20-30 hours on CPU!

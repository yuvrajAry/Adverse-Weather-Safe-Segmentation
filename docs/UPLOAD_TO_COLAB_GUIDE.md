# 📤 How to Upload to Google Colab - COMPLETE GUIDE

## ⚠️ Important: Your Dataset is 20GB!

Your IDDAW dataset is **20GB**, which is too large to ZIP easily. Here are the best approaches:

---

## 🎯 RECOMMENDED: Method 1 - Upload Folders Directly to Google Drive

This is the **BEST** method for large datasets.

### **Step 1: Upload to Google Drive**

1. Go to https://drive.google.com
2. Click **"New"** → **"New folder"**
3. Name it **"iddaw"**
4. Open the **"iddaw"** folder

### **Step 2: Upload Project Folder**

1. Click **"New"** → **"Folder upload"**
2. Select **`D:\iddaw\pro\project`** folder
3. Wait for upload (~5-10 minutes, 300MB)

### **Step 3: Upload IDDAW Dataset**

1. Click **"New"** → **"Folder upload"**
2. Select **`D:\iddaw\pro\IDDAW`** folder
3. Wait for upload (~2-4 HOURS, 20GB) ⏰
   - **TIP**: Start this before going to bed or leaving for the day
   - Keep your computer on and connected to internet

### **Step 4: Upload Notebook**

1. Go back to **"My Drive"**
2. Click **"New"** → **"File upload"**
3. Select **`D:\iddaw\pro\IDDAW_Colab_Training.ipynb`**

### **Your Drive Structure Should Look Like:**
```
My Drive/
├── iddaw/
│   ├── project/
│   │   ├── train.py
│   │   ├── train_improved.py
│   │   ├── models.py
│   │   ├── dataset.py
│   │   └── ...
│   └── IDDAW/
│       ├── train/
│       │   ├── FOG/
│       │   ├── RAIN/
│       │   ├── SNOW/
│       │   └── LOWLIGHT/
│       └── val/
└── IDDAW_Colab_Training.ipynb
```

---

## 🚀 Method 2 - Code Only + Dataset Already in Drive

If you already have IDDAW in Drive or want to upload code separately:

### **Step 1: Create Code-Only ZIP**

```powershell
cd D:\iddaw\pro
Compress-Archive -Path project -DestinationPath iddaw_code.zip
```

Size: ~300MB (much faster!)

### **Step 2: Upload**

1. Upload `iddaw_code.zip` to Google Drive
2. Upload IDDAW folder separately (or use existing one)
3. Upload notebook

---

## 💻 Method 3 - Use Colab's Direct Upload (Not Recommended for Large Files)

Only use this for testing with small datasets.

---

## 📝 After Upload: Using in Colab

### **Step 1: Open Notebook**

1. In Google Drive, right-click **`IDDAW_Colab_Training.ipynb`**
2. Select **"Open with"** → **"Google Colaboratory"**

### **Step 2: Enable GPU**

1. **Runtime** → **Change runtime type**
2. **Hardware accelerator** → **GPU**
3. Click **Save**

### **Step 3: Mount Drive**

Run this cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### **Step 4: Navigate to Project**

Run this cell:
```python
import os
os.chdir('/content/drive/MyDrive/iddaw')
print(f"Working directory: {os.getcwd()}")

# Verify files
!ls -la project/
!ls -la IDDAW/
```

### **Step 5: Install Dependencies**

```python
!pip install -q opencv-python-headless pillow numpy
```

### **Step 6: Train!**

```python
# Train Mid-Fusion (best model)
!python -m project.train_improved \
    --modality mid \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --amp \
    --scheduler cosine \
    --class_weights \
    --out_dir project/ckpts
```

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Upload project folder (300MB) | 5-10 min |
| Upload IDDAW dataset (20GB) | 2-4 hours |
| Upload notebook (50KB) | 10 seconds |
| Training (50 epochs) | 2-3 hours |
| **Total** | **~5-7 hours** |

**TIP**: Upload dataset overnight, train the next day!

---

## 🎯 Quick Start (Fastest Way)

If you want to start training ASAP:

### **Option A: Use Subset of Data**

1. Create a smaller dataset (1000 images instead of all)
2. ZIP and upload quickly
3. Train and test the pipeline
4. Later, train on full dataset

### **Option B: Upload Overnight**

1. Start uploading IDDAW to Drive before bed
2. Next morning, it's ready
3. Train during the day

---

## 🔍 Verify Upload is Complete

### **Check File Counts**

In Colab, after mounting Drive:

```python
import os

# Count files in project
project_files = sum([len(files) for r, d, files in os.walk('project')])
print(f"Project files: {project_files}")

# Count IDDAW images
iddaw_files = sum([len(files) for r, d, files in os.walk('IDDAW')])
print(f"IDDAW files: {iddaw_files}")

# Should be around:
# Project: ~50-100 files
# IDDAW: ~3000-4000 files
```

### **Check Specific Files**

```python
import os

required_files = [
    'project/train_improved.py',
    'project/models.py',
    'project/dataset.py',
    'project/splits/train.csv',
    'project/splits/val.csv',
    'IDDAW/train/FOG',
    'IDDAW/train/RAIN',
]

for f in required_files:
    exists = os.path.exists(f)
    status = "✓" if exists else "✗"
    print(f"{status} {f}")
```

---

## 🆘 Troubleshooting

### **Upload Stuck/Slow**

- Check internet connection
- Try uploading smaller folders first
- Use Google Drive desktop app for large uploads

### **"Quota Exceeded"**

- Free Google Drive: 15GB limit
- Your IDDAW is 20GB
- **Solution**: 
  - Delete old files in Drive
  - Or upgrade to Google One (100GB for $2/month)
  - Or use a subset of data

### **Upload Failed**

- Resume upload (Drive usually auto-resumes)
- Or delete partial upload and restart

---

## 💡 Pro Tips

### **1. Compress IDDAW Images**

If IDDAW is too large, compress images:

```powershell
# Resize images to 512x512 (they might be larger)
# This can reduce size by 50-70%
```

### **2. Use Google Drive Desktop App**

- Download: https://www.google.com/drive/download/
- Sync folders automatically
- More reliable for large uploads

### **3. Upload in Batches**

Upload one weather condition at a time:
- Upload FOG folder
- Upload RAIN folder
- etc.

---

## 🎯 My Recommendation

**For your 20GB dataset:**

1. **Tonight**: Start uploading IDDAW to Google Drive (2-4 hours)
2. **Tomorrow morning**: Upload project folder (10 min) and notebook
3. **Tomorrow**: Train on Colab (2-3 hours)
4. **Tomorrow evening**: Download improved models!

**Total time**: ~6-8 hours spread over 2 days

---

## ✅ Checklist

Before starting training in Colab, verify:

- [ ] Project folder uploaded to Drive
- [ ] IDDAW folder uploaded to Drive
- [ ] Notebook uploaded to Drive
- [ ] Opened notebook in Colab
- [ ] GPU enabled (Runtime → Change runtime type)
- [ ] Drive mounted successfully
- [ ] Can see files with `!ls -la`
- [ ] Dependencies installed
- [ ] Ready to train!

---

**Start uploading now! The sooner you start, the sooner you can train!** 🚀

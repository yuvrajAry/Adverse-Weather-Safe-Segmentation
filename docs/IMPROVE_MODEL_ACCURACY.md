# 🎯 IDDAW Model Accuracy Improvement Guide

## Current Setup Analysis

Your current training configuration:
- **Models**: RGB, NIR, Early4, Mid-fusion (MobileNetV3 + FastSCNN)
- **Image Size**: 512x512
- **Batch Size**: 4
- **Learning Rate**: 1e-3
- **Epochs**: 10 (default)
- **Optimizer**: AdamW with 1e-4 weight decay
- **Augmentations**: Flip, color jitter, noise, fog, rain

---

## 🚀 Top 10 Ways to Improve Accuracy

### 1. **Train for More Epochs** ⭐⭐⭐
**Impact**: HIGH | **Effort**: LOW

Your default is only 10 epochs, which is very low for segmentation.

```bash
# Train mid-fusion model for 50 epochs
python -m project.train \
    --modality mid \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --amp \
    --resume
```

**Recommended**: 50-100 epochs for good results

---

### 2. **Use Learning Rate Scheduling** ⭐⭐⭐
**Impact**: HIGH | **Effort**: MEDIUM

Add cosine annealing or step decay to your training.

**File**: `project/train.py`

Add after optimizer creation:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=args.epochs, 
    eta_min=1e-6
)
```

Add after each epoch:
```python
scheduler.step()
```

---

### 3. **Increase Batch Size** ⭐⭐
**Impact**: MEDIUM | **Effort**: LOW

Larger batch size = more stable gradients

```bash
# If you have enough GPU memory
python -m project.train \
    --modality mid \
    --batch_size 16 \  # or 32 if possible
    --epochs 50 \
    --amp
```

**Note**: If you get OOM, reduce image_size to 384 or 256

---

### 4. **Add More Augmentations** ⭐⭐⭐
**Impact**: HIGH | **Effort**: MEDIUM

Your current augmentations are good but can be improved.

**File**: `project/augment.py`

Add these functions:
```python
def _random_crop(image: np.ndarray, nir: np.ndarray, mask: np.ndarray, crop_size: float = 0.8):
    """Random crop to crop_size of original"""
    H, W = image.shape[:2]
    new_h, new_w = int(H * crop_size), int(W * crop_size)
    top = np.random.randint(0, H - new_h + 1)
    left = np.random.randint(0, W - new_w + 1)
    
    image = image[top:top+new_h, left:left+new_w]
    nir = nir[top:top+new_h, left:left+new_w]
    mask = mask[top:top+new_h, left:left+new_w]
    
    # Resize back
    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
    nir = cv2.resize(nir, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    
    return image, nir, mask

def _random_rotate(image: np.ndarray, nir: np.ndarray, mask: np.ndarray, max_angle: float = 15):
    """Random rotation"""
    angle = np.random.uniform(-max_angle, max_angle)
    H, W = image.shape[:2]
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    nir = cv2.warpAffine(nir, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    return image, nir, mask

def _random_scale(image: np.ndarray, nir: np.ndarray, mask: np.ndarray, scale_range: tuple = (0.8, 1.2)):
    """Random scaling"""
    scale = np.random.uniform(*scale_range)
    H, W = image.shape[:2]
    new_h, new_w = int(H * scale), int(W * scale)
    
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    nir = cv2.resize(nir, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Crop or pad to original size
    if scale > 1.0:  # Crop
        top = (new_h - H) // 2
        left = (new_w - W) // 2
        image = image[top:top+H, left:left+W]
        nir = nir[top:top+H, left:left+W]
        mask = mask[top:top+H, left:left+W]
    else:  # Pad
        pad_h = (H - new_h) // 2
        pad_w = (W - new_w) // 2
        image = cv2.copyMakeBorder(image, pad_h, H-new_h-pad_h, pad_w, W-new_w-pad_w, cv2.BORDER_REFLECT)
        nir = cv2.copyMakeBorder(nir, pad_h, H-new_h-pad_h, pad_w, W-new_w-pad_w, cv2.BORDER_REFLECT)
        mask = cv2.copyMakeBorder(mask, pad_h, H-new_h-pad_h, pad_w, W-new_w-pad_w, cv2.BORDER_CONSTANT, value=255)
    
    return image, nir, mask
```

Update `Augmenter.__call__`:
```python
def __call__(self, *, image: np.ndarray, nir: np.ndarray, mask: np.ndarray):
    if not self.enable:
        return {"image": image, "nir": nir, "mask": mask}
    
    # Geometric augmentations
    image, nir, mask = _random_flip(image, nir, mask)
    
    if np.random.rand() < 0.5:
        image, nir, mask = _random_crop(image, nir, mask)
    
    if np.random.rand() < 0.3:
        image, nir, mask = _random_rotate(image, nir, mask)
    
    if np.random.rand() < 0.5:
        image, nir, mask = _random_scale(image, nir, mask)
    
    # Color augmentations (RGB only)
    if np.random.rand() < 0.8:
        image = _color_jitter(image)
    
    if np.random.rand() < 0.3:
        image = _gaussian_noise(image)
    
    # Weather augmentations
    if np.random.rand() < 0.2:  # Increased from 0.15
        image = _synthetic_fog(image)
    
    if np.random.rand() < 0.2:  # Increased from 0.15
        image = _synthetic_rain(image)
    
    return {"image": image, "nir": nir, "mask": mask}
```

---

### 5. **Use Class Weights** ⭐⭐⭐
**Impact**: HIGH | **Effort**: MEDIUM

Handle class imbalance (e.g., road is much more common than pedestrians).

**File**: `project/train.py`

Add function to compute class weights:
```python
def compute_class_weights(loader: DataLoader, num_classes: int, ignore_index: int):
    """Compute inverse frequency class weights"""
    class_counts = torch.zeros(num_classes)
    
    for batch in loader:
        mask = batch["mask"]
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum()
    
    # Inverse frequency
    total = class_counts.sum()
    weights = total / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalize
    weights[ignore_index] = 0  # Don't weight ignore class
    
    return weights
```

Update criterion:
```python
# After creating dataloaders
print("Computing class weights...", flush=True)
class_weights = compute_class_weights(train_loader, len(lbl.id_to_class), lbl.ignore_index)
print(f"Class weights: {class_weights}", flush=True)

criterion = nn.CrossEntropyLoss(
    ignore_index=lbl.ignore_index,
    weight=class_weights.to(device)
)
```

---

### 6. **Use Larger Backbone** ⭐⭐
**Impact**: MEDIUM-HIGH | **Effort**: MEDIUM

MobileNetV3-Small is lightweight. Try MobileNetV3-Large or ResNet.

**File**: `project/models.py`

Add ResNet50 backbone:
```python
from torchvision.models import resnet50

class ResNet50Seg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.encoder = resnet50(weights=None)
        
        # Adjust first conv
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove FC layers
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_size = x.shape[-2:]
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=inp_size, mode="bilinear", align_corners=False)
        return x
```

---

### 7. **Multi-Scale Training** ⭐⭐
**Impact**: MEDIUM | **Effort**: HIGH

Train on multiple resolutions.

**File**: `project/train.py`

```python
# In training loop
scales = [384, 448, 512, 576]
current_scale = scales[epoch % len(scales)]

# Update dataset image size dynamically
train_loader.dataset.size = (current_scale, current_scale)
```

---

### 8. **Use Focal Loss** ⭐⭐
**Impact**: MEDIUM | **Effort**: LOW

Better for hard examples.

**File**: `project/train.py`

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Use it
criterion = FocalLoss(ignore_index=lbl.ignore_index)
```

---

### 9. **Ensemble Models** ⭐⭐⭐
**Impact**: HIGH | **Effort**: LOW (for inference)

Combine predictions from multiple models.

**File**: `project/backend_with_models.py`

```python
# In process_images_with_models function
def ensemble_predict(models_dict, rgb_tensor, nir_tensor):
    """Average predictions from multiple models"""
    all_logits = []
    
    # Mid-fusion model
    if 'mid' in models_dict:
        with torch.no_grad():
            logits = models_dict['mid'](rgb_tensor, nir_tensor)
            all_logits.append(logits)
    
    # RGB model
    if 'rgb' in models_dict:
        with torch.no_grad():
            logits = models_dict['rgb'](rgb_tensor)
            all_logits.append(logits)
    
    # Early4 model
    if 'early4' in models_dict:
        early4_input = torch.cat([rgb_tensor, nir_tensor], dim=1)
        with torch.no_grad():
            logits = models_dict['early4'](early4_input)
            all_logits.append(logits)
    
    # Average
    ensemble_logits = torch.stack(all_logits).mean(dim=0)
    return ensemble_logits
```

---

### 10. **Use Pre-trained Weights** ⭐⭐⭐
**Impact**: HIGH | **Effort**: LOW

Initialize with ImageNet weights.

**File**: `project/models.py`

```python
# In MobileNetV3LiteSeg
self.encoder = mobilenet_v3_small(weights='IMAGENET1K_V1')  # Add pre-trained weights
```

---

## 📊 Quick Wins (Do These First)

### Priority 1: Train Longer
```bash
python -m project.train \
    --modality mid \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --amp \
    --resume
```

### Priority 2: Add Learning Rate Scheduler
Just add 3 lines of code (see #2 above)

### Priority 3: Use Class Weights
Add the function and update criterion (see #5 above)

---

## 🎯 Expected Improvements

| Change | Expected mIoU Gain |
|--------|-------------------|
| Train 50 epochs instead of 10 | +5-10% |
| Add LR scheduler | +2-5% |
| Class weights | +3-7% |
| More augmentations | +2-4% |
| Larger backbone | +3-6% |
| Ensemble | +2-5% |
| **Total Potential** | **+15-35%** |

---

## 🔍 Monitor Training

Check your training logs:
```bash
# View training progress
tail -f project/ckpts/training.log

# Check best mIoU
python -c "import torch; ckpt=torch.load('project/ckpts/best_mid_mbv3.pt'); print(f'Best mIoU: {ckpt[\"miou\"]:.3f}')"
```

---

## 🚀 Recommended Training Command

```bash
# Best configuration for improved accuracy
python -m project.train \
    --modality mid \
    --backbone mbv3 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --image_size 512 \
    --workers 4 \
    --amp \
    --resume \
    --out_dir project/ckpts
```

---

## 📝 Next Steps

1. **Implement Priority 1-3** (quick wins)
2. **Train for 50-100 epochs**
3. **Monitor validation mIoU**
4. **Add more augmentations** if still underfitting
5. **Try ensemble** for final boost

---

**Start with training longer - that alone will give you the biggest improvement!**

"""
Enhanced augmentations for better training
Drop-in replacement for augment.py
"""

from __future__ import annotations

import numpy as np
import cv2


def _random_flip(image: np.ndarray, nir: np.ndarray, mask: np.ndarray):
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        nir = np.fliplr(nir)
        mask = np.fliplr(mask)
    return image, nir, mask


def _color_jitter(image: np.ndarray, strength: float = 0.3):
    """Enhanced color jitter"""
    alpha = 1.0 + strength * (np.random.rand() * 2 - 1)  # contrast
    beta = 30.0 * (np.random.rand() * 2 - 1)  # brightness
    
    # Also adjust saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + 0.2 * (np.random.rand() * 2 - 1))
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    img = image.astype(np.float32) * alpha + beta
    return np.clip(img, 0, 255).astype(np.uint8)


def _gaussian_noise(image: np.ndarray, sigma: float = 8.0):
    """Stronger noise for robustness"""
    noise = np.random.randn(*image.shape).astype(np.float32) * sigma
    img = image.astype(np.float32) + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def _gaussian_blur(image: np.ndarray):
    """Random blur for weather simulation"""
    ksize = np.random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def _synthetic_fog(image: np.ndarray, strength: float = None):
    """Enhanced fog with variable strength"""
    if strength is None:
        strength = np.random.uniform(0.02, 0.08)
    
    H, W = image.shape[:2]
    X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    dist = np.sqrt(X * X + Y * Y)
    fog = (np.random.rand(H, W) * 0.5 + 0.5) * (1 - np.clip(dist, 0, 1))
    fog = cv2.GaussianBlur(fog, (0, 0), sigmaX=H * 0.03)
    
    out = image.astype(np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - strength) + 255.0 * fog * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def _synthetic_rain(image: np.ndarray, density: float = None):
    """Enhanced rain with variable density"""
    if density is None:
        density = np.random.uniform(0.001, 0.004)
    
    H, W = image.shape[:2]
    n_drops = int(H * W * density)
    img = image.copy()
    canvas = np.zeros((H, W), dtype=np.uint8)
    
    # Variable rain parameters
    length = np.random.randint(12, 20)
    angle = np.random.uniform(-40, -20)
    
    for _ in range(n_drops):
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        x2 = int(x + length * np.cos(np.deg2rad(angle)))
        y2 = int(y + length * np.sin(np.deg2rad(angle)))
        cv2.line(canvas, (x, y), (x2, y2), 255, 1)
    
    blur = cv2.blur(canvas, (3, 3))
    mask = (blur > 0).astype(np.float32)[..., None]
    img = img.astype(np.float32)
    img = img * (1 - mask) + 220.0 * mask
    return np.clip(img, 0, 255).astype(np.uint8)


def _synthetic_snow(image: np.ndarray, density: float = 0.003):
    """Add snow effect"""
    H, W = image.shape[:2]
    n_flakes = int(H * W * density)
    img = image.copy().astype(np.float32)
    
    for _ in range(n_flakes):
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        size = np.random.randint(1, 3)
        cv2.circle(img, (x, y), size, (255, 255, 255), -1)
    
    return np.clip(img, 0, 255).astype(np.uint8)


def _coarse_dropout(image: np.ndarray, mask: np.ndarray, max_holes: int = 8):
    """Random rectangular dropout for robustness"""
    H, W = image.shape[:2]
    img = image.copy()
    
    for _ in range(np.random.randint(1, max_holes + 1)):
        h = np.random.randint(H // 16, H // 8)
        w = np.random.randint(W // 16, W // 8)
        y = np.random.randint(0, H - h)
        x = np.random.randint(0, W - w)
        
        img[y:y+h, x:x+w] = 0
    
    return img, mask


def _random_scale(image: np.ndarray, nir: np.ndarray, mask: np.ndarray):
    """Random scale augmentation"""
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.75, 1.25)
        H, W = image.shape[:2]
        new_H, new_W = int(H * scale), int(W * scale)
        
        image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        nir = cv2.resize(nir, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        
        # Crop or pad to original size
        if scale > 1.0:
            # Crop center
            y = (new_H - H) // 2
            x = (new_W - W) // 2
            image = image[y:y+H, x:x+W]
            nir = nir[y:y+H, x:x+W]
            mask = mask[y:y+H, x:x+W]
        else:
            # Pad
            pad_h = (H - new_H) // 2
            pad_w = (W - new_W) // 2
            image = cv2.copyMakeBorder(image, pad_h, H - new_H - pad_h, pad_w, W - new_W - pad_w, cv2.BORDER_REFLECT)
            nir = cv2.copyMakeBorder(nir, pad_h, H - new_H - pad_h, pad_w, W - new_W - pad_w, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, pad_h, H - new_H - pad_h, pad_w, W - new_W - pad_w, cv2.BORDER_CONSTANT, value=255)
    
    return image, nir, mask


class EnhancedAugmenter:
    """Enhanced augmentation pipeline"""
    
    def __init__(self, enable: bool, aggressive: bool = False):
        self.enable = enable
        self.aggressive = aggressive  # More augmentation for longer training
    
    def __call__(self, *, image: np.ndarray, nir: np.ndarray, mask: np.ndarray):
        if not self.enable:
            return {"image": image, "nir": nir, "mask": mask}
        
        # Always apply flip
        image, nir, mask = _random_flip(image, nir, mask)
        
        # Scale augmentation
        if self.aggressive and np.random.rand() < 0.3:
            image, nir, mask = _random_scale(image, nir, mask)
        
        # Color jitter (higher probability)
        if np.random.rand() < 0.85:
            image = _color_jitter(image, strength=0.3 if self.aggressive else 0.2)
        
        # Gaussian noise
        if np.random.rand() < 0.4:
            image = _gaussian_noise(image, sigma=10.0 if self.aggressive else 8.0)
        
        # Blur (new!)
        if np.random.rand() < 0.2:
            image = _gaussian_blur(image)
        
        # Weather augmentations (increased probability)
        weather_roll = np.random.rand()
        if weather_roll < 0.25:  # 25% fog
            image = _synthetic_fog(image)
        elif weather_roll < 0.45:  # 20% rain
            image = _synthetic_rain(image)
        elif weather_roll < 0.55:  # 10% snow
            image = _synthetic_snow(image)
        
        # Coarse dropout for robustness
        if self.aggressive and np.random.rand() < 0.3:
            image, mask = _coarse_dropout(image, mask)
        
        return {"image": image, "nir": nir, "mask": mask}


def build_augmentations(augment: bool):
    """Build enhanced augmenter"""
    return EnhancedAugmenter(enable=augment, aggressive=False)


def build_aggressive_augmentations(augment: bool):
    """Build aggressive augmenter for longer training"""
    return EnhancedAugmenter(enable=augment, aggressive=True)

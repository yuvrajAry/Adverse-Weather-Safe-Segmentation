import numpy as np
from PIL import Image
import cv2


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min < 1e-8:
        return (np.zeros_like(x) * 255).astype(np.uint8)
    x_norm = (x - x_min) / (x_max - x_min)
    return (x_norm * 255).astype(np.uint8)


def overlay_heatmap_on_image(rgb_pil: Image.Image, heatmap: np.ndarray):
    if heatmap is None:
        return None
    rgb = np.array(rgb_pil)
    hm_uint8 = normalize_to_uint8(heatmap)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    alpha = 0.5
    overlay = (alpha * hm_color + (1 - alpha) * rgb).astype(np.uint8)
    return Image.fromarray(overlay)


def colorize_mask(mask_np: np.ndarray, num_classes: int):
    # simple palette generation
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color[mask_np == cls] = palette[cls]
    return Image.fromarray(color)


import os
import json
from typing import List, Tuple
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import numpy as np


def _find_samples_split(root: str, split: str) -> List[Tuple[str, str, str]]:
    samples: List[Tuple[str, str, str]] = []
    weather_types = ["FOG", "RAIN", "SNOW", "LOWLIGHT"]
    for weather in weather_types:
        rgb_dir = os.path.join(root, split, weather, "rgb")
        nir_dir = os.path.join(root, split, weather, "nir")
        gt_dir = os.path.join(root, split, weather, "gtSeg")
        if not os.path.isdir(rgb_dir):
            continue
        for city in os.listdir(rgb_dir):
            city_rgb_dir = os.path.join(rgb_dir, city)
            if not os.path.isdir(city_rgb_dir):
                continue
            for fname in os.listdir(city_rgb_dir):
                if not fname.endswith("_rgb.png"):
                    continue
                stem = fname.replace("_rgb.png", "")
                rgb_path = os.path.join(city_rgb_dir, fname)
                nir_path = os.path.join(nir_dir, city, f"{stem}_nir.png")
                mask_path = os.path.join(gt_dir, city, f"{stem}_mask.json")
                if os.path.isfile(rgb_path) and os.path.isfile(nir_path) and os.path.isfile(mask_path):
                    samples.append((rgb_path, nir_path, mask_path))
    if len(samples) == 0:
        raise RuntimeError(f"No samples found under {root}/{split} (expected weather/rgb|nir|gtSeg structure)")
    return samples


def _rasterize_mask_from_json(json_path: str, size: Tuple[int, int], num_classes: int) -> np.ndarray:
    with open(json_path, 'r') as f:
        data = json.load(f)

    width, height = size
    # Initialize background as class 0 (valid). Using 255 everywhere can lead to
    # all-ignored targets and NaN losses. We'll reserve 255 only for explicit ignores.
    mask = Image.new('L', (width, height), color=0)
    draw = ImageDraw.Draw(mask)

    # Case 1: direct 2D array stored
    if isinstance(data, dict) and 'mask' in data:
        arr = np.array(data['mask'], dtype=np.int64)
        if arr.shape != (height, width):
            arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((width, height), Image.NEAREST))
        arr = np.clip(arr, 0, num_classes - 1, out=arr)
        return arr

    # Case 2: COCO-like annotations list with polygons and category_id
    anns = None
    if 'annotations' in data and isinstance(data['annotations'], list):
        anns = data['annotations']
    elif 'objects' in data and isinstance(data['objects'], list):
        anns = data['objects']
    elif 'shapes' in data and isinstance(data['shapes'], list):
        anns = data['shapes']

    if anns is not None:
        for obj in anns:
            # Try to get class id/name
            cls_id = obj.get('category_id') if isinstance(obj, dict) else None
            if cls_id is None and 'label' in obj:
                try:
                    cls_id = int(obj['label'])
                except Exception:
                    cls_id = None
            if cls_id is None:
                cls_id = obj.get('class') if isinstance(obj, dict) else None
            if cls_id is None:
                continue
            try:
                cls_id = int(cls_id)
            except Exception:
                continue
            if cls_id < 0 or cls_id >= num_classes:
                continue

            # Extract polygon(s)
            poly = None
            if 'polygon' in obj and isinstance(obj['polygon'], list):
                poly = obj['polygon']
            elif 'segmentation' in obj and isinstance(obj['segmentation'], list) and len(obj['segmentation']) > 0:
                # COCO stores list of flat coordinates lists
                poly = obj['segmentation'][0]
            elif 'points' in obj and isinstance(obj['points'], list):
                poly = obj['points']

            if poly is None:
                continue

            # Normalize polygon to list of (x,y) tuples
            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in poly):
                pts = [(float(x), float(y)) for x, y in poly]
            else:
                # flat list
                it = iter(poly)
                pts = [(float(x), float(y)) for x, y in zip(it, it)]

            # Draw polygon
            draw.polygon(pts, fill=int(cls_id))

        arr = np.array(mask, dtype=np.int64)
        # Ensure label range is valid
        arr = np.clip(arr, 0, num_classes - 1, out=arr)
        return arr

    # Fallback: empty mask → background (class 0)
    arr = np.array(mask, dtype=np.int64)
    arr = np.clip(arr, 0, num_classes - 1, out=arr)
    return arr


class IDDAWDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        size: Tuple[int, int] = (512, 512),
        mean = (0.485, 0.456, 0.406, 0.5),
        std = (0.229, 0.224, 0.225, 0.25),
        fusion_mode: str = "early",
        num_classes: int = 20,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.fusion_mode = fusion_mode
        self.num_classes = num_classes

        self.samples = _find_samples_split(root, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rgb_path, nir_path, mask_path = self.samples[idx]
        rgb = Image.open(rgb_path).convert("RGB").resize(self.size, Image.BILINEAR)
        nir = Image.open(nir_path).convert("L").resize(self.size, Image.BILINEAR)
        mask_np = _rasterize_mask_from_json(mask_path, self.size, self.num_classes)

        rgb_np = np.array(rgb).astype(np.float32) / 255.0
        nir_np = np.array(nir).astype(np.float32) / 255.0
        nir_np = nir_np[:, :, None]
        x = np.concatenate([rgb_np, nir_np], axis=2)
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))  # C,H,W
        x_t = torch.from_numpy(x)

        # Sanitize labels to valid range [0, num_classes-1]
        mask_np = np.clip(mask_np, 0, self.num_classes - 1)
        y = torch.from_numpy(mask_np.astype(np.int64))
        return x_t, y


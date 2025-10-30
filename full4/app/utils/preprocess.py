from typing import Tuple
from PIL import Image
import numpy as np
import torch


def preprocess_pair(rgb_img: Image.Image, nir_img: Image.Image, size: Tuple[int, int], fusion_mode: str,
                    mean, std):
    width, height = size
    rgb_resized = rgb_img.resize((width, height), Image.BILINEAR)
    nir_resized = nir_img.resize((width, height), Image.BILINEAR)

    rgb_np = np.array(rgb_resized).astype(np.float32) / 255.0
    nir_np = np.array(nir_resized.convert('L')).astype(np.float32) / 255.0

    if fusion_mode == 'early':
        # 4-channel input
        nir_ch = nir_np[:, :, None]
        x = np.concatenate([rgb_np, nir_ch], axis=2)
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        x = (x - mean_arr) / std_arr
        x = np.transpose(x, (2, 0, 1))  # C,H,W
        tensor = torch.from_numpy(x).unsqueeze(0)  # 1,C,H,W
        return tensor, rgb_resized
    else:
        # mid-level: return stacked RGB and NIR as separate inputs in one tensor: 4 channels but model splits
        nir_ch = nir_np[:, :, None]
        x = np.concatenate([rgb_np, nir_ch], axis=2)
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        x = (x - mean_arr) / std_arr
        x = np.transpose(x, (2, 0, 1))
        tensor = torch.from_numpy(x).unsqueeze(0)
        return tensor, rgb_resized


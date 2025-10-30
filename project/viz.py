from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import cv2


def color_map(num_classes: int) -> np.ndarray:
	cm = np.zeros((num_classes, 3), dtype=np.uint8)
	for i in range(num_classes):
		cm[i] = ((37 * i) % 255, (17 * i) % 255, (29 * i) % 255)
	return cm


def overlay_segmentation(image: np.ndarray, mask: np.ndarray, cm: np.ndarray, alpha: float = 0.5) -> np.ndarray:
	color = cm[mask.clip(0, len(cm)-1)]
	image = image.copy()
	if image.dtype != np.uint8:
		image = (image * 255.0).clip(0, 255).astype(np.uint8)
	return cv2.addWeighted(image, 1 - alpha, color, alpha, 0)


def safety_heatmap(image: np.ndarray, mask: np.ndarray, safety_groups: Dict[str, Tuple[int, ...]], id_to_class: Dict[int, str], alpha: float = 0.6) -> np.ndarray:
	heat = np.zeros(image.shape[:2], dtype=np.float32)
	class_to_id = {v: k for k, v in id_to_class.items()}
	for group, names in safety_groups.items():
		ids = [class_to_id[n] for n in names if n in class_to_id]
		for cid in ids:
			heat = np.maximum(heat, (mask == cid).astype(np.float32))
	heat = (cv2.GaussianBlur(heat, (0, 0), 5) * 255).astype(np.uint8)
	jet = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
	if image.dtype != np.uint8:
		image = (image * 255.0).clip(0, 255).astype(np.uint8)
	return cv2.addWeighted(image, 1 - alpha, jet, alpha, 0)


def confidence_heatmap(image: np.ndarray, entropy: np.ndarray, alpha: float = 0.6) -> np.ndarray:
	"""Generate confidence-based heatmap with color coding:
	- Green: High confidence (low entropy)
	- Yellow: Medium confidence (medium entropy) 
	- Red: Low confidence (high entropy)
	"""
	if image.dtype != np.uint8:
		image = (image * 255.0).clip(0, 255).astype(np.uint8)
	
	# Use percentile-based normalization for better color distribution
	ent_min, ent_max = np.percentile(entropy, [10, 90])  # Use 10th and 90th percentiles
	ent_norm = np.clip((entropy - ent_min) / (ent_max - ent_min + 1e-8), 0, 1)
	
	# Create confidence heatmap: 0=high conf (green), 1=low conf (red)
	conf_heat = 1.0 - ent_norm  # Invert: low entropy = high confidence
	
	# Apply Gaussian blur for smooth transitions
	conf_heat = cv2.GaussianBlur(conf_heat, (0, 0), 7)
	
	# Create color map using OpenCV's built-in colormap for better quality
	# Scale to 0-255 for colormap
	conf_scaled = (conf_heat * 255).astype(np.uint8)
	
	# Use JET colormap: Red=0, Yellow=64, Green=128, Cyan=192, Blue=255
	# We want: Green=high conf, Yellow=medium conf, Red=low conf
	jet_map = cv2.applyColorMap(conf_scaled, cv2.COLORMAP_JET)
	
	# Create custom color mapping: Green -> Yellow -> Red
	heat_colored = np.zeros_like(jet_map)
	
	# Map confidence values to colors
	# 0-0.33: Green to Yellow (high to medium confidence)
	# 0.33-0.66: Yellow to Orange (medium confidence)
	# 0.66-1.0: Orange to Red (low confidence)
	
	mask_green_yellow = conf_heat <= 0.33
	mask_yellow_orange = (conf_heat > 0.33) & (conf_heat <= 0.66)
	mask_orange_red = conf_heat > 0.66
	
	# Green to Yellow
	if np.any(mask_green_yellow):
		val = conf_heat[mask_green_yellow] / 0.33  # Scale to [0,1]
		heat_colored[mask_green_yellow, 0] = (val * 255).astype(np.uint8)  # Red channel
		heat_colored[mask_green_yellow, 1] = 255  # Green channel
		heat_colored[mask_green_yellow, 2] = 0    # Blue channel
	
	# Yellow to Orange
	if np.any(mask_yellow_orange):
		val = (conf_heat[mask_yellow_orange] - 0.33) / 0.33  # Scale to [0,1]
		heat_colored[mask_yellow_orange, 0] = 255  # Red channel
		heat_colored[mask_yellow_orange, 1] = ((1 - val) * 255).astype(np.uint8)  # Green channel
		heat_colored[mask_yellow_orange, 2] = 0    # Blue channel
	
	# Orange to Red
	if np.any(mask_orange_red):
		val = (conf_heat[mask_orange_red] - 0.66) / 0.34  # Scale to [0,1]
		heat_colored[mask_orange_red, 0] = 255  # Red channel
		heat_colored[mask_orange_red, 1] = ((1 - val) * 128).astype(np.uint8)  # Green channel
		heat_colored[mask_orange_red, 2] = 0    # Blue channel
	
	return cv2.addWeighted(image, 1 - alpha, heat_colored, alpha, 0)




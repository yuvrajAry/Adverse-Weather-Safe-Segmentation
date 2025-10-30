from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from .labels import LabelMapper, SafetyGroups
from .augment import build_augmentations


def load_image(path: str) -> Image.Image:
	img = Image.open(path).convert("RGB")
	return img


def load_mask_polygon_json(path: str, class_to_id: Dict[str, int], ignore_index: int) -> np.ndarray:
	with open(path, "r") as f:
		data = json.load(f)
	H, W = int(data["imgHeight"]), int(data["imgWidth"])
	mask = Image.new("I", (W, H), color=ignore_index)
	draw = ImageDraw.Draw(mask)
	for obj in data.get("objects", []):
		label = obj.get("label", "")
		poly = obj.get("polygon", [])
		# Validate polygon: need at least 3 coordinates to form a filled polygon
		if not poly or len(poly) < 3:
			continue
		# Coerce to tuples and filter malformed points
		pts = []
		for p in poly:
			try:
				x, y = float(p[0]), float(p[1])
				pts.append((x, y))
			except Exception:
				continue
		if len(pts) < 3:
			continue
		cid = class_to_id.get(label, ignore_index)
		# Draw filled polygon for this class id
		try:
			draw.polygon(pts, fill=int(cid))
		except Exception:
			# Skip degenerate polygons that PIL refuses
			continue
	return np.array(mask, dtype=np.int64)


class IDDAWDataset(Dataset):
	def __init__(
		self,
		csv_path: str,
		mode: str,
		label_mapper: Optional[LabelMapper] = None,
		augment: bool = False,
		fusion: str = "rgb",  # rgb | nir | early4 | mid
		image_size: Optional[Tuple[int, int]] = None,
		safety_groups: Optional[SafetyGroups] = None,
	):
		assert fusion in {"rgb", "nir", "early4", "mid"}
		self.samples = self._read_csv(csv_path)
		self.mode = mode
		self.fusion = fusion
		self.size = image_size
		self.augment = build_augmentations(augment=augment)
		self.label_mapper = label_mapper or LabelMapper.default()
		self.safety_groups = safety_groups or SafetyGroups.default()
		self.ignore_index = self.label_mapper.ignore_index

	def _read_csv(self, csv_path: str) -> List[Dict[str, str]]:
		rows: List[Dict[str, str]] = []
		with open(csv_path, "r") as f:
			headers = None
			for i, line in enumerate(f):
				parts = [p.strip() for p in line.rstrip("\n").split(",")]
				if i == 0:
					headers = parts
					continue
				if not parts or not headers:
					continue
				row = {h: v for h, v in zip(headers, parts)}
				rows.append(row)
		return rows

	def __len__(self) -> int:
		return len(self.samples)

	def _resize_if_needed(self, img: Image.Image) -> Image.Image:
		if self.size is None:
			return img
		return img.resize(self.size, resample=Image.BILINEAR)

	def _resize_mask_if_needed(self, mask: Image.Image) -> Image.Image:
		if self.size is None:
			return mask
		return mask.resize(self.size, resample=Image.NEAREST)

	def __getitem__(self, idx: int):
		r = self.samples[idx]
		rgb = load_image(r["rgb_path"])  # RGB
		nir_rgb = load_image(r["nir_path"])  # 3-channel but grayscale content
		# Convert NIR to single channel from its red channel (image is replicated)
		nir = np.array(nir_rgb)[:, :, 0:1]

		mask_np = load_mask_polygon_json(r["mask_path"], self.label_mapper.class_to_id, self.ignore_index)
		mask_img = Image.fromarray(mask_np.astype(np.int32), mode="I")

		# Resize
		rgb = self._resize_if_needed(rgb)
		nir_img = self._resize_if_needed(Image.fromarray(np.concatenate([nir, nir, nir], axis=2)))
		mask_img = self._resize_mask_if_needed(mask_img)

		# Augment jointly (expects numpy arrays HxWxC and HxW)
		aug = self.augment(image=np.array(rgb), nir=np.array(nir_img), mask=np.array(mask_img, dtype=np.int64))
		rgb_np = aug["image"]
		nir_np = aug["nir"][:, :, 0:1]  # keep single channel
		mask_np = aug["mask"].astype(np.int64)

		# Normalize to [0,1]
		rgb_np = rgb_np.astype(np.float32) / 255.0
		nir_np = nir_np.astype(np.float32) / 255.0

		if self.fusion == "rgb":
			img_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
		elif self.fusion == "nir":
			img_tensor = torch.from_numpy(nir_np).permute(2, 0, 1)
		elif self.fusion == "early4":
			img_tensor = torch.from_numpy(np.concatenate([rgb_np, nir_np], axis=2)).permute(2, 0, 1)
		else:  # mid
			img_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
			nir_tensor = torch.from_numpy(nir_np).permute(2, 0, 1)
			return {
				"image_rgb": img_tensor,
				"image_nir": nir_tensor,
				"mask": torch.from_numpy(mask_np),
				"meta": {"weather": r["weather"], "sequence": r["sequence"], "frame": r["frame"]},
			}

		return {
			"image": img_tensor,
			"mask": torch.from_numpy(mask_np),
			"meta": {"weather": r["weather"], "sequence": r["sequence"], "frame": r["frame"]},
		}



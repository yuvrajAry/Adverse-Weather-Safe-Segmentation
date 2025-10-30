from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import IDDAWDataset
from .labels import LabelMapper, SafetyGroups
from .models import build_model
from .metrics import entropy_map
from .viz import color_map, overlay_segmentation, safety_heatmap, confidence_heatmap


def to_numpy_img(t: torch.Tensor) -> np.ndarray:
	# CxHxW -> HxWxC, [0,1]
	arr = t.detach().cpu().float().numpy().transpose(1, 2, 0)
	arr = np.clip(arr, 0.0, 1.0)
	return arr


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--splits_dir", type=str, default="project/splits")
	parser.add_argument("--ckpt", type=str, required=True)
	parser.add_argument("--modality", type=str, choices=["rgb", "nir", "early4", "mid"], default="rgb")
	parser.add_argument("--backbone", type=str, choices=["mbv3", "fastscnn"], default="mbv3")
	parser.add_argument("--image_size", type=int, default=512)
	parser.add_argument("--num_samples", type=int, default=8)
	parser.add_argument("--out_dir", type=str, default="project/outputs")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	lbl = LabelMapper.default()
	val_ds = IDDAWDataset(f"{args.splits_dir}/val.csv", mode="val", label_mapper=lbl, augment=False, fusion=args.modality, image_size=(args.image_size, args.image_size))
	loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

	variant = f"{args.modality}_{args.backbone}" if args.modality != "mid" else "mid_mbv3"
	model = build_model(variant, num_classes=len(lbl.id_to_class)).to(device)
	state = torch.load(args.ckpt, map_location=device)
	model.load_state_dict(state["model"])
	model.eval()

	cm = color_map(len(lbl.id_to_class))
	Path(args.out_dir).mkdir(parents=True, exist_ok=True)
	sub = Path(args.out_dir) / variant
	sub.mkdir(parents=True, exist_ok=True)

	safety_groups = SafetyGroups.default().group_to_classes

	saved = 0
	errors = 0
	with torch.no_grad():
		for i, batch in enumerate(loader):
			if i % 50 == 0:
				print(f"Processing batch {i}, saved {saved}, errors {errors}", flush=True)
			if saved >= args.num_samples:
				break
			try:
				meta = batch["meta"]
				frame = meta["frame"][0]
				weather = meta["weather"][0]
				if args.modality == "mid":
					logits = model(batch["image_rgb"].to(device), batch["image_nir"].to(device))
					img_np = to_numpy_img(batch["image_rgb"][0])
				else:
					logits = model(batch["image"].to(device))
					img_np = to_numpy_img(batch["image"][0])
					# For early4 fusion, take RGB view for visualization (first 3 channels)
					if args.modality == "early4" and img_np.shape[2] >= 3:
						img_np = img_np[:, :, :3]
				# Ensure visualization image is 3-channel for overlay functions
				if img_np.ndim == 2:
					img_np = np.stack([img_np, img_np, img_np], axis=2)
				elif img_np.shape[2] == 1:
					img_np = np.repeat(img_np, 3, axis=2)
				pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int32)
				ent = entropy_map(logits)[0].cpu().numpy()

				# Normalize entropy to [0,255] for visualization
				ent_norm = ent - ent.min()
				if ent_norm.max() > 0:
					ent_norm = ent_norm / ent_norm.max()
				ent_vis = (ent_norm * 255).astype(np.uint8)

				overlay = overlay_segmentation((img_np * 255).astype(np.uint8), pred, cm, alpha=0.5)
				shm = safety_heatmap((img_np * 255).astype(np.uint8), pred, safety_groups, lbl.id_to_class, alpha=0.6)
				conf_hm = confidence_heatmap((img_np * 255).astype(np.uint8), ent, alpha=0.6)

				base = f"{weather}_{frame}"
				out_overlay = sub / f"{base}_overlay.png"
				out_safety = sub / f"{base}_safety.png"
				out_entropy = sub / f"{base}_entropy.png"
				out_confidence = sub / f"{base}_confidence.png"

				import cv2
				cv2.imwrite(str(out_overlay), overlay[:, :, ::1])
				cv2.imwrite(str(out_safety), shm[:, :, ::1])
				cv2.imwrite(str(out_entropy), ent_vis)
				cv2.imwrite(str(out_confidence), conf_hm[:, :, ::1])
				saved += 1
			except Exception as e:
				errors += 1
				if errors < 10:  # Only print first 10 errors
					print(f"Error processing batch {i}: {e}", flush=True)
				continue

	print(f"Saved {saved} samples to {sub}, {errors} errors encountered")


if __name__ == "__main__":
	main()



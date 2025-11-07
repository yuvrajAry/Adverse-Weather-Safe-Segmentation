from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import IDDAWDataset
from .labels import LabelMapper, SafetyGroups
from .models import build_model
from .metrics import fast_hist, compute_miou, compute_safe_miou, entropy_map


def enable_mc_dropout(model: torch.nn.Module):
	for m in model.modules():
		if isinstance(m, torch.nn.Dropout):
			m.train()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--splits_dir", type=str, default="project/splits")
	parser.add_argument("--ckpt", type=str, required=True)
	parser.add_argument("--modality", type=str, default="rgb", choices=["rgb", "nir", "early4", "mid"])
	parser.add_argument("--backbone", type=str, default="mbv3", choices=["mbv3", "fastscnn"])
	parser.add_argument("--image_size", type=int, default=512)
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--workers", type=int, default=2)
	parser.add_argument("--mc_runs", type=int, default=5)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	lbl = LabelMapper.default()
	val_ds = IDDAWDataset(f"{args.splits_dir}/val.csv", mode="val", label_mapper=lbl, augment=False, fusion=args.modality, image_size=(args.image_size, args.image_size))
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

	variant = f"{args.modality}_{args.backbone}" if args.modality != "mid" else "mid_mbv3"
	model = build_model(variant, num_classes=len(lbl.id_to_class)).to(device)
	state = torch.load(args.ckpt, map_location=device, weights_only=True)
	model.load_state_dict(state["model"])
	model.eval()

	# Deterministic evaluation
	conf = torch.zeros((len(lbl.id_to_class), len(lbl.id_to_class)), dtype=torch.long, device=device)
	with torch.no_grad():
		for batch in val_loader:
			if args.modality == "mid":
				logits = model(batch["image_rgb"].to(device), batch["image_nir"].to(device))
			else:
				logits = model(batch["image"].to(device))
			pred = torch.argmax(logits, dim=1)
			target = batch["mask"].to(device)
			conf += fast_hist(pred, target, num_classes=len(lbl.id_to_class), ignore_index=lbl.ignore_index)
	_, mean_iou = compute_miou(conf.float())
	safety = compute_safe_miou(conf.float(), lbl.id_to_class, SafetyGroups.default().group_to_classes)
	print(f"mIoU={float(mean_iou):.3f}, safe_mIoU={safety.get('safe_mIoU',0):.3f}")

	# Uncertainty via MC Dropout and entropy
	ensemble_entropy = []
	prob_accum = None
	enable_mc_dropout(model)
	for _ in range(args.mc_runs):
		with torch.no_grad():
			for batch in val_loader:
				if args.modality == "mid":
					logits = model(batch["image_rgb"].to(device), batch["image_nir"].to(device))
				else:
					logits = model(batch["image"].to(device))
				ent = entropy_map(logits)
				ensemble_entropy.append(ent.cpu())
	print("Computed entropy maps for MC Dropout.")


if __name__ == "__main__":
	main()




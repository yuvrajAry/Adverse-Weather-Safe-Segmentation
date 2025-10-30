from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import IDDAWDataset
from .labels import LabelMapper, SafetyGroups
from .models import build_model
from .metrics import fast_hist, compute_miou, compute_safe_miou, entropy_map


def create_dataloaders(splits_dir: str, image_size: int, batch_size: int, num_workers: int, modality: str):
	lbl = LabelMapper.default()
	fusion = modality  # Map modality to fusion parameter
	train_ds = IDDAWDataset(f"{splits_dir}/train.csv", mode="train", label_mapper=lbl, augment=True, fusion=fusion, image_size=(image_size, image_size))
	val_ds = IDDAWDataset(f"{splits_dir}/val.csv", mode="val", label_mapper=lbl, augment=False, fusion=fusion, image_size=(image_size, image_size))
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader, lbl


def evaluate(model: nn.Module, loader: DataLoader, lbl: LabelMapper, device: torch.device, variant: str):
	model.eval()
	conf = torch.zeros((len(lbl.id_to_class), len(lbl.id_to_class)), dtype=torch.long, device=device)
	with torch.no_grad():
		for batch in loader:
			if variant.startswith("mid"):
				x_rgb = batch["image_rgb"].to(device)
				x_nir = batch["image_nir"].to(device)
				logits = model(x_rgb, x_nir)
			else:
				x = batch["image"].to(device)
				logits = model(x)
			pred = torch.argmax(logits, dim=1)
			target = batch["mask"].to(device)
			conf += fast_hist(pred, target, num_classes=len(lbl.id_to_class), ignore_index=lbl.ignore_index)
	iou, mean_iou = compute_miou(conf.float())
	safety = compute_safe_miou(conf.float(), lbl.id_to_class, SafetyGroups.default().group_to_classes)
	return float(mean_iou), {i: float(iou[i]) for i in range(len(iou))}, safety


def train(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, val_loader, lbl = create_dataloaders(args.splits_dir, args.image_size, args.batch_size, args.workers, args.modality)
	print(f"Device: {device}", flush=True)
	print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}", flush=True)
	# choose backbone
	variant = f"{args.modality}_{args.backbone}" if args.modality != "mid" else "mid_mbv3"
	model = build_model(variant, num_classes=len(lbl.id_to_class)).to(device)
	criterion = nn.CrossEntropyLoss(ignore_index=lbl.ignore_index)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
	scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

	# Resume from checkpoint if provided
	start_epoch = 0
	best_miou = float("-inf")
	if args.resume:
		checkpoint_path = f"{args.out_dir}/last_{variant}.pt"
		if Path(checkpoint_path).exists():
			print(f"Resuming from {checkpoint_path}", flush=True)
			checkpoint = torch.load(checkpoint_path, map_location=device)
			model.load_state_dict(checkpoint["model"])
			
			# Only load optimizer/scaler if they exist and are not empty
			if "optimizer" in checkpoint and checkpoint["optimizer"]:
				optimizer.load_state_dict(checkpoint["optimizer"])
			if "scaler" in checkpoint and checkpoint["scaler"]:
				scaler.load_state_dict(checkpoint["scaler"])
			
			start_epoch = checkpoint.get("epoch", 0) + 1
			best_miou = checkpoint.get("miou", float("-inf"))
			print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.3f}", flush=True)
		else:
			print(f"No checkpoint found at {checkpoint_path}, starting from scratch", flush=True)

	for epoch in range(start_epoch, args.epochs):
		model.train()
		try:
			total_batches = len(train_loader)
		except Exception:
			total_batches = None
		print(f"Starting epoch {epoch+1}/{args.epochs}" + (f" with {total_batches} batches" if total_batches is not None else ""), flush=True)
		for batch in train_loader:
			optimizer.zero_grad(set_to_none=True)
			if args.modality == "mid":
				x_rgb = batch["image_rgb"].to(device)
				x_nir = batch["image_nir"].to(device)
				target = batch["mask"].to(device)
				with torch.cuda.amp.autocast(enabled=args.amp):
					logits = model(x_rgb, x_nir)
					loss = criterion(logits, target)
			else:
				x = batch["image"].to(device)
				target = batch["mask"].to(device)
				with torch.cuda.amp.autocast(enabled=args.amp):
					logits = model(x)
					loss = criterion(logits, target)
			if scaler.is_enabled():
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				loss.backward()
				optimizer.step()
			# lightweight progress print
			if total_batches is not None:
				# print every ~5% of epoch
				# fallback: modulo by max(1, total_batches//20)
				mod = max(1, total_batches // 20)
				if (getattr(train, "_seen", 0) + 1) % mod == 0:
					print(f"  batch {getattr(train, '_seen', 0)+1}/{total_batches}, loss={float(loss.detach().cpu()):.4f}", flush=True)
				setattr(train, "_seen", getattr(train, "_seen", 0) + 1)
		
		miou, _, safety = evaluate(model, val_loader, lbl, device, variant)
		Path(args.out_dir).mkdir(parents=True, exist_ok=True)
		# always save last (with full state for resuming)
		_t0 = time.perf_counter()
		torch.save({
			"model": model.state_dict(), 
			"optimizer": optimizer.state_dict(),
			"scaler": scaler.state_dict(),
			"epoch": epoch,
			"miou": miou, 
			"safety": safety
		}, f"{args.out_dir}/last_{variant}.pt")
		last_dt = time.perf_counter() - _t0
		print(f"Saved last checkpoint in {last_dt*1000:.1f} ms", flush=True)
		# save best (inclusive)
		if miou >= best_miou:
			best_miou = miou
			_t1 = time.perf_counter()
			torch.save({"model": model.state_dict(), "miou": miou, "safety": safety}, f"{args.out_dir}/best_{variant}.pt")
			best_dt = time.perf_counter() - _t1
			print(f"Saved best checkpoint in {best_dt*1000:.1f} ms", flush=True)
		print(f"epoch {epoch}: mIoU={miou:.3f}, safe_mIoU={safety.get('safe_mIoU',0):.3f}", flush=True)
	print(f"Checkpoints saved under {args.out_dir} (best/last for {variant}).", flush=True)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--splits_dir", type=str, default="project/splits")
	parser.add_argument("--image_size", type=int, default=512)
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--out_dir", type=str, default="project/ckpts")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
	parser.add_argument("--modality", type=str, default="rgb", choices=["rgb", "nir", "early4", "mid"])
	parser.add_argument("--backbone", type=str, default="mbv3", choices=["mbv3", "fastscnn"])
	args = parser.parse_args()
	train(args)


if __name__ == "__main__":
	main()



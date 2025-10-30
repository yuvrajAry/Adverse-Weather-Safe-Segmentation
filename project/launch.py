from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run(cmd: list[str]):
	print("$", " ".join(cmd))
	return subprocess.call(cmd)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="project/configs/iddaw.yaml")
	args = parser.parse_args()

	with open(args.config, "r") as f:
		cfg = yaml.safe_load(f)

	# Preprocess: generate CSV splits
	data_root = cfg.get("data_root", "IDDAW")
	splits_dir = cfg.get("splits_dir", "project/splits")
	split = cfg.get("split", {})
	rc = run([
		sys.executable, "-m", "project.preprocess",
		"--data_root", data_root,
		"--out_dir", splits_dir,
		"--train_ratio", str(split.get("train", 0.8)),
		"--val_ratio", str(split.get("val", 0.1)),
		"--test_ratio", str(split.get("test", 0.1)),
		"--seed", str(split.get("seed", 42)),
	])
	if rc != 0:
		sys.exit(rc)

	# Train all variants
	train_cfg = cfg.get("train", {})
	out_dir = cfg.get("output_dir", "project/ckpts")
	variants = cfg.get("variants", [])
	for v in variants:
		modality = v.get("modality", "rgb")
		backbone = v.get("backbone", "mbv3")
		rc = run([
			sys.executable, "-m", "project.train",
			"--splits_dir", splits_dir,
			"--image_size", str(train_cfg.get("image_size", 512)),
			"--batch_size", str(train_cfg.get("batch_size", 4)),
			"--workers", str(train_cfg.get("workers", 4)),
			"--epochs", str(train_cfg.get("epochs", 10)),
			"--lr", str(train_cfg.get("lr", 1e-3)),
			"--out_dir", out_dir,
			"--modality", modality,
			"--backbone", backbone,
			"--amp" if bool(train_cfg.get("amp", True)) else "",
		])
		if rc != 0:
			print(f"Variant {modality}_{backbone} failed with code {rc}")


if __name__ == "__main__":
	main()



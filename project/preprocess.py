import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter


def find_triples(root: Path):
	entries = []
	# Expect structure: IDDAW/{split}/{WEATHER}/{rgb|nir|gtSeg}/{sequence}/{frame_*}
	for split in ["train", "val"]:
		base_split = root / split
		if not base_split.exists():
			continue
		for weather_dir in sorted([p for p in base_split.iterdir() if p.is_dir()]):
			weather = weather_dir.name
			for seq_dir in sorted((weather_dir / "rgb").glob("*")):
				if not seq_dir.is_dir():
					continue
				seq = seq_dir.name
				rgb_seq = weather_dir / "rgb" / seq
				nir_seq = weather_dir / "nir" / seq
				gt_seq = weather_dir / "gtSeg" / seq
				if not (rgb_seq.exists() and nir_seq.exists() and gt_seq.exists()):
					continue
				for rgb_path in sorted(rgb_seq.glob("*_rgb.png")):
					stem = rgb_path.name.replace("_rgb.png", "")
					nir_path = nir_seq / f"{stem}_nir.png"
					gt_path = gt_seq / f"{stem}_mask.json"
					if nir_path.exists() and gt_path.exists():
						entries.append({
							"orig_split": split,
							"weather": weather,
							"sequence": seq,
							"frame": stem,
							"rgb_path": str(rgb_path.as_posix()),
							"nir_path": str(nir_path.as_posix()),
							"mask_path": str(gt_path.as_posix()),
						})
	return entries


def stratified_split(entries, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
	assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
	random.Random(seed).shuffle(entries)
	# Stratify by weather and sequence to avoid leakage across splits
	by_group = defaultdict(list)
	for e in entries:
		key = (e["weather"], e["sequence"])  # keep frames from the same sequence together
		by_group[key].append(e)
	groups = list(by_group.keys())
	random.Random(seed).shuffle(groups)

	# Count total frames for proportional allocation while keeping groups intact
	group_sizes = {g: len(by_group[g]) for g in groups}
	total = sum(group_sizes.values())
	alloc = {"train": int(total * train_ratio), "val": int(total * val_ratio)}
	alloc["test"] = total - alloc["train"] - alloc["val"]

	result = {"train": [], "val": [], "test": []}
	remaining = alloc.copy()
	for g in groups:
		size = group_sizes[g]
		# Greedy: assign the group to the split with the highest remaining need
		target = max(remaining, key=lambda k: remaining[k])
		result[target].extend(by_group[g])
		remaining[target] -= size

	# Report actual distribution
	return result


def summarize(splits):
	stats = {}
	for split, items in splits.items():
		wc = Counter([e["weather"] for e in items])
		stats[split] = {"count": len(items), "weather": dict(wc)}
	return stats


def write_csv(splits, out_dir: Path):
	out_dir.mkdir(parents=True, exist_ok=True)
	for split, items in splits.items():
		csv_path = out_dir / f"{split}.csv"
		with csv_path.open("w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=[
				"split", "orig_split", "weather", "sequence", "frame", "rgb_path", "nir_path", "mask_path"
			])
			writer.writeheader()
			for e in items:
				row = {"split": split, **e}
				writer.writerow(row)


def main():
	parser = argparse.ArgumentParser(description="IDD-AW preprocessing: stratified splits and triple verification")
	parser.add_argument("--data_root", type=str, default="IDDAW", help="Path to IDD-AW root")
	parser.add_argument("--out_dir", type=str, default="project/splits", help="Where to write CSV splits")
	parser.add_argument("--train_ratio", type=float, default=0.8)
	parser.add_argument("--val_ratio", type=float, default=0.1)
	parser.add_argument("--test_ratio", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	root = Path(args.data_root)
	entries = find_triples(root)
	if not entries:
		raise SystemExit("No entries found. Check --data_root path and expected structure.")

	splits = stratified_split(entries, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
	stats = summarize(splits)
	write_csv(splits, Path(args.out_dir))

	print("Split statistics:")
	for split, info in stats.items():
		print(f"  {split}: {info['count']} frames, weather={info['weather']}")


if __name__ == "__main__":
	main()



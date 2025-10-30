from __future__ import annotations

from typing import Dict, Tuple, List, Set

import torch
import torch.nn.functional as F


def fast_hist(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
	mask = target != ignore_index
	p = pred[mask].view(-1)
	t = target[mask].view(-1)
	K = num_classes
	# bincount trick
	idx = t * K + p
	h = torch.bincount(idx, minlength=K * K)
	return h.view(K, K)


def compute_miou(confmat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	# IoU per class and mean
	TP = torch.diag(confmat)
	FP = confmat.sum(0) - TP
	FN = confmat.sum(1) - TP
	den = TP + FP + FN
	iou = torch.where(den > 0, TP / torch.clamp(den, min=1), torch.zeros_like(TP, dtype=torch.float))
	m = iou[den > 0].mean() if (den > 0).any() else torch.tensor(0.0)
	return iou, m


def compute_safe_miou(confmat: torch.Tensor, id_to_class: Dict[int, str], safety_groups: Dict[str, Set[str]]) -> Dict[str, float]:
	# Aggregate confusion for safety groups and compute group IoU
	class_to_id = {v: k for k, v in id_to_class.items()}
	results: Dict[str, float] = {}
	for group, names in safety_groups.items():
		ids = [class_to_id[c] for c in names if c in class_to_id]
		if not ids:
			results[group] = 0.0
			continue
		TP = confmat[ids][:, ids].trace()
		FP = confmat[:, ids].sum() - TP
		FN = confmat[ids, :].sum() - TP
		den = TP + FP + FN
		val = float(TP / den) if den > 0 else 0.0
		results[group] = val
	# Mean over defined safety groups
	if results:
		results["safe_mIoU"] = sum(results.values()) / len(results)
	else:
		results["safe_mIoU"] = 0.0
	return results


def entropy_map(logits: torch.Tensor) -> torch.Tensor:
	# logits: BxCxHxW -> per-pixel entropy
	prob = F.softmax(logits, dim=1)
	logp = torch.log(torch.clamp(prob, min=1e-8))
	ent = -(prob * logp).sum(dim=1)
	return ent



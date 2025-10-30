from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class LabelMapper:
	class_to_id: Dict[str, int]
	id_to_class: Dict[int, str]
	ignore_index: int = 255

	@staticmethod
	def from_classes(classes: List[str], ignore_index: int = 255) -> "LabelMapper":
		class_to_id = {c: i for i, c in enumerate(classes)}
		id_to_class = {i: c for c, i in class_to_id.items()}
		return LabelMapper(class_to_id, id_to_class, ignore_index)

	@staticmethod
	def default() -> "LabelMapper":
		# Minimal robust set; extend as needed when full taxonomy is known.
		classes = [
			"road", "sidewalk", "building", "sky", "vegetation", "terrain",
			"person", "rider", "car", "truck", "bus", "motorcycle", "bicycle",
			"animal", "traffic sign", "traffic light", "fallback background",
			"drivable fallback", "obs-str-bar-fallback",
		]
		return LabelMapper.from_classes(classes)


@dataclass
class SafetyGroups:
	# Define safety-critical super-classes for Safe-mIoU
	group_to_classes: Dict[str, Set[str]]

	@staticmethod
	def default() -> "SafetyGroups":
		return SafetyGroups(
			group_to_classes={
				"pedestrian": {"person", "rider"},
				"vehicle": {"car", "truck", "bus", "motorcycle", "bicycle"},
				"cyclist": {"bicycle", "rider"},
				"animal": {"animal"},
			}
		)



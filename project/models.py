from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


def _make_classifier(num_classes: int) -> nn.Sequential:
	return nn.Sequential(
		nn.Conv2d(128, 128, 3, padding=1, bias=False),
		nn.BatchNorm2d(128),
		nn.ReLU(inplace=True),
		nn.Conv2d(128, num_classes, 1),
	)


class FastSCNN(nn.Module):
	def __init__(self, in_channels: int, num_classes: int):
		super().__init__()
		
		self.stem = nn.Sequential(
			nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(32), nn.ReLU(inplace=True),
			nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(48), nn.ReLU(inplace=True),
			nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(64), nn.ReLU(inplace=True),
		)
		self.context = nn.Sequential(
			nn.Conv2d(64, 96, 3, padding=2, dilation=2, bias=False),
			nn.BatchNorm2d(96), nn.ReLU(inplace=True),
			nn.Conv2d(96, 128, 3, padding=4, dilation=4, bias=False),
			nn.BatchNorm2d(128), nn.ReLU(inplace=True),
		)
		self.classifier = _make_classifier(num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		inp_size = x.shape[-2:]
		x = self.stem(x)
		x = self.context(x)
		x = self.classifier(x)
		x = F.interpolate(x, size=inp_size, mode="bilinear", align_corners=False)
		return x


class MobileNetV3LiteSeg(nn.Module):
	def __init__(self, in_channels: int, num_classes: int):
		super().__init__()
		self.encoder = mobilenet_v3_small(weights=None)
		# adjust first conv
		first = self.encoder.features[0][0]
		self.encoder.features[0][0] = nn.Conv2d(in_channels, first.out_channels, kernel_size=first.kernel_size,
				stride=first.stride, padding=first.padding, bias=False)
		enc_out = 576  # mobilenet_v3_small final channels
		self.decoder = nn.Sequential(
			nn.Conv2d(enc_out, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
			nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(inplace=True),
			nn.ConvTranspose2d(64, 64, 2, stride=2), nn.ReLU(inplace=True),
			nn.Conv2d(64, num_classes, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		inp_size = x.shape[-2:]
		f = self.encoder.features(x)
		logits = self.decoder(f)
		logits = F.interpolate(logits, size=inp_size, mode="bilinear", align_corners=False)
		return logits


class MidFusionDualEncoder(nn.Module):
	def __init__(self, num_classes: int):
		super().__init__()
		# two encoders share architecture but not weights
		self.enc_rgb = mobilenet_v3_small(weights=None)
		self.enc_nir = mobilenet_v3_small(weights=None)
		# adjust first conv: RGB=3, NIR=1
		first_rgb = self.enc_rgb.features[0][0]
		self.enc_rgb.features[0][0] = nn.Conv2d(3, first_rgb.out_channels, kernel_size=first_rgb.kernel_size,
				stride=first_rgb.stride, padding=first_rgb.padding, bias=False)
		first_nir = self.enc_nir.features[0][0]
		self.enc_nir.features[0][0] = nn.Conv2d(1, first_nir.out_channels, kernel_size=first_nir.kernel_size,
				stride=first_nir.stride, padding=first_nir.padding, bias=False)
		enc_out = 576
		self.attn = nn.Sequential(
			nn.Conv2d(enc_out * 2, enc_out, 1, bias=False), nn.BatchNorm2d(enc_out), nn.ReLU(inplace=True),
			nn.Conv2d(enc_out, enc_out * 2, 1), nn.Sigmoid(),
		)
		self.fuse_conv = nn.Conv2d(enc_out * 2, 128, 1)
		self.cls = _make_classifier(num_classes)

	def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
		inp_size = rgb.shape[-2:]
		fr = self.enc_rgb.features(rgb)
		fn = self.enc_nir.features(nir)
		x = torch.cat([fr, fn], dim=1)
		w = self.attn(x)
		x = x * w
		x = self.fuse_conv(x)
		x = self.cls(x)
		x = F.interpolate(x, size=inp_size, mode="bilinear", align_corners=False)
		return x


def build_model(variant: str, num_classes: int) -> nn.Module:
	"""variant: rgb|nir|early4|mid, backbones: fastscnn|mbv3
	Use 'variant_backbone' format, e.g., 'rgb_fastscnn', 'early4_mbv3', 'mid_mbv3'.
	"""
	parts = variant.split("_")
	if len(parts) == 1:
		modality, backbone = parts[0], "mbv3"
	else:
		modality, backbone = parts
	if modality == "mid":
		assert backbone in {"mbv3"}
		return MidFusionDualEncoder(num_classes=num_classes)
	if backbone == "fastscnn":
		in_ch = 1 if modality == "nir" else (4 if modality == "early4" else 3)
		return FastSCNN(in_channels=in_ch, num_classes=num_classes)
	else:
		in_ch = 1 if modality == "nir" else (4 if modality == "early4" else 3)
		return MobileNetV3LiteSeg(in_channels=in_ch, num_classes=num_classes)



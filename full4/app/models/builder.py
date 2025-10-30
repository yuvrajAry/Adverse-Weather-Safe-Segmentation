import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        return self.conv(x)


class FastSCNNStub(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = SimpleHead(64, num_classes)

    def forward_logits(self, x):
        feats = self.encoder(x)
        logits = self.head(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward_logits(x)
            probs = F.softmax(logits, dim=1)
            mask = probs.argmax(dim=1)[0].cpu().numpy()
            return mask, probs[0].cpu().numpy()


class MobileNetV3LiteStub(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.head = SimpleHead(96, num_classes)

    def forward_logits(self, x):
        feats = self.encoder(x)
        logits = self.head(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward_logits(x)
            probs = F.softmax(logits, dim=1)
            mask = probs.argmax(dim=1)[0].cpu().numpy()
            return mask, probs[0].cpu().numpy()


class MidLevelFusionWrapper(nn.Module):
    def __init__(self, base_encoder_ch: int, num_classes: int):
        super().__init__()
        # simple dual-encoder attention-like fusion
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(3, base_encoder_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_encoder_ch),
            nn.ReLU(inplace=True),
        )
        self.nir_enc = nn.Sequential(
            nn.Conv2d(1, base_encoder_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_encoder_ch),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(base_encoder_ch * 2, base_encoder_ch, 1),
            nn.BatchNorm2d(base_encoder_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        self.head = SimpleHead(base_encoder_ch, num_classes)

    def forward_logits(self, x):
        # x is [B,4,H,W]: first 3 rgb, last 1 nir
        rgb = x[:, :3]
        nir = x[:, 3:4]
        fr = self.rgb_enc(rgb)
        fn = self.nir_enc(nir)
        fused = torch.cat([fr, fn], dim=1)
        fused = self.fusion(fused)
        logits = self.head(fused)
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward_logits(x)
            probs = F.softmax(logits, dim=1)
            mask = probs.argmax(dim=1)[0].cpu().numpy()
            return mask, probs[0].cpu().numpy()


def build_model(model_name: str, fusion_mode: str, num_classes: int, weights_path: Optional[str] = None):
    in_ch = 4  # early fusion and mid-level wrapper both take 4-ch input tensor
    model_name = (model_name or 'fast_scnn').lower()
    fusion_mode = (fusion_mode or 'early').lower()

    if fusion_mode == 'mid':
        net = MidLevelFusionWrapper(base_encoder_ch=64, num_classes=num_classes)
    else:
        if model_name == 'mobilenetv3_lite':
            net = MobileNetV3LiteStub(in_ch=in_ch, num_classes=num_classes)
        else:
            net = FastSCNNStub(in_ch=in_ch, num_classes=num_classes)

    if weights_path:
        try:
            state = torch.load(weights_path, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                net.load_state_dict(state['state_dict'], strict=False)
            else:
                net.load_state_dict(state, strict=False)
        except Exception:
            # continue with random weights
            pass
    return net


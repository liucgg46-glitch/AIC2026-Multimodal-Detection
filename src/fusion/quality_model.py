"""One RGB+IR+Depth detector with a frozen-RGB-capable quality-aware residual path."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.tasks import DetectionModel


TRANSPORT_CHANNELS = 11
RGB_SLICE = slice(0, 3)
IR_SLICE = slice(3, 6)
DEPTH_SLICE = slice(6, 9)  # linear, inverse, validity
IR_VALID_CHANNEL = 9
DEPTH_FORMAT_CHANNEL = 10  # zero for physical uint16 PNG, one for encoded JPG


class SmallPyramid(nn.Module):
    """Low-memory stride-8/16/32 auxiliary feature extractor."""

    def __init__(self, in_channels: int, stem_channels: int = 16):
        super().__init__()
        widths = (stem_channels, 24, 48, 64, 96)
        self.blocks = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(previous, width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(inplace=True),
            ))
            previous = width

    def forward_from_stem(self, stem):
        features = {}
        x = stem
        for index, block in enumerate(self.blocks[1:], start=1):
            x = block(x)
            if index in (2, 3, 4):
                features[(8, 16, 32)[index - 2]] = x
        return features

    def forward(self, x):
        return self.forward_from_stem(self.blocks[0](x))


class FormatAwareDepthPyramid(SmallPyramid):
    """Separate first adapters for physically scaled PNG and unknown-scale JPG."""

    def __init__(self):
        super().__init__(3)
        self.jpg_stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, jpg_flag):
        png = self.blocks[0](x)
        jpg = self.jpg_stem(x)
        flag = (jpg_flag.mean((2, 3), keepdim=True) > 0.5).to(x.dtype)
        return self.forward_from_stem(png * (1.0 - flag) + jpg * flag)


class QualityResidual(nn.Module):
    """Zero-initialized residual, attenuated by the valid region and coverage."""

    def __init__(self, rgb_channels: int, aux_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(aux_channels, rgb_channels, 1)
        self.gate = nn.Conv2d(rgb_channels + aux_channels + 2, 1, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, rgb, aux, valid):
        if rgb.shape[2:] != aux.shape[2:]:
            raise ValueError("QualityResidual feature shapes differ")
        mask = F.interpolate(valid, size=rgb.shape[2:], mode="nearest")
        mask = (mask > 0.5).to(rgb.dtype)
        coverage = mask.mean((2, 3), keepdim=True)
        context = torch.cat((rgb, aux, mask, coverage.expand_as(mask)), dim=1)
        return rgb + mask * coverage * torch.sigmoid(self.gate(context)) * self.proj(aux)


class QualityTriModalModel(DetectionModel):
    """YOLO11m RGB path with trainable small IR/Depth pyramids at P3/P4/P5."""

    def __init__(self, nc=12, verbose=False):
        # Parent constructor calls predict() while discovering native strides.
        super().__init__("yolo11m.yaml", ch=3, nc=nc, verbose=verbose)
        self.assert_graph()
        self.ir_pyramid = SmallPyramid(3)
        self.depth_pyramid = FormatAwareDepthPyramid()
        self.ir_fusion = nn.ModuleDict({
            str(stride): QualityResidual(512, channels)
            for stride, channels in ((8, 48), (16, 64), (32, 96))
        })
        self.depth_fusion = nn.ModuleDict({
            str(stride): QualityResidual(512, channels)
            for stride, channels in ((8, 48), (16, 64), (32, 96))
        })
        self.yaml["channels"] = TRANSPORT_CHANNELS
        self.yaml["fusion"] = "rgb_ir_depth_quality_p345_v1"
        self.freeze_rgb_statistics = True

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "freeze_rgb_statistics", False):
            for module in self.model:
                module.eval()  # freeze=24 alone does not freeze RGB BatchNorm running statistics
        return self

    def assert_graph(self):
        if len(self.model) != 24 or self.stride.tolist() != [8.0, 16.0, 32.0]:
            raise RuntimeError("Unsupported YOLO11m detection graph")
        expected_links = {12: [-1, 6], 15: [-1, 4], 18: [-1, 13], 21: [-1, 10], 23: [16, 19, 22]}
        for index, module in enumerate(self.model):
            if module.i != index or module.f != expected_links.get(index, -1):
                raise RuntimeError("Unsupported YOLO11m graph edge at %d" % index)

    def fused_features(self, x):
        if x.ndim != 4 or x.shape[1] != TRANSPORT_CHANNELS:
            raise ValueError("Expected normalized [B,11,H,W] RGB/IR/Depth transport")
        rgb, ir = x[:, RGB_SLICE], x[:, IR_SLICE]
        depth = x[:, DEPTH_SLICE]
        ir_valid = x[:, IR_VALID_CHANNEL:IR_VALID_CHANNEL + 1]
        depth_valid = depth[:, 2:3]
        ir_features = self.ir_pyramid(ir)
        depth_features = self.depth_pyramid(depth, x[:, DEPTH_FORMAT_CHANNEL:DEPTH_FORMAT_CHANNEL + 1])
        features = {}
        for index in range(11):
            rgb = self.model[index](rgb)
            if index in (4, 6, 10):
                stride = {4: 8, 6: 16, 10: 32}[index]
                rgb = self.ir_fusion[str(stride)](rgb, ir_features[stride], ir_valid)
                rgb = self.depth_fusion[str(stride)](rgb, depth_features[stride], depth_valid)
                features[index] = rgb
        return features

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if not hasattr(self, "ir_pyramid"):
            return super().predict(x)  # parent stride discovery
        if profile or visualize or augment or embed:
            raise ValueError("Fusion TTA/profiling/embeddings are not supported")
        features = self.fused_features(x)
        saved = [features.get(index) for index in range(11)]
        x = features[10]
        for module in self.model[11:]:
            if module.f != -1:
                x = saved[module.f] if isinstance(module.f, int) else [
                    x if source == -1 else saved[source] for source in module.f
                ]
            x = module(x)
            saved.append(x if module.i in self.save else None)
        return x

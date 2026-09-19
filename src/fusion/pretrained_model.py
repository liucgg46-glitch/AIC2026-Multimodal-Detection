"""Fusion model backed by the trained single-modality YOLO11m encoders."""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel

from .pretrained_dataset import TRANSPORT_CHANNELS
from .quality_model import QualityResidual


class YOLO11mBackbone(nn.Module):
    """Layers 0..10 of YOLO11m, exposing P4 and P5 semantic features."""

    def __init__(self, nc=12):
        super().__init__()
        graph = DetectionModel("yolo11m.yaml", ch=3, nc=nc, verbose=False)
        self.blocks = nn.ModuleList(list(graph.model[:11]))

    def forward(self, x):
        features = {}
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in (6, 10):
                features[{6: 16, 10: 32}[index]] = x
        return features


class SpatialQualityResidual(QualityResidual):
    """Zero-start residual whose 3x3 projection can correct a one-cell offset."""

    def __init__(self, channels=512):
        super().__init__(channels, channels)
        self.proj = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)


class PretrainedTriModalModel(DetectionModel):
    def __init__(self, nc=12, use_depth=True, verbose=False):
        super().__init__("yolo11m.yaml", ch=3, nc=nc, verbose=verbose)
        self.assert_graph()
        self.ir_backbone = YOLO11mBackbone(nc)
        self.depth_backbone = YOLO11mBackbone(nc) if use_depth else None
        self.ir_fusion = nn.ModuleDict({str(s): SpatialQualityResidual() for s in (16, 32)})
        self.depth_fusion = nn.ModuleDict(
            {str(s): SpatialQualityResidual() for s in (16, 32)} if use_depth else {}
        )
        self.use_depth = bool(use_depth)
        self.freeze_pretrained_statistics = True
        self.yaml["channels"] = TRANSPORT_CHANNELS
        self.yaml["fusion"] = "pretrained_rgb_ir_depth_p45_v1"
        for backbone in filter(None, (self.ir_backbone, self.depth_backbone)):
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)

    def assert_graph(self):
        if len(self.model) != 24 or self.stride.tolist() != [8.0, 16.0, 32.0]:
            raise RuntimeError("Unsupported YOLO11m detection graph")
        expected = {12: [-1, 6], 15: [-1, 4], 18: [-1, 13], 21: [-1, 10], 23: [16, 19, 22]}
        for index, module in enumerate(self.model):
            if module.i != index or module.f != expected.get(index, -1):
                raise RuntimeError("Unsupported YOLO11m graph edge at %d" % index)

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "freeze_pretrained_statistics", False):
            for module in self.model:
                module.eval()
            self.ir_backbone.eval()
            if self.depth_backbone is not None:
                self.depth_backbone.eval()
        return self

    def fused_features(self, x):
        if x.ndim != 4 or x.shape[1] != TRANSPORT_CHANNELS:
            raise ValueError("Expected normalized [B,11,H,W] pretrained transport")
        rgb, ir, depth = x[:, :3], x[:, 3:6], x[:, 6:9]
        ir_valid, depth_valid = x[:, 9:10], x[:, 10:11]
        with torch.no_grad():
            ir_features = self.ir_backbone(ir)
            depth_features = self.depth_backbone(depth) if self.use_depth else None
        features = {}
        for index in range(11):
            rgb = self.model[index](rgb)
            if index in (6, 10):
                stride = {6: 16, 10: 32}[index]
                rgb = self.ir_fusion[str(stride)](rgb, ir_features[stride], ir_valid)
                if self.use_depth:
                    rgb = self.depth_fusion[str(stride)](
                        rgb, depth_features[stride], depth_valid
                    )
                features[index] = rgb
            elif index == 4:
                features[index] = rgb
        return features

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if not hasattr(self, "ir_backbone"):
            return super().predict(x)
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

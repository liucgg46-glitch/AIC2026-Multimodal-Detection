"""Independent YOLO11n encoders and the original RGB neck/detection head."""
from copy import deepcopy

from ultralytics.nn.tasks import DetectionModel
from . import require_ultralytics
from .fusion_modules import GatedResidualFusion


class DualStreamModel(DetectionModel):
    def __init__(self, nc=12, zero_init=True, verbose=False):
        require_ultralytics()
        # Parent stride discovery calls our override before auxiliary modules exist.
        super().__init__("yolo11n.yaml", ch=3, nc=nc, verbose=verbose)
        self.assert_graph()
        self.ir_encoder = deepcopy(self.model[:11])
        self.fusion4 = GatedResidualFusion(128, zero_init)
        self.fusion5 = GatedResidualFusion(256, zero_init)
        self.yaml["channels"] = 6  # transport/warmup metadata, never a six-channel convolution
        self.yaml["fusion"] = "rgb_ir_gated_p45"

    def backbone_features(self, x):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError("Fusion input must be [B,6,H,W]")
        rgb, ir = x[:, :3], x[:, 3:6]
        rgb_features, ir_features = {}, {}
        for i in range(11):
            rgb = self.model[i](rgb)
            ir = self.ir_encoder[i](ir)
            if i in (4, 6, 10):
                rgb_features[i], ir_features[i] = rgb, ir
        return rgb_features, ir_features

    def fused_features(self, x):
        rgb, ir = self.backbone_features(x)
        return {4: rgb[4], 6: self.fusion4(rgb[6], ir[6]), 10: self.fusion5(rgb[10], ir[10])}

    def assert_graph(self):
        expected = ["Conv", "Conv", "C3k2", "Conv", "C3k2", "Conv", "C3k2",
                    "Conv", "C3k2", "SPPF", "C2PSA", "Upsample", "Concat", "C3k2",
                    "Upsample", "Concat", "C3k2", "Conv", "Concat", "C3k2",
                    "Conv", "Concat", "C3k2", "Detect"]
        if [type(m).__name__ for m in self.model] != expected:
            raise RuntimeError("Unsupported YOLO11 graph/module types")
        links = {12: [-1, 6], 15: [-1, 4], 18: [-1, 13], 21: [-1, 10], 23: [16, 19, 22]}
        for i, m in enumerate(self.model):
            if m.i != i or m.f != links.get(i, -1):
                raise RuntimeError("Unsupported YOLO11 graph edge at %s" % i)
        if self.stride.tolist() != [8.0, 16.0, 32.0]:
            raise RuntimeError("Unsupported detection stride semantics")

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """Public prediction extension; keep BaseModel dict->loss dispatch unchanged."""
        if not hasattr(self, "ir_encoder"):
            return super().predict(x)  # native initialization/stride discovery
        if profile or visualize or augment or embed:
            raise ValueError("Fusion profiling/visualization/TTA/embeddings are not supported")
        features = self.fused_features(x)
        saved = [features.get(i) for i in range(11)]
        x = features[10]
        for m in self.model[11:]:
            if m.f != -1:
                x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x)
            saved.append(x if m.i in self.save else None)
        return x

    def load(self, weights, verbose=True):
        source = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
        if isinstance(source, DualStreamModel):
            self.load_state_dict(source.float().state_dict(), strict=True)
        else:
            from .initialization import load_matching_state
            self.load_report = load_matching_state(self, source.float().state_dict(), verbose=verbose)
        return self

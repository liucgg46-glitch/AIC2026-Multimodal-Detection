from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pytest
from ultralytics.cfg import get_cfg
from src.fusion.paired_dataset import DISABLED


@pytest.fixture
def hyp():
    torch.set_num_threads(2)
    disabled = {k: 0.0 for k in DISABLED}
    disabled["multi_scale"] = False  # bool in 8.3.253, float-or-bool in 8.4.144
    return get_cfg(overrides={**disabled, "hsv_h": 0.0, "hsv_s": 0.0,
                              "hsv_v": 0.0, "flipud": 0.5, "fliplr": 0.5})

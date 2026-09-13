"""Shared fixed-shape YOLOX loader used by ONNX and Core ML exporters."""

from pathlib import Path

import torch
from torch import nn
from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


class YOLOXRuntimeModel(nn.Module):
    def __init__(self, model: nn.Module, test_size: tuple[int, int]) -> None:
        super().__init__()
        model.head.decode_in_inference = False
        self.model = model

        grids = []
        strides = []
        for stride in model.head.strides:
            height, width = (test_size[0] // stride, test_size[1] // stride)
            y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
            grid = torch.stack((x, y), dim=-1).reshape(1, -1, 2).float()
            grids.append(grid)
            strides.append(torch.full((1, height * width, 1), float(stride)))
        self.register_buffer("grid", torch.cat(grids, dim=1))
        self.register_buffer("strides", torch.cat(strides, dim=1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raw = self.model(image)
        centers = (raw[..., :2] + self.grid) * self.strides
        sizes = torch.exp(raw[..., 2:4]) * self.strides
        return torch.cat((centers, sizes, raw[..., 4:]), dim=-1)


def load_yolox_runtime(experiment: Path, checkpoint: Path):
    exp = get_exp(str(experiment), None)
    model = replace_module(exp.get_model(), nn.SiLU, SiLU).eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("ema", saved["model"]))
    return exp, YOLOXRuntimeModel(model, exp.test_size).eval()

#!/usr/bin/env python3
"""Convert the official Real-ESRGAN Anime 6B weights to Core ML."""

import argparse
import hashlib
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WEIGHTS_SHA256 = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"


class ResidualDenseBlock(nn.Module):
    """BasicSR RRDB block, reproduced under Apache-2.0."""

    def __init__(self, features: int = 64, growth: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + growth * 2, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + growth * 3, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + growth * 4, features, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.activation(self.conv1(value))
        second = self.activation(self.conv2(torch.cat((value, first), 1)))
        third = self.activation(self.conv3(torch.cat((value, first, second), 1)))
        fourth = self.activation(self.conv4(torch.cat((value, first, second, third), 1)))
        fifth = self.conv5(torch.cat((value, first, second, third, fourth), 1))
        return value + fifth * 0.2


class RRDB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock()
        self.rdb2 = ResidualDenseBlock()
        self.rdb3 = ResidualDenseBlock()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.rdb3(self.rdb2(self.rdb1(value)))
        return value + output * 0.2


class RealESRGANAnime6B(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB() for _ in range(6)))
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.conv_first(value)
        features = features + self.conv_body(self.body(features))
        features = self.activation(self.conv_up1(F.interpolate(features, scale_factor=2, mode="nearest")))
        features = self.activation(self.conv_up2(F.interpolate(features, scale_factor=2, mode="nearest")))
        return self.conv_last(self.activation(self.conv_hr(features)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-edge", type=int, default=512)
    args = parser.parse_args()

    if sha256(args.weights) != WEIGHTS_SHA256:
        raise SystemExit("official weight SHA-256 mismatch")

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = RealESRGANAnime6B().eval()
    model.load_state_dict(checkpoint["params_ema"], strict=True)

    example = torch.zeros((1, 3, args.input_edge, args.input_edge), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    converted.author = "NicoComic contributors"
    converted.short_description = "Real-ESRGAN Anime 6B 4× super-resolution"
    converted.version = "0.1.0"
    converted.user_defined_metadata["source_weights_sha256"] = WEIGHTS_SHA256
    converted.user_defined_metadata["license"] = "BSD-3-Clause"

    if args.output.exists():
        shutil.rmtree(args.output)
    converted.save(args.output)

    torch.manual_seed(20260915)
    sample = torch.rand(example.shape, dtype=torch.float32)
    with torch.no_grad():
        reference = model(sample).numpy()
    prediction = converted.predict({"input": sample.numpy()})["output"]
    expected = (1, 3, args.input_edge * 4, args.input_edge * 4)
    if prediction.shape != expected:
        raise SystemExit(f"unexpected output shape: {prediction.shape}, expected {expected}")
    delta = np.abs(reference - prediction)
    mean_error = float(delta.mean())
    max_error = float(delta.max())
    if mean_error > 0.002 or max_error > 0.01:
        raise SystemExit(f"Core ML parity failed: mean={mean_error}, max={max_error}")
    print(f"Core ML parity passed: mean={mean_error:.6f}, max={max_error:.6f}")


if __name__ == "__main__":
    main()

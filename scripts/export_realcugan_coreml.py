#!/usr/bin/env python3
"""Convert the official Real-CUGAN 2x no-denoise weights to Core ML."""

import argparse
import hashlib
import importlib.util
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SOURCE_SHA256 = "ef6c4e433bcac37b75ffba0a4044987ddd3ecfe7765a74a3c93887954e45562b"
WEIGHTS_SHA256 = "f491f9ecf6964ead9f3a36bf03e83527f32c6a341b683f7378ac6c1e2a5f0d16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(source: Path, weights: Path) -> nn.Module:
    spec = importlib.util.spec_from_file_location("realcugan_upcunet", source)
    if spec is None or spec.loader is None:
        raise SystemExit("could not load official Real-CUGAN source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.UpCunet2x().eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model


class FixedRealCUGAN2x(nn.Module):
    """Official whole-image forward without its final uint8 conversion."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.unet1 = model.unet1
        self.unet2 = model.unet2

    def run_unet1(self, value: torch.Tensor) -> torch.Tensor:
        first = self.unet1.conv1(value)
        second = F.leaky_relu(self.unet1.conv1_down(first), 0.1)
        first = first[:, :, 4:-4, 4:-4]
        second = F.leaky_relu(self.unet1.conv2_up(self.unet1.conv2(second)), 0.1)
        return self.unet1.conv_bottom(F.leaky_relu(self.unet1.conv3(first + second), 0.1))

    def run_unet2(self, value: torch.Tensor) -> torch.Tensor:
        first = self.unet2.conv1(value)
        second = F.leaky_relu(self.unet2.conv1_down(first), 0.1)
        first = first[:, :, 16:-16, 16:-16]
        second = self.unet2.conv2(second)
        third = F.leaky_relu(self.unet2.conv2_down(second), 0.1)
        second = second[:, :, 4:-4, 4:-4]
        third = F.leaky_relu(self.unet2.conv3_up(self.unet2.conv3(third)), 0.1)
        fourth = F.leaky_relu(self.unet2.conv4_up(self.unet2.conv4(second + third)), 0.1)
        fifth = F.leaky_relu(self.unet2.conv5(first + fourth), 0.1)
        return self.unet2.conv_bottom(fifth)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.run_unet1(F.pad(value, (18, 18, 18, 18), "reflect"))
        second = self.run_unet2(first)
        return torch.clamp(first[:, :, 20:-20, 20:-20] + second, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if sha256(args.source) != SOURCE_SHA256:
        raise SystemExit("official source SHA-256 mismatch")
    if sha256(args.weights) != WEIGHTS_SHA256:
        raise SystemExit("official weight SHA-256 mismatch")

    official = load_model(args.source, args.weights)
    model = FixedRealCUGAN2x(official).eval()
    example = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    torch.manual_seed(20260915)
    sample = torch.rand(example.shape, dtype=torch.float32)
    with torch.no_grad():
        official_output = official(sample, 0, 0, 1, False).float().numpy() / 255
        rewritten_output = model(sample).numpy()
        source_delta = np.abs(official_output - rewritten_output)
        if float(source_delta.max()) > 0.002:
            raise SystemExit(f"official forward mismatch: max={float(source_delta.max())}")
        print(f"Official forward parity passed: max={float(source_delta.max()):.6f}")
        traced = torch.jit.trace(model, example)

    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
    )
    converted.author = "NicoComic contributors"
    converted.short_description = "Real-CUGAN 2× no-denoise super-resolution"
    converted.version = "0.1.0"
    converted.user_defined_metadata["source_sha256"] = SOURCE_SHA256
    converted.user_defined_metadata["source_weights_sha256"] = WEIGHTS_SHA256
    converted.user_defined_metadata["license"] = "MIT"

    if args.output.exists():
        shutil.rmtree(args.output)
    converted.save(args.output)

    reference = rewritten_output
    prediction = converted.predict({"input": sample.numpy()})["output"]
    expected = (1, 3, 512, 512)
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

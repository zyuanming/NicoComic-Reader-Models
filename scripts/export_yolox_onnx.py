#!/usr/bin/env python3
"""Export a NicoComic YOLOX checkpoint with current public PyTorch APIs."""

import argparse
from pathlib import Path

import onnx
import torch

from yolox_export_support import load_yolox_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    exp, model = load_yolox_runtime(args.experiment, args.checkpoint)

    example = torch.zeros((1, 3, *exp.test_size), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["images"],
        output_names=["output"],
        opset_version=13,
        dynamo=False,
    )

    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    dimensions = [item.dim_value for item in exported.graph.input[0].type.tensor_type.shape.dim]
    if dimensions != [1, 3, *exp.test_size]:
        raise SystemExit(f"Unexpected ONNX input shape: {dimensions}")


if __name__ == "__main__":
    main()

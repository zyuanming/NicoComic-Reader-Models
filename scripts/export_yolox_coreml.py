#!/usr/bin/env python3
"""Convert a NicoComic YOLOX checkpoint to a fixed-input Core ML package."""

import argparse
import shutil
from pathlib import Path

import coremltools as ct
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
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    if args.output.exists():
        shutil.rmtree(args.output)
    converted = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="images",
                shape=example.shape,
                scale=1.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="detections")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    converted.author = "NicoComic"
    converted.short_description = f"YOLOX-Nano {exp.test_size[0]} comic panel detector."
    converted.version = "0.1.0"
    converted.user_defined_metadata["architecture"] = "YOLOX-Nano"
    converted.user_defined_metadata["input_edge"] = str(exp.test_size[0])
    converted.user_defined_metadata["license"] = "Apache-2.0"
    converted.save(args.output)


if __name__ == "__main__":
    main()

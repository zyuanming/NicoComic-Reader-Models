#!/usr/bin/env python3
"""Patch the pinned YOLOX checkout for PyTorch MPS tensor conversion."""

import argparse
from pathlib import Path


REPLACEMENTS = {
    "yolox/models/losses.py": {
        "(tl < br).type(tl.type()).prod(dim=1)": "(tl < br).to(dtype=tl.dtype).prod(dim=1)",
    },
    "yolox/models/yolo_head.py": {
        "torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)": (
            "torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2)"
            ".to(device=output.device, dtype=output.dtype)"
        ),
        "torch.cat(grids, dim=1).type(dtype)": (
            "torch.cat(grids, dim=1).to(device=outputs.device, dtype=outputs.dtype)"
        ),
        "torch.cat(strides, dim=1).type(dtype)": (
            "torch.cat(strides, dim=1).to(device=outputs.device, dtype=outputs.dtype)"
        ),
    },
    "yolox/utils/boxes.py": {
        "(tl < br).type(tl.type()).prod(dim=2)": "(tl < br).to(dtype=tl.dtype).prod(dim=2)",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()

    for relative_path, replacements in REPLACEMENTS.items():
        path = args.checkout / relative_path
        source = path.read_text()
        for original, replacement in replacements.items():
            if original in source:
                source = source.replace(original, replacement)
            elif replacement not in source:
                raise SystemExit(f"Pinned YOLOX source mismatch: {relative_path}")
        path.write_text(source)


if __name__ == "__main__":
    main()

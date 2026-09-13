#!/usr/bin/env python3
"""Convert the fixed COMICS manual panel archive into a YOLOX COCO dataset."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image


SOURCE = "https://obj.umiacs.umd.edu/comics/panels_annotations.zip"
SOURCE_SHA256 = "9d6bcfa5d9c3a1650529db9fd5b8ea2281529ea0ada00ad7a48ce61d3687ec07"


def split_for(stem):
    return "val2017" if hashlib.sha256(stem.encode()).digest()[0] < 51 else "train2017"


def build_dataset(archive, output, expected_sha256=SOURCE_SHA256):
    hasher = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected_sha256:
        raise ValueError("unexpected COMICS archive SHA-256")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")

    annotations = {"train2017": [], "val2017": []}
    images = {"train2017": [], "val2017": []}
    annotation_id = 1
    with zipfile.ZipFile(archive) as source:
        image_names = sorted(name for name in source.namelist() if name.startswith("Images/") and name.endswith(".jpg"))
        annotation_names = set(source.namelist())
        for image_id, image_name in enumerate(image_names, start=1):
            stem = Path(image_name).stem
            split = split_for(stem)
            data = source.read(image_name)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size

            destination = output / split / Path(image_name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            images[split].append({"id": image_id, "file_name": destination.name, "width": width, "height": height})

            annotation_name = f"Annotations/{stem}.txt"
            if annotation_name not in annotation_names:
                continue
            for line in source.read(annotation_name).decode().splitlines():
                label, x1, y1, x2, y2 = (int(value) for value in line.split())
                if (x1, y1, x2, y2) == (0, 0, 0, 0):
                    continue
                if label != 1 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"invalid annotation in {annotation_name}: {line}")
                box_width, box_height = x2 - x1, y2 - y1
                annotations[split].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                })
                annotation_id += 1

    annotation_directory = output / "annotations"
    annotation_directory.mkdir(parents=True, exist_ok=True)
    for split in ("train2017", "val2017"):
        payload = {
            "info": {"description": "COMICS manual panel annotations", "source": SOURCE, "sha256": digest},
            "licenses": [{"id": 1, "name": "MIT", "url": "https://github.com/miyyer/comics/blob/master/LICENSE"}],
            "categories": [{"id": 1, "name": "panel"}],
            "images": images[split],
            "annotations": annotations[split],
        }
        (annotation_directory / f"instances_{split}.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    return {
        "sha256": digest,
        "trainPages": len(images["train2017"]),
        "validationPages": len(images["val2017"]),
        "panelBoxes": sum(len(values) for values in annotations.values()),
        "fallbackPages": len(image_names) - len({item["image_id"] for values in annotations.values() for item in values}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()

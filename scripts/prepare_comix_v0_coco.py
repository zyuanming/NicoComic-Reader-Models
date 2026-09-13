#!/usr/bin/env python3
"""Convert verified Comix v0 WebDataset shards into a YOLOX COCO dataset."""

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image


POSITIVE_PAGE_CLASS = "story"
NEGATIVE_PAGE_CLASSES = {"advertisement", "cover", "textstory"}


def split_for(book_id):
    """Keep every page from a book in one deterministic split."""
    return "val2017" if hashlib.sha256(book_id.encode()).digest()[0] < 51 else "train2017"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clipped_box(raw_box, width, height):
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in raw_box)
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def build_dataset(manifest_path, archives, output):
    manifest = json.loads(manifest_path.read_text())
    expected = {item["file"]: item for item in manifest["shards"]}
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")

    for archive in archives:
        descriptor = expected.get(archive.name)
        if descriptor is None:
            raise ValueError(f"unlisted shard: {archive.name}")
        if archive.stat().st_size != descriptor["bytes"] or sha256(archive) != descriptor["sha256"]:
            raise ValueError(f"unexpected shard bytes or SHA-256: {archive.name}")

    images = {"train2017": [], "val2017": []}
    annotations = {"train2017": [], "val2017": []}
    page_classes = Counter()
    excluded_classes = Counter()
    seen_pages = set()
    annotation_id = 1
    image_id = 1
    positive_pages = 0
    negative_pages = 0

    for archive in sorted(archives, key=lambda item: item.name):
        with tarfile.open(archive) as source:
            members = {member.name: member for member in source.getmembers() if member.isfile()}
            for metadata_name in sorted(name for name in members if name.endswith(".json")):
                metadata = json.load(source.extractfile(members[metadata_name]))
                page_id = metadata["page_id"]
                if not page_id or Path(page_id).name != page_id:
                    raise ValueError(f"unsafe page id: {page_id!r}")
                if page_id in seen_pages:
                    raise ValueError(f"duplicate page id: {page_id}")
                seen_pages.add(page_id)

                page_class = metadata.get("page_class", "unknown").lower()
                page_classes[page_class] += 1
                if page_class != POSITIVE_PAGE_CLASS and page_class not in NEGATIVE_PAGE_CLASSES:
                    excluded_classes[page_class] += 1
                    continue

                image_name = metadata["image"]["file"]
                image_member = members.get(image_name)
                if image_member is None:
                    raise ValueError(f"missing image for {page_id}: {image_name}")
                image_data = source.extractfile(image_member).read()
                with Image.open(io.BytesIO(image_data)) as image:
                    width, height = image.size
                if [width, height] != [metadata["image"]["width"], metadata["image"]["height"]]:
                    raise ValueError(f"image dimensions changed: {page_id}")

                panel_boxes = []
                if page_class == POSITIVE_PAGE_CLASS:
                    panels = metadata.get("detections", {}).get("fasterrcnn", {}).get("panels", [])
                    panel_boxes = [box for panel in panels if (box := clipped_box(panel.get("bbox"), width, height))]
                    if len(panel_boxes) < 2:
                        excluded_classes["story-with-fewer-than-two-panels"] += 1
                        continue

                split = split_for(metadata["book_id"])
                destination_name = f"{page_id}{Path(image_name).suffix.lower()}"
                destination = output / split / destination_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(image_data)
                images[split].append({
                    "id": image_id,
                    "file_name": destination_name,
                    "width": width,
                    "height": height,
                    "license": 1,
                })

                if page_class == POSITIVE_PAGE_CLASS:
                    positive_pages += 1
                    for box in panel_boxes:
                        annotations[split].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": box,
                            "area": box[2] * box[3],
                            "iscrowd": 0,
                        })
                        annotation_id += 1
                else:
                    negative_pages += 1
                image_id += 1

    annotation_directory = output / "annotations"
    annotation_directory.mkdir(parents=True, exist_ok=True)
    for split in images:
        payload = {
            "info": {
                "description": "Comix v0 tiny pages; story-page Faster R-CNN pseudo labels and non-story negative pages",
                "source": manifest["sourceURL"],
                "source_revision": manifest["repositoryRevision"],
                "annotation_kind": manifest["annotationKind"],
            },
            "licenses": [{"id": 1, "name": manifest["license"], "url": "https://creativecommons.org/publicdomain/zero/1.0/"}],
            "categories": [{"id": 1, "name": "panel", "supercategory": "panel"}],
            "images": images[split],
            "annotations": annotations[split],
        }
        (annotation_directory / f"instances_{split}.json").write_text(json.dumps(payload, separators=(",", ":")))

    return {
        "sourceRevision": manifest["repositoryRevision"],
        "shards": [archive.name for archive in sorted(archives, key=lambda item: item.name)],
        "pagesSeen": len(seen_pages),
        "pagesIncluded": sum(map(len, images.values())),
        "positivePages": positive_pages,
        "negativePages": negative_pages,
        "panelBoxes": sum(map(len, annotations.values())),
        "pageClasses": dict(sorted(page_classes.items())),
        "excludedClasses": dict(sorted(excluded_classes.items())),
        "splits": {split: len(items) for split, items in images.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.manifest, args.archives, args.output), indent=2))


if __name__ == "__main__":
    main()

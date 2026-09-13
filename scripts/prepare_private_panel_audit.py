#!/usr/bin/env python3
"""Build an anonymous local audit set from NicoComic's private manifest."""

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "heic", "heif", "avif", "tiff", "tif"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def natural_key(value):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value.replace("\\", "/"))]


def ordered_images(names):
    result = []
    for name in names:
        normalized = name.replace("\\", "/")
        parts = normalized.split("/")
        if normalized.startswith("__MACOSX") or any(part.startswith(".") for part in parts):
            continue
        if Path(normalized).suffix.lower().lstrip(".") in IMAGE_EXTENSIONS:
            result.append(name)
    return sorted(result, key=natural_key)


def rectangle(value):
    origin, size = value
    x, y = origin
    width, height = size
    if min(x, y, width, height) < 0 or x + width > 1 or y + height > 1:
        raise ValueError(f"Invalid normalized rectangle: {value}")
    return [x, y, x + width, y + height]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output directory must be empty")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schemaVersion") != 1:
        raise SystemExit("Unsupported manifest schema")
    archive_path = args.manifest.parent / manifest["annotationArchive"]
    annotations = json.loads(archive_path.read_text())
    corrections = annotations.get("layoutCorrections", {})

    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    truth_pages = []
    for volume in manifest["volumes"]:
        comic = args.manifest.parent / volume["file"]
        if digest(comic) != volume["sha256"]:
            raise SystemExit(f"Content hash mismatch: {volume['id']}")
        correction = corrections.get(volume["sha256"])
        if correction is None:
            raise SystemExit(f"Missing layout correction: {volume['id']}")
        reviewed = set(correction.get("reviewedPanelPages", []))
        disabled = set(correction.get("panelDisabledPages", []))
        panel_pages = set(volume["panelPages"])
        fallback_pages = set(volume["fallbackPages"])
        if not panel_pages <= reviewed or not fallback_pages <= disabled:
            raise SystemExit(f"Incomplete reviewed annotations: {volume['id']}")

        with zipfile.ZipFile(comic) as source:
            names = ordered_images(source.namelist())
            selected = sorted(panel_pages | fallback_pages)
            if not selected or selected[-1] >= len(names):
                raise SystemExit(f"Page index out of range: {volume['id']}")
            for page_index in selected:
                source_name = names[page_index]
                suffix = Path(source_name).suffix.lower()
                anonymous = f"{volume['id']}-{page_index:04d}{suffix}"
                with source.open(source_name) as page, (images_dir / anonymous).open("wb") as destination:
                    shutil.copyfileobj(page, destination)
                regions = correction.get("panelRegions", {}).get(str(page_index), [])
                truth_pages.append({
                    "id": f"{volume['id']}-{page_index:04d}",
                    "file": anonymous,
                    "readingDirection": volume["readingDirection"],
                    "expectation": "fallback" if page_index in fallback_pages else "panels",
                    "regions": [
                        {"order": item["order"], "box": rectangle(item["rect"])}
                        for item in sorted(regions, key=lambda item: item["order"])
                    ],
                })

    directions = {page["readingDirection"] for page in truth_pages}
    truth = {
        "schemaVersion": 1,
        "readingDirection": next(iter(directions)) if len(directions) == 1 else "mixed",
        "rights": manifest["rights"],
        "pages": truth_pages,
    }
    (args.output / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(json.dumps({
        "pages": len(truth_pages),
        "panelPages": sum(page["expectation"] == "panels" for page in truth_pages),
        "fallbackPages": sum(page["expectation"] == "fallback" for page in truth_pages),
    }))


if __name__ == "__main__":
    main()

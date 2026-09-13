#!/usr/bin/env python3
"""Run a released Core ML panel detector over a local image corpus."""

import argparse
import json
import statistics
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image, ImageDraw

RTDETR_INPUT_EDGE = 1280
YOLOX_INPUT_EDGE = 416
PANEL_LABEL = 2


def containment(candidate, container):
    x1, y1 = max(candidate[0], container[0]), max(candidate[1], container[1])
    x2, y2 = min(candidate[2], container[2]), min(candidate[3], container[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(0, candidate[2] - candidate[0]) * max(0, candidate[3] - candidate[1])
    return overlap / area if area else 0


def intersection_over_union(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0


def postprocess_rtdetr(prediction, minimum_score):
    labels = np.asarray(prediction["labels"])[0].astype(int)
    boxes = np.asarray(prediction["boxes"])[0]
    scores = np.asarray(prediction["scores"])[0]
    candidates = sorted(
        (
            {"score": float(score), "box": [float(value / RTDETR_INPUT_EDGE) for value in box]}
            for label, box, score in zip(labels, boxes, scores)
            if label == PANEL_LABEL and score >= minimum_score
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    accepted = []
    for candidate in candidates:
        if any(containment(candidate["box"], item["box"]) >= 0.90 for item in accepted):
            continue
        accepted.append(candidate)
    return accepted


def postprocess_yolox(prediction, minimum_score, content_size, nms_threshold=0.45):
    rows = np.asarray(prediction["detections"])[0]
    width, height = content_size
    candidates = []
    for center_x, center_y, box_width, box_height, object_score, class_score in rows:
        score = float(object_score * class_score)
        if score < minimum_score:
            continue
        box = [
            max(0.0, float(center_x - box_width / 2)) / width,
            max(0.0, float(center_y - box_height / 2)) / height,
            min(width, float(center_x + box_width / 2)) / width,
            min(height, float(center_y + box_height / 2)) / height,
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        candidates.append({"score": score, "box": box})

    accepted = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if any(intersection_over_union(candidate["box"], item["box"]) > nms_threshold for item in accepted):
            continue
        accepted.append(candidate)
    return accepted


def prepare_image(source, architecture):
    if architecture == "yolox":
        ratio = min(YOLOX_INPUT_EDGE / source.width, YOLOX_INPUT_EDGE / source.height)
        size = (int(source.width * ratio), int(source.height * ratio))
        canvas = Image.new("RGB", (YOLOX_INPUT_EDGE, YOLOX_INPUT_EDGE), (114, 114, 114))
        canvas.paste(source.resize(size, Image.Resampling.BILINEAR), (0, 0))
        return canvas, size
    return source.resize((RTDETR_INPUT_EDGE, RTDETR_INPUT_EDGE), Image.Resampling.BILINEAR), None


def draw_overlay(image, detections):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    stroke = max(2, round(min(width, height) / 180))
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        rect = (x1 * width, y1 * height, x2 * width, y2 * height)
        draw.rectangle(rect, outline=(255, 60, 45), width=stroke)
        draw.text((rect[0] + stroke, rect[1] + stroke), str(index), fill=(255, 60, 45), stroke_width=1, stroke_fill="white")
    return overlay


def write_contact_sheet(overlays, destination):
    thumb_size = (240, 340)
    columns = 5
    rows = (len(overlays) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "#202020")
    for index, (name, overlay) in enumerate(overlays):
        thumbnail = overlay.copy()
        thumbnail.thumbnail((thumb_size[0] - 8, thumb_size[1] - 28))
        x = index % columns * thumb_size[0] + (thumb_size[0] - thumbnail.width) // 2
        row_y = index // columns * thumb_size[1]
        y = row_y + 20 + (thumb_size[1] - 20 - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
        ImageDraw.Draw(sheet).text((index % columns * thumb_size[0] + 6, row_y + 4), name, fill="white")
    sheet.save(destination, quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-score", type=float)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No images found")

    args.output.mkdir(parents=True, exist_ok=True)
    overlay_directory = args.output / "overlays"
    overlay_directory.mkdir(exist_ok=True)
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_ONLY)
    specification = model.get_spec()
    output_names = {item.name for item in specification.description.output}
    architecture = "yolox" if "detections" in output_names else "rtdetr"
    input_name = specification.description.input[0].name
    minimum_score = args.minimum_score if args.minimum_score is not None else (0.01 if architecture == "yolox" else 0.35)

    timings = []
    results = []
    overlays = []
    for path in paths:
        with Image.open(path) as opened:
            source = opened.convert("RGB")
        prepared, content_size = prepare_image(source, architecture)
        started = time.perf_counter()
        prediction = model.predict({input_name: prepared})
        detections = (
            postprocess_yolox(prediction, minimum_score, content_size)
            if architecture == "yolox"
            else postprocess_rtdetr(prediction, minimum_score)
        )
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)
        overlay = draw_overlay(source, detections)
        overlay.save(overlay_directory / f"{path.stem}.jpg", quality=90)
        overlays.append((path.name, overlay))
        results.append({
            "page": path.name,
            "panelCount": len(detections),
            "elapsedMilliseconds": elapsed,
            "detections": detections,
        })

    counts = [len(page["detections"]) for page in results]
    report = {
        "schemaVersion": 1,
        "model": args.model.name,
        "architecture": architecture,
        "minimumScore": minimum_score,
        "pageCount": len(results),
        "zeroOrOnePanelPages": sum(count <= 1 for count in counts),
        "medianPanelCount": statistics.median(counts),
        "medianMilliseconds": statistics.median(timings),
        "p95Milliseconds": float(np.percentile(timings, 95)),
        "pages": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_contact_sheet(overlays, args.output / "contact-sheet.jpg")
    print(json.dumps({key: value for key, value in report.items() if key != "pages"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a NicoComic YOLOX checkpoint and retain every page result."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolox.data.data_augment import preproc
from yolox.utils import postprocess

from yolox_export_support import load_yolox_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--nms", type=float, default=0.45)
    args = parser.parse_args()

    if not 0 <= args.confidence <= 1 or not 0 <= args.nms <= 1:
        raise SystemExit("Confidence and NMS thresholds must be between zero and one")

    exp, model = load_yolox_runtime(args.experiment, args.checkpoint)
    truth = COCO(str(args.annotations))
    category_id = truth.getCatIds()[0]
    detections = []
    pages = []

    with torch.no_grad():
        for image_id in sorted(truth.getImgIds()):
            info = truth.loadImgs(image_id)[0]
            source = cv2.imread(str(args.images / info["file_name"]))
            if source is None:
                raise SystemExit(f"Cannot read validation image: {info['file_name']}")
            prepared, ratio = preproc(source, exp.test_size)
            tensor = torch.from_numpy(prepared).unsqueeze(0)
            output = postprocess(model(tensor), 1, args.confidence, args.nms)[0]
            page_detections = []
            if output is not None:
                for row in output.cpu().numpy():
                    x1, y1, x2, y2 = np.clip(
                        row[:4] / ratio,
                        [0, 0, 0, 0],
                        [info["width"], info["height"], info["width"], info["height"]],
                    )
                    width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
                    if width == 0 or height == 0:
                        continue
                    item = {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "score": float(row[4] * row[5]),
                    }
                    detections.append(item)
                    page_detections.append(item)
            pages.append({"imageID": image_id, "file": info["file_name"], "detections": page_detections})

    stats = [0.0] * 12
    if detections:
        result = truth.loadRes(detections)
        evaluator = COCOeval(truth, result, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = [float(value) for value in evaluator.stats]

    report = {
        "schemaVersion": 1,
        "checkpoint": args.checkpoint.name,
        "imageCount": len(pages),
        "detectionCount": len(detections),
        "confidence": args.confidence,
        "nms": args.nms,
        "metrics": {"ap50To95": stats[0], "ap50": stats[1], "ar100": stats[8]},
        "pages": pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check Core ML parity, size, latency, and expected panel counts."""

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import onnxruntime as ort
from PIL import Image


INPUT_EDGE = 1280
PANEL_LABEL = 2
MINIMUM_SCORE = 0.35


def intersection_over_union(lhs, rhs):
    x1, y1 = max(lhs[0], rhs[0]), max(lhs[1], rhs[1])
    x2, y2 = min(lhs[2], rhs[2]), min(lhs[3], rhs[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    lhs_area = (lhs[2] - lhs[0]) * (lhs[3] - lhs[1])
    rhs_area = (rhs[2] - rhs[0]) * (rhs[3] - rhs[1])
    return intersection / (lhs_area + rhs_area - intersection) if lhs_area + rhs_area > intersection else 0


def containment(candidate, container):
    x1, y1 = max(candidate[0], container[0]), max(candidate[1], container[1])
    x2, y2 = min(candidate[2], container[2]), min(candidate[3], container[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    return intersection / area if area else 0


def detections(labels, boxes, scores):
    candidates = [
        {"score": float(score), "box": [float(value / INPUT_EDGE) for value in box]}
        for label, box, score in zip(labels.astype(int), boxes, scores)
        if label == PANEL_LABEL and score >= MINIMUM_SCORE
    ]
    candidates.sort(key=lambda value: value["score"], reverse=True)
    accepted = []
    for candidate in candidates:
        if not any(containment(candidate["box"], other["box"]) >= 0.90 for other in accepted):
            accepted.append(candidate)
    return accepted


def directory_size(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("coreml_model", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--expected-count", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if directory_size(args.coreml_model) > 150 * 1024 * 1024:
        raise SystemExit("Core ML package exceeds the 150 MiB product limit")

    # CPU_ONLY is deliberate: this model produced invalid boxes with CPU_AND_GPU on the audit Mac.
    core_ml = ct.models.MLModel(str(args.coreml_model), compute_units=ct.ComputeUnit.CPU_ONLY)
    onnx_model = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])
    results = []
    for path in args.images:
        image = Image.open(path).convert("RGB").resize((INPUT_EDGE, INPUT_EDGE), Image.Resampling.BILINEAR)
        tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255
        labels, boxes, scores = onnx_model.run(None, {
            "images": tensor,
            "orig_target_sizes": np.array([[INPUT_EDGE, INPUT_EDGE]], dtype=np.int64),
        })
        expected = detections(labels[0], boxes[0], scores[0])

        timings = []
        for _ in range(6):
            started = time.perf_counter()
            prediction = core_ml.predict({"image": image})
            timings.append((time.perf_counter() - started) * 1000)
        actual = detections(
            np.asarray(prediction["labels"])[0],
            np.asarray(prediction["boxes"])[0],
            np.asarray(prediction["scores"])[0],
        )
        remaining = list(range(len(actual)))
        matched = []
        for detection in expected:
            index = max(remaining, key=lambda value: intersection_over_union(detection["box"], actual[value]["box"])) if remaining else None
            matched.append(intersection_over_union(detection["box"], actual[index]["box"]) if index is not None else 0)
            if index is not None:
                remaining.remove(index)
        result = {
            "page": path.name,
            "onnxCount": len(expected),
            "coreMLCount": len(actual),
            "minimumMatchedIoU": min(matched, default=0),
            "warmP95Milliseconds": float(np.percentile(timings[1:], 95)),
        }
        results.append(result)
        if len(expected) != args.expected_count or len(actual) != args.expected_count:
            raise SystemExit(f"{path.name}: expected {args.expected_count} panels")
        if result["minimumMatchedIoU"] < 0.99 or result["warmP95Milliseconds"] >= 1000:
            raise SystemExit(f"{path.name}: parity or latency gate failed")

    report = json.dumps(results, indent=2) + "\n"
    print(report, end="")
    if args.report:
        args.report.write_text(report)


if __name__ == "__main__":
    main()

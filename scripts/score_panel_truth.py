#!/usr/bin/env python3
"""Score an audit report against ordered, normalized panel rectangles."""

import argparse
import json
from functools import lru_cache
from pathlib import Path


def containment(candidate, container):
    x1, y1 = max(candidate[0], container[0]), max(candidate[1], container[1])
    x2, y2 = min(candidate[2], container[2]), min(candidate[3], container[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, candidate[2] - candidate[0]) * max(0.0, candidate[3] - candidate[1])
    return overlap / area if area else 0.0


def ordered(boxes, direction):
    boxes = [
        [max(0.0, box[0]), max(0.0, box[1]), min(1.0, box[2]), min(1.0, box[3])]
        for box in boxes
    ]
    rows = []
    for rect in sorted(boxes, key=lambda box: (box[1], box[0])):
        best = min(
            range(len(rows)),
            key=lambda index: abs((rect[1] + rect[3]) / 2 - rows[index]["center"]),
            default=None,
        )
        if best is not None:
            row = rows[best]
            tolerance = min(rect[3] - rect[1], row["minimum_height"]) * 0.35
            if abs((rect[1] + rect[3]) / 2 - row["center"]) <= tolerance:
                row["boxes"].append(rect)
                row["center"] = sum((box[1] + box[3]) / 2 for box in row["boxes"]) / len(row["boxes"])
                row["minimum_height"] = min(row["minimum_height"], rect[3] - rect[1])
                continue
        rows.append({"boxes": [rect], "center": (rect[1] + rect[3]) / 2, "minimum_height": rect[3] - rect[1]})
    rows.sort(key=lambda row: min(box[1] for box in row["boxes"]))
    reverse = direction == "rightToLeft"
    return [
        box
        for row in rows
        for box in sorted(row["boxes"], key=lambda box: (box[0] + box[2]) / 2, reverse=reverse)
    ]


def postprocess(
    detections,
    minimum_score=0.80,
    minimum_dimension=0.06,
    maximum_overlap=0.50,
    direction="rightToLeft",
):
    accepted = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        box = detection["box"]
        if (
            detection["score"] < minimum_score
            or box[2] - box[0] < minimum_dimension
            or box[3] - box[1] < minimum_dimension
        ):
            continue
        if any(containment(box, item) >= 0.90 for item in accepted):
            continue
        accepted.append(box)
    if len(accepted) < 2:
        return []
    for index, box in enumerate(accepted):
        for other in accepted[index + 1 :]:
            if max(containment(box, other), containment(other, box)) > maximum_overlap:
                return []
    return ordered(accepted, direction)


def iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - overlap
    )
    return overlap / union if union else 0.0


def maximum_match(expected, detected):
    if not expected or not detected:
        return [], []

    @lru_cache(maxsize=None)
    def solve(expected_index, used_mask):
        if expected_index == len(expected):
            return 0.0, ()
        best = (-1.0, ())
        for detected_index in range(len(detected)):
            if used_mask & (1 << detected_index):
                continue
            score, suffix = solve(expected_index + 1, used_mask | (1 << detected_index))
            candidate = (
                iou(expected[expected_index], detected[detected_index]) + score,
                (detected_index,) + suffix,
            )
            if candidate[0] > best[0]:
                best = candidate
        return best

    _, indices = solve(0, 0)
    return list(indices), [
        iou(expected[index], detected[detected_index])
        for index, detected_index in enumerate(indices)
    ]


def score(truth, audit, iou_threshold=0.70):
    predictions = {page["page"]: page for page in audit["pages"]}
    results = []
    for page in truth["pages"]:
        prediction = predictions.get(page["file"], {"detections": []})
        expected = [
            region["box"]
            for region in sorted(page["regions"], key=lambda region: region["order"])
        ]
        detected = postprocess(
            prediction["detections"],
            direction=truth["readingDirection"],
        )
        indices, matched_ious = maximum_match(expected, detected)
        passed = (
            len(expected) == len(detected)
            and all(value >= iou_threshold for value in matched_ious)
            and indices == list(range(len(detected)))
        )
        results.append(
            {
                "id": page["id"],
                "file": page["file"],
                "expectedCount": len(expected),
                "detectedCount": len(detected),
                "matchedIoU": matched_ious,
                "matchedIndices": indices,
                "passed": passed,
            }
        )
    passed = sum(result["passed"] for result in results)
    return {
        "schemaVersion": 1,
        "pageCount": len(results),
        "passedPages": passed,
        "noModificationRate": passed / len(results) if results else 0.0,
        "pages": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("truth", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = score(json.loads(args.truth.read_text()), json.loads(args.audit.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("pageCount", "passedPages", "noModificationRate")}, indent=2))


if __name__ == "__main__":
    main()

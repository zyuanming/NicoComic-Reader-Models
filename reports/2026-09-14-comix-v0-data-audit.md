# Comix v0 CC0 training-data audit

## Decision

Use `emanuelevivoli/comix_v0_tiny_pages` as additional training input for the next native `640 × 640` YOLOX-Nano experiment. Do not use its generated detections as validation truth, and do not change the NicoComic download descriptor until the existing private 60-panel/20-fallback gate passes.

## Provenance

- Dataset revision: `5347bee7a327ea794a281af68107c386596242c4`.
- Dataset card license: CC0-1.0; source identified as public-domain Digital Comic Museum pages.
- Public access check: no account or gated-dataset grant was required.
- Published tiny set: 6,750 pages in 14 WebDataset shards, 2,229,207,040 bytes total.
- Labels used for training: `detections.fasterrcnn.panels`. These are pseudo labels, not human annotations.

The checked-in manifest records every shard's byte length and SHA-256 at the fixed dataset revision. The converter rejects an unlisted shard, changed byte length, changed digest, duplicate page ID, missing image, or changed image dimensions.

## First-shard evidence

`comix-pages-train-0000.tar` matched SHA-256 `3c39768e98e35929ce907762b8f76d5d164aa239c1c03c1c5ff4c49d66a7fc22` and contained 500 pages:

| Page class | Pages | Training treatment |
|---|---:|---|
| story | 358 | positive, 2,542 panel boxes |
| advertisement | 58 | whole-page negative |
| cover | 13 | whole-page negative |
| textstory | 20 | whole-page negative |
| first-page | 51 | excluded because the class mixes covers and story layouts |

Story pages contained 3–15 boxes, median 7 and mean 7.10. A deterministic 24-page overlay review showed the selected boxes generally follow visible gutters on the sampled Golden Age pages. This is a semantic spot check, not an accuracy percentage.

The generated COCO subset contains 449 pages and 2,542 boxes. It splits by `book_id`, keeping all pages of a comic in one split; the first shard produced 309 train pages and 140 validation pages. The validation split only measures fit to pseudo labels. NicoComic's independent private gate remains the product decision.

## Training contract

The next experiment uses the existing Apache-2.0 YOLOX-Nano code with a fixed `640 × 640` inference tensor. Source pages are decoded once, aspect fitted, and their normalized boxes map back to the original page. This makes model compute materially smaller than the current fixed `1280 × 1280` RT-DETR tensor while retaining more boundary detail than the rejected 416/320 experiments.

Required order:

1. Convert all 14 verified shards and train only on public data.
2. Record public pseudo-label validation separately.
3. Export ONNX and Core ML, then measure model-declared 640 input latency and memory.
4. Run the unchanged private 60-panel/20-fallback scorer.
5. Publish a model and update the App only if exact panels are at least 95%, adjacent-order error is below 2%, fallback is 100%, and memory materially improves.

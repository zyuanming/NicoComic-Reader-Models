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

## Full-corpus evidence

All 14 shards matched the pinned byte lengths and SHA-256 values. The converter processed all 6,750 pages and included 5,868 pages: 5,108 story-page positives with 40,383 pseudo-label boxes, plus 760 cover, advertisement, and text-story whole-page negatives. It excluded 875 ambiguous `first-page` pages and seven story pages with fewer than two valid boxes. The comic-grouped split contains 4,793 train pages and 1,075 validation pages.

A deterministic 48-page overlay review sampled 36 positive and 12 negative pages from the full converted corpus. Positive boxes generally followed visible panel gutters; the sampled advertisements and text pages remained unboxed. This is a semantic suitability check, not an accuracy score, and does not replace human truth.

## Fixed-input architecture benchmark

Untrained YOLOX-Nano packages isolate the cost of the fixed input shape without claiming detection quality. Both packages use the same 896,754-parameter architecture and FP16 Core ML conversion. A release-built Swift process ran one warm-up and 80 CPU-only predictions against each package on the same Mac and input page:

| Input | Median | P95 | Process maximum RSS | Package |
|---|---:|---:|---:|---:|
| 416 × 416 | 9.82 ms | 10.25 ms | 41,713,664 bytes | 1.9 MiB |
| 640 × 640 | 21.45 ms | 23.11 ms | 48,955,392 bytes | 1.9 MiB |

The 640 tensor costs 2.18 times the median latency of 416 while remaining a small, tens-of-milliseconds architecture on this host. Across all 40,383 training boxes, the scaled panel short-side first percentile is 89 px at 640, versus 71 px at 512 and 58 px at 416. This measures annotated panel geometry, not gutter-line thickness; 640 is the next quality/speed compromise to train, not proof that a smaller input cannot work.

Reproduce latency with `xcrun swiftc -O scripts/benchmark_coreml_architecture.swift -o /tmp/nicocomic-coreml-benchmark`, then run the executable under `/usr/bin/time -l` to record process maximum RSS. These architecture measurements do not replace the private 60/20 quality gate or iPhone/iPad simulator memory evidence.

Required order:

1. [x] Verify and convert all 14 public-data shards.
2. [ ] Train the 640 model and record public pseudo-label validation separately.
3. [ ] Export ONNX and Core ML, then measure model-declared 640 input latency and memory.
4. [ ] Run the unchanged private 60-panel/20-fallback scorer.
5. [ ] Publish a model update to the App only if exact panels are at least 95%, adjacent-order error is below 2%, fallback is 100%, and memory materially improves.

# YOLOX-Nano 416/320 quality gate

Date: 2026-09-14

This report contains no private page, source filename, annotation, or model weight.

## Training contract

- Model: Apache-2.0 YOLOX-Nano, 896,754 parameters, one `panel` class.
- Source: official YOLOX tag `0.3.0`, commit `419778480ab6ec0590e5d3831b3afb3b46ab2aa3`.
- Data: UMD COMICS manual panel archive, verified SHA-256 `9d6bcfa5d9c3a1650529db9fd5b8ea2281529ea0ada00ad7a48ce61d3687ec07`.
- Split: 392 train pages / 2,302 boxes and 109 validation pages / 678 boxes.
- Training: 100 CPU epochs from scratch; Mosaic disabled for the final 10 epochs; EMA checkpoint used for every export.
- Executed time: 6,236.91 seconds (1.73 hours). Loss moved from 15.1242 to 1.7305.

## Public validation

Every row uses all 109 validation pages and COCO evaluation. Intermediate snapshots were retained because the last checkpoint was not assumed to be the best.

| Checkpoint | Input | AP50:95 | AP50 | AR100 |
| --- | ---: | ---: | ---: | ---: |
| epoch 29 | 416 | 0.5863 | 0.8892 | 0.6824 |
| epoch 41 | 416 | 0.6997 | **0.9389** | 0.7718 |
| epoch 50 | 416 | 0.6108 | 0.8800 | 0.7214 |
| epoch 60 | 416 | 0.6756 | 0.8998 | 0.7771 |
| epoch 80 | 416 | 0.7209 | 0.9257 | 0.8032 |
| epoch 100 | 416 | **0.7563** | 0.9309 | **0.8229** |
| epoch 100 | 320 | 0.6825 | 0.9298 | 0.7646 |

The 320 input kept AP50 but lost 7.39 points of stricter AP50:95 against 416. It therefore remains an experiment rather than the default contract.

## Core ML and private gate

The final EMA checkpoint exported to a 3,704,906-byte ONNX model. FP16 Core ML packages were 1,994,256 bytes at 416 and 1,985,596 bytes at 320. The audit reads the model-declared image size, applies top-left letterbox resize, and maps every box back to normalized original-page coordinates.

The same anonymous 60-panel/20-fallback truth set scored both inputs. Thresholds from 0.10 through 0.90 were retained as per-page JSON outside this repository.

| Input | Median | P95 | Best panel pages at any threshold | Fallback at that threshold | Best panel pages while fallback is 20/20 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 416 | 12.54 ms | 15.63 ms | 9/60 | 16/20 at 0.45 | 0/60 at 0.85 |
| 320 | 7.32 ms | 10.80 ms | 10/60 | 14/20 at 0.40 | 0/60 at 0.85 |

Neither input approaches the required 57/60 panel pages, less than 2% adjacent-order error, and 20/20 fallback simultaneously. The failure is missing or differently grouped panels, not inability to see boundaries at reduced resolution.

## Public-only teacher experiment

The released Apache-2.0 RT-DETR candidate generated high-confidence pseudo labels on the public UMD train split only. A 416 student snapshot was fine-tuned without Mosaic on 189 pages and 1,059 accepted boxes. No private page was used for training.

The result reached public AP50:95 0.7123 / AP50 0.9320. On the private gate, threshold 0.45 achieved 20/20 fallback but only 5/60 exact panel pages. Median matched IoU improved, but missing and grouped regions still prevented a usable reading plan. Distillation on the same small public domain did not close the gap.

## Data findings

- UMD documents 198,657 public-domain pages, but the downloadable human panel archive contains only the 501 pages used above. The full original-page archive is 129,199,164,463 bytes and its large-scale panel locations are pseudo annotations.
- [`emanuelevivoli/comix_books_v0`](https://huggingface.co/datasets/emanuelevivoli/comix_books_v0) declares CC0-1.0, 952,433 public-domain pages, and 930,944 pages with segmentation. Its panel boxes and masks are Faster R-CNN/SAM pseudo annotations, and file access currently requires an authenticated gated-dataset grant.
- DCM772 is described by its authors as 772 public-domain pages with human panel boxes, but the former public repository currently redirects to a sign-in page. It is not used until the exact annotation package and terms can be acquired and preserved.
- eBDtheque explicitly limits use to scientific, non-commercial work and remains excluded.

## Decision

**No-Go for release and App integration.** Keep the current RT-DETR download descriptor and production reader path. The 416/320 exports and checkpoints remain local evidence and are not published.

The next training attempt needs substantially more legally usable pages with labels that match NicoComic reading-region grouping. A gated CC0 public-domain corpus can be evaluated after authenticated access; pseudo labels remain training input, never quality truth. The fixed anonymous 60/20 set remains the replacement gate.

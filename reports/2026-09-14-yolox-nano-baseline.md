# YOLOX-Nano panel baseline

Date: 2026-09-14

## Inputs

- Architecture: Megvii YOLOX tag `0.3.0`, commit `419778480ab6ec0590e5d3831b3afb3b46ab2aa3`, Apache-2.0.
- Dataset: UMD COMICS manual panel archive, SHA-256 `9d6bcfa5d9c3a1650529db9fd5b8ea2281529ea0ada00ad7a48ce61d3687ec07`, MIT.
- Model contract: one `panel` class, `416 × 416`, 896,754 parameters.
- Initialization: from scratch. No Ultralytics code or weights and no COCO pretrained weights are used.

## Deterministic data build

`scripts/prepare_comics_coco.py` verifies the archive digest, ignores its single all-zero sentinel, validates every remaining rectangle against the JPEG dimensions, and writes COCO JSON plus source JPEG bytes:

| Split | Pages | Panel boxes | Pages without boxes |
| --- | ---: | ---: | ---: |
| train2017 | 392 | 2,302 | 50 |
| val2017 | 109 | 678 | 8 |
| total | 501 | 2,980 | 58 |

The split is derived from the first byte of each filename stem's SHA-256, so another machine produces the same sets.

## Executed smoke evidence

- Dataset converter test passed, including one valid panel and one whole-page negative.
- The complete official archive converted successfully with the counts above.
- Model evaluation forward pass returned `[1, 3549, 6]`; after warm-up, MPS forward took 30–31 ms on the current Apple Silicon Mac.
- A real augmented two-image training batch completed forward and backward on CPU in 527.8 ms and on MPS in 2208.2 ms after two upstream MPS compatibility substitutions. MPS is therefore not selected as the training baseline.

## Gate

These checks prove the data and model contracts connect; they do not prove panel quality. Full training, Core ML conversion, public-validation metrics, private 60-panel/20-fallback scoring, and simulator memory comparison remain required before an App candidate exists.

## Local Apple Silicon limit

A bounded one-epoch MPS run without dynamic Mosaic completed 49 batches at batch size 8 in 219.88 seconds with total loss 17.0731. The process used about 1 GiB resident memory. A linear 100-epoch run would take at least 6.1 hours before validation and export, so local MPS is retained as a smoke path rather than the production training route. The public Colab notebook uses the official CUDA trainer and preserves checkpoints.

The first temporary benchmark accidentally enumerated YOLOX's intentionally infinite sampler. The figure above comes from a corrected run bounded to `len(loader)` batches; no timing from the invalid run is used.

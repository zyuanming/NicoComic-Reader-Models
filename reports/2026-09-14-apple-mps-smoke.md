# Apple MPS training smoke

## Purpose

Verify that the pinned Apache-2.0 YOLOX source can train the native 512 panel model on Apple silicon without changing checkpoint or evaluation formats.

## Environment

- Mac Studio, Apple M1 Max, 32 GB memory.
- macOS 26.6.2, arm64.
- PyTorch 2.7.0 with MPS available.
- YOLOX commit `419778480ab6ec0590e5d3831b3afb3b46ab2aa3` patched by `scripts/patch_yolox_mps.py`.
- Public UMD smoke split: 392 training pages.
- Input 512×512, batch 4, two data workers, one epoch.

## Result

The complete epoch finished in 223.67 seconds with finite mean loss `20.7553` and learning rate `2.5e-05`. Both `latest_ckpt.pth` and `epoch_001_ckpt.pth` were written at about 11 MiB each. The MPS process stayed below 1 GiB resident memory during observation.

At this measured 1.75 pages/second, a linear estimate for 4,793 training pages over 100 epochs is about 76 hours. This is a valid free and resumable fallback, not the primary full run while a free CUDA runtime is available. The estimate is not a completed full-corpus benchmark.

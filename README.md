# NicoComic Reader Models

Open model artifacts and reproducible conversion tools for NicoComic's on-device adaptive comic reader.

## RT-DETRv4-X Manga109-s Core ML candidate

The first release converts [`tori29umai/rtdetrv4-x-manga109s`](https://huggingface.co/tori29umai/rtdetrv4-x-manga109s) from ONNX to an FP16 Core ML package. The source model and this converted artifact are distributed under Apache-2.0.

This is a **candidate**, not a production quality claim. It passes two anonymized hard-page smoke cases and the local conversion gates. Independent 60-panel/20-fallback truth and physical iPhone/iPad measurements remain open.

### Contract

- Input: `image`, RGB `1280 × 1280`; Core ML applies `1/255` scaling.
- Outputs: `labels [1,300]`, `boxes [1,300,4]`, `scores [1,300]`.
- Labels: `0 body`, `1 text`, `2 frame`.
- Boxes: `xyxy` coordinates in the resized 1280-square input.
- Current frame threshold: `0.35`.
- Post-process: keep higher-confidence boxes first and suppress a candidate when at least 90% of its area is contained by an accepted box.
- Compute units: use CPU-only until the documented heterogeneous-compute coordinate failure is resolved on real Apple devices.

### Reproduce

Use Python 3.11 on macOS with Xcode installed:

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python scripts/export_rtdetr_coreml.py model.onnx NicoComicPanelRTDETR.mlpackage
```

The converter accepts only the audited source ONNX SHA-256:

```text
fba50583bfaaba3eed33f3eac6ca37be09b8c4882bac05da93f96697010a45b1
```

The release ZIP SHA-256 is recorded in [`checksums.txt`](checksums.txt).

## Data and privacy

This repository contains no Manga109-s images, private comics, private annotations, or audit screenshots. Obtain Manga109-s only through its official application process and comply with its terms. Do not use this model to redistribute or sell reproductions or derivatives of Manga109-s manga images.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for required attribution and citations.

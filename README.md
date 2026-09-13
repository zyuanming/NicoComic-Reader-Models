# NicoComic Reader Models

Open model artifacts and reproducible conversion tools for NicoComic's on-device adaptive comic reader.

## RT-DETRv4-X Manga109-s Core ML candidate

The first release converts [`tori29umai/rtdetrv4-x-manga109s`](https://huggingface.co/tori29umai/rtdetrv4-x-manga109s) from ONNX to an FP16 Core ML package. The source model and this converted artifact are distributed under Apache-2.0.

This is a **candidate**, not a production quality claim. It passes the local conversion gate and runs across a 60-page untouched-source audit. Independent 60-panel/20-fallback truth and physical iPhone/iPad measurements remain open.

### Contract

- Input: `image`, RGB `1280 × 1280`; Core ML applies `1/255` scaling.
- Outputs: `labels [1,300]`, `boxes [1,300,4]`, `scores [1,300]`.
- Labels: `0 body`, `1 text`, `2 frame`.
- Boxes: `xyxy` coordinates in the resized 1280-square input.
- Raw audit threshold: `0.35`; NicoComic accepts frames at `0.80`.
- App post-process: require normalized width and height of at least `0.06`, suppress a candidate when at least 90% of its area is contained by an accepted box, and return the whole page when two accepted boxes overlap by more than 50% of the smaller box.
- Compute units: use CPU-only until the documented heterogeneous-compute coordinate failure is resolved on real Apple devices.

### Reproduce

Use Python 3.11 on macOS with Xcode installed:

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python scripts/export_rtdetr_coreml.py model.onnx NicoComicPanelRTDETR.mlpackage
```

Audit the compiled model against a local image directory without copying source images into this repository:

```bash
.venv/bin/python scripts/audit_coreml_corpus.py NicoComicPanelRTDETR.mlpackage /path/to/pages /tmp/nicocomic-panel-audit --limit 60
```

The audit writes per-page JSON, private overlay images, and a contact sheet to the requested output directory. These outputs are evidence for manual review; they are not ground truth or a quality score.

Score an audit report against a private ordered-rectangle truth file:

```bash
python3 scripts/score_panel_truth.py /path/to/truth.json /path/to/report.json /tmp/panel-score.json
python3 -m unittest scripts/test_score_panel_truth.py
```

The scorer uses one-to-one maximum IoU matching at 0.70 and mirrors NicoComic deterministic right-to-left row ordering and safety fallback. It emits a result for every page and never copies source images.

The converter accepts only the audited source ONNX SHA-256:

```text
fba50583bfaaba3eed33f3eac6ca37be09b8c4882bac05da93f96697010a45b1
```

The release ZIP SHA-256 is recorded in [`checksums.txt`](checksums.txt). Input-size and simulator-memory experiments are recorded in [`reports/2026-09-14-input-memory-investigation.md`](reports/2026-09-14-input-memory-investigation.md).

### NicoComic integration evidence

NicoComic offers this candidate as an optional download. An iPhone simulator completed public Release download, SHA-256 verification, extraction, and Core ML compilation in 27.052 seconds. Two local 60-page raw-corpus runs measured 513–521 ms median / 647–757 ms P95 on the Mac host. Two iPad simulator warm runs ranged from 1.037 to 1.228 seconds per page, so a one-second physical-device target is not claimed.

The production App path now scores an independently reviewed private set at 57/60 exact pages (95%), 0.47% adjacent-order error, and 20/20 whole-page fallbacks. Three remaining failures stay in the per-page private report. These samples guide engineering; they are not a general accuracy claim.

## Lightweight training baseline

The current RT-DETR candidate meets the private quality floor but has a large simulator inference working set. The replacement experiment uses official Apache-2.0 YOLOX-Nano at `416 × 416` with one `panel` class. The first reproducible dataset is UMD's MIT-licensed COMICS manual panel archive: 501 public-domain pages, 2,980 valid boxes, and 58 pages without boxes. One all-zero sentinel annotation is ignored.

```sh
python3.11 -m venv .venv-training
.venv-training/bin/pip install -r scripts/requirements-training.txt
.venv-training/bin/pip install torch==2.7.0 torchvision==0.22.0
curl -fL -o panels_annotations.zip https://obj.umiacs.umd.edu/comics/panels_annotations.zip
.venv-training/bin/python scripts/prepare_comics_coco.py panels_annotations.zip /tmp/nicocomic-comics-coco
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
git -C YOLOX checkout 419778480ab6ec0590e5d3831b3afb3b46ab2aa3
.venv-training/bin/pip install -e YOLOX --no-build-isolation --no-deps
.venv-training/bin/python scripts/train_yolox_panels.py experiments/yolox_nano_panels.py /tmp/nicocomic-comics-coco /tmp/nicocomic-yolox-output --device cpu
.venv-training/bin/python scripts/export_yolox_onnx.py experiments/yolox_nano_panels.py /tmp/nicocomic-yolox-output/latest_ckpt.pth NicoComicPanelYOLOXNano416.onnx
.venv-training/bin/pip install coremltools==9.0
.venv-training/bin/python scripts/export_yolox_coreml.py experiments/yolox_nano_panels.py /tmp/nicocomic-yolox-output/latest_ckpt.pth NicoComicPanelYOLOXNano416.mlpackage
.venv-training/bin/python scripts/evaluate_yolox_coco.py experiments/yolox_nano_panels.py /tmp/nicocomic-yolox-output/latest_ckpt.pth /tmp/nicocomic-comics-coco/annotations/instances_val2017.json /tmp/nicocomic-comics-coco/val2017 /tmp/nicocomic-yolox-validation.json
```

The converter verifies the archive SHA-256 before writing a deterministic 392-page train and 109-page validation split. Public validation only brings up the pipeline; the private 60-panel/20-fallback gate remains the replacement decision. Executed smoke evidence is recorded in [`reports/2026-09-14-yolox-nano-baseline.md`](reports/2026-09-14-yolox-nano-baseline.md).

For a free GPU run, open [`notebooks/train_yolox_nano_panels_colab.ipynb`](notebooks/train_yolox_nano_panels_colab.ipynb) in Google Colab and select a GPU runtime. The notebook downloads only the public archive, trains with the pinned official YOLOX source, and exports an ONNX checkpoint for Core ML conversion and private quality scoring.

## Data and privacy

This repository contains no Manga109-s images, private comics, private annotations, or audit screenshots. Obtain Manga109-s only through its official application process and comply with its terms. Do not use this model to redistribute or sell reproductions or derivatives of Manga109-s manga images.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for required attribution and citations.

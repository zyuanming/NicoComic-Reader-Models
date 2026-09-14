# NicoComic Reader Models

Open model artifacts and reproducible conversion tools for NicoComic's on-device adaptive comic reader.

Model binaries, source weights, datasets, and generated packages stay out of
Git history. Publish them as GitHub release assets and enable the checked-in
1 MiB commit guard after cloning:

```sh
git config core.hooksPath .githooks
```

## Real-CUGAN 2× no-denoise Core ML

The 2× release converts the official MIT `up2x-latest-no-denoise.pth`
weight with the pinned `UpCunet2x` implementation. The converter replaces
PyTorch's negative-padding crop notation with equivalent tensor slices because
coremltools rejects negative padding, then verifies the rewritten graph against
the official forward path before checking Core ML parity.

### Contract

- Input: `input`, FP32 multi-array `[1, 3, 256, 256]` in RGB order.
- Output: `output`, FP32 multi-array `[1, 3, 512, 512]`.
- Scale: 2×.
- Official source commit: `2799af78ef105b414cc4b796c67c8511acdcdf6f`.
- Source SHA-256: `ef6c4e433bcac37b75ffba0a4044987ddd3ecfe7765a74a3c93887954e45562b`.
- Source-weight SHA-256: `f491f9ecf6964ead9f3a36bf03e83527f32c6a341b683f7378ac6c1e2a5f0d16`.

### Reproduce

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
mkdir -p .weights .release
curl -fL -o .weights/upcunet_v3.py \
  https://raw.githubusercontent.com/bilibili/ailab/2799af78ef105b414cc4b796c67c8511acdcdf6f/Real-CUGAN/upcunet_v3.py
curl -fL -o .weights/realcugan-updated_weights.zip \
  https://github.com/bilibili/ailab/releases/download/Real-CUGAN/updated_weights.zip
unzip -oj .weights/realcugan-updated_weights.zip \
  updated_weights/up2x-latest-no-denoise.pth -d .weights
.venv/bin/python scripts/export_realcugan_coreml.py \
  .weights/upcunet_v3.py \
  .weights/up2x-latest-no-denoise.pth \
  .release/NicoComicRealCUGAN2xNoDenoise-v0.1.0.mlpackage
```

The published FP32 conversion measured mean absolute error below `0.000001`
and maximum absolute error `0.000004`. FP16 was rejected because its mean and
maximum errors were `0.002985` and `0.026500`. The release ZIP digest is
recorded in [`checksums.txt`](checksums.txt).

## Real-ESRGAN Anime 4× Core ML

The super-resolution release converts the official BSD-3-Clause
`RealESRGAN_x4plus_anime_6B.pth` weights into a fixed-input FP16 Core ML
package. The RRDB inference graph follows BasicSR's Apache-2.0 architecture.

### Contract

- Input: `input`, FP16 multi-array `[1, 3, 512, 512]` in RGB order.
- Output: `output`, FP16 multi-array `[1, 3, 2048, 2048]`.
- Scale: 4×.
- Source-weight SHA-256: `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da`.

### Reproduce

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
mkdir -p .weights .release
curl -fL -o .weights/RealESRGAN_x4plus_anime_6B.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
.venv/bin/python scripts/export_realesrgan_anime_coreml.py \
  .weights/RealESRGAN_x4plus_anime_6B.pth \
  .release/NicoComicRealESRGANAnime4x-v0.1.0.mlpackage
mkdir -p .release/NicoComicRealESRGANAnime4x-v0.1.0
cp -R .release/NicoComicRealESRGANAnime4x-v0.1.0.mlpackage \
  .release/NicoComicRealESRGANAnime4x-v0.1.0/
cp -R LICENSES .release/NicoComicRealESRGANAnime4x-v0.1.0/
COPYFILE_DISABLE=1 zip -qry \
  .release/NicoComicRealESRGANAnime4x-v0.1.0.zip \
  .release/NicoComicRealESRGANAnime4x-v0.1.0
```

The converter rejects any other source-weight digest, loads all parameters
strictly, and compares a deterministic random prediction with PyTorch. The
published conversion measured mean absolute error `0.000882` and maximum
absolute error `0.002839`; `coremlcompiler` also compiled the final package.
The release ZIP digest is recorded in [`checksums.txt`](checksums.txt).

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

Build the private 60-panel/20-fallback audit set locally from NicoComic's manifest and `.nicoreading` annotations. The command verifies each comic hash and writes anonymous page IDs; keep the output outside this repository.

```bash
python3 scripts/prepare_private_panel_audit.py /path/to/ReaderEngine/manifest.json /tmp/nicocomic-private-panel-audit
```

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

The completed 416/320 training, Core ML timing, public-only distillation, and private replacement decision are recorded in [`reports/2026-09-14-yolox-nano-quality-gate.md`](reports/2026-09-14-yolox-nano-quality-gate.md). Both small inputs missed the private panel gate, so no lightweight weight is released and NicoComic keeps its current model descriptor.

For a free GPU run, open [`notebooks/train_yolox_nano_panels_colab.ipynb`](notebooks/train_yolox_nano_panels_colab.ipynb) in Google Colab and select a GPU runtime. The notebook downloads only the public archive, trains with the pinned official YOLOX source, and exports an ONNX checkpoint for Core ML conversion and private quality scoring.

### CC0 Comix v0 input for the 512 experiment

The next experiment expands the public training input with `emanuelevivoli/comix_v0_tiny_pages`: 6,750 CC0-1.0 pages at fixed revision `5347bee7a327ea794a281af68107c386596242c4`. Its Faster R-CNN panel boxes are pseudo labels, so they are training input only; the private 60-panel/20-fallback set remains the release gate.

```sh
.venv-training/bin/python scripts/download_comix_v0.py datasets/comix_v0_tiny_pages.json /tmp/nicocomic-comix-v0
.venv-training/bin/python scripts/prepare_comix_v0_coco.py datasets/comix_v0_tiny_pages.json /tmp/nicocomic-comix-v0-coco /tmp/nicocomic-comix-v0/*.tar
.venv-training/bin/python scripts/train_yolox_panels.py experiments/yolox_nano_panels_512.py /tmp/nicocomic-comix-v0-coco /tmp/nicocomic-yolox-512 --device cuda
```

On Apple silicon with an MPS-enabled PyTorch build, apply the compatibility patch to the pinned YOLOX checkout once, then pass `--device mps`; checkpoints and evaluation keep the same format:

```sh
python scripts/patch_yolox_mps.py YOLOX
.venv-training/bin/python scripts/train_yolox_panels.py experiments/yolox_nano_panels_512.py /tmp/nicocomic-comix-v0-coco /tmp/nicocomic-yolox-512 --device mps
```

The patcher is idempotent and fails if the pinned upstream source no longer matches. The verified M1 Max smoke run and its runtime boundary are recorded in [`reports/2026-09-14-apple-mps-smoke.md`](reports/2026-09-14-apple-mps-smoke.md).

The downloader and converter use Python's standard library plus Pillow already required by training. They verify every shard against the checked-in byte length and SHA-256, exclude the ambiguous `first-page` class, use story boxes as positives, retain cover/advertisement/text-story pages as whole-page negatives, and split by comic rather than page to prevent book leakage. The provenance and first-shard audit are in [`reports/2026-09-14-comix-v0-data-audit.md`](reports/2026-09-14-comix-v0-data-audit.md).

For the full free-GPU run, open [`notebooks/train_yolox_nano_comix_512_colab.ipynb`](notebooks/train_yolox_nano_comix_512_colab.ipynb). It downloads all verified shards, converts them, trains the fixed 512 model, retains checkpoints at the experiment evaluation interval for later private-gate selection, reports public pseudo-label validation, and exports ONNX without accessing private data. Keep 640 as the single fallback size if 512 misses the unchanged private quality gate.

## Data and privacy

This repository contains no Manga109-s images, private comics, private annotations, or audit screenshots. Obtain Manga109-s only through its official application process and comply with its terms. Do not use this model to redistribute or sell reproductions or derivatives of Manga109-s manga images.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for required attribution and citations.

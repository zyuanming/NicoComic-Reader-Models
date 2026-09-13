# RT-DETR input and simulator memory investigation

Date: 2026-09-14

This report contains no source pages, filenames, or private annotations. `PRIVATE-003` is an anonymous corpus ID.

## Question

Can the published 1280-square RT-DETR candidate reduce latency and memory by decoding a smaller source image, changing Core ML compute units, or exporting the same ONNX weights at 960 square?

## Evidence

NicoComic first varied the ImageIO decode long edge while leaving the model's fixed `1280 × 1280` input unchanged. The same independent 20-page truth set scored:

| Decode long edge | Exact pages | Adjacent-order error |
| ---: | ---: | ---: |
| 1280 | 19/20 (95%) | 0% |
| 1248 or 1184 | 18/20 (90%) | 2.82% |
| 1216, 1152, 1120, 1024, 960, or 768 | 17/20 (85%) | about 2.90% |

The smaller bitmap is enlarged back to the fixed model input, so this does not reduce model FLOPs and fails the current quality gate.

A production-reader UI probe then opened a real ZIP and advanced through 20 displayed pages at 0.5-second dwell on an arm64 iPhone 13 Pro Max, iOS 26.2 simulator. Host `ps` sampled the NicoComic process RSS while the released FP16 package ran through the real prefetch and cancellation path:

| Core ML compute units | Samples | P95 RSS | Peak RSS | UI duration |
| --- | ---: | ---: | ---: | ---: |
| CPU only | 165 | 707.2 MiB | 881.6 MiB | 54.978 s |
| All | 167 | 1472.0 MiB | 1579.8 MiB | 55.163 s |

Both UI runs passed. Heterogeneous compute had no speed benefit in this simulator and increased peak RSS by 698.2 MiB, so the App retains CPU only. Simulator RSS is not a physical-device `phys_footprint` result.

Finally, the audited ONNX was traced with a true `960 × 960` tensor and matching target-size input. The source graph contains static feature-grid tensors for its 1280 contract; tracing stopped at a 900-versus-1600 element shape mismatch. A smaller Core ML input therefore requires exporting from the original training implementation or training a new model.

## Decision

Keep the 1280 FP16 candidate unchanged. The next candidate is an Apache-2.0 YOLOX-Nano trained for a native `416 × 416` contract. It can replace RT-DETR only after the same private 60-panel/20-fallback scorer reaches at least 95% exact pages, less than 2% adjacent-order error, and 100% fallback, while simulator memory is materially lower. Physical iPhone and iPad evidence remains a separate release gate.

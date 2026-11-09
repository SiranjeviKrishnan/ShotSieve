# AI Models (all permissive, total ≈ 131 MB, peak RAM < 1.5 GB)

| Feature | Model | Runtime | Input | Output | License | URL |
|---|---|---|---|---|---|---|
| Face detection + 5 landmarks | YuNet 2023mar (232 KB) | OpenCV `cv2.FaceDetectorYN` | image scaled so long edge = 640 | bbox, 5 pts, score | MIT | https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx |
| Face identity embedding (same-person, face-dup) | SFace 2021dec (38.7 MB) | OpenCV `cv2.FaceRecognizerSF` | aligned 112×112 crop (alignCrop with YuNet landmarks) | 128-d | Apache-2.0 | https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx |
| Eyes closed / blink, smile, expression | MediaPipe FaceLandmarker float16 v1 (3.8 MB) with blendshapes | `mediapipe` Tasks API, IMAGE mode, num_faces=1 on each expanded face crop (crop = bbox × 1.6, resized ≤ 256) | face crop | 52 blendshapes (`eyeBlinkLeft/Right`, `mouthSmileLeft/Right`, …) + 478 landmarks | Apache-2.0 | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task |
| Scene embedding (duplicates, personal model features) | DINOv2-small **fp32** ONNX (88.5 MB, see "DINOv2 variant") | onnxruntime **CPU EP**, 6 intra-op threads, batch 8 | 224×224, ImageNet norm, NCHW float32 input | CLS token 384-d (`last_hidden_state[:,0]`) | Apache-2.0 | https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx |

Classic (non-NN, fast, deterministic) features:
- **Sharpness**: variance of Laplacian on luma resized to fixed 1024 long edge (scale-invariant) + Tenengrad;
  face-ROI sharpness on the face crop resized to 160 px. Normalized by a robust per-shoot percentile so
  thresholds adapt to each shoot (`score = clip(value / p90_of_shoot, 0, 1)`), plus absolute floor.
- **Exposure**: luma histogram → mean, % pixels ≤ 2 and ≥ 253, EV offset vs mid-grey.
- **Flash misfire**: EXIF Flash tag (bit 0 fired; mode bits) + mean luma vs median of the burst.

## Why these
- YuNet is ~1–3 ms/img on CPU; good for small faces in wide shots at 640 px.
- MediaPipe blendshapes give calibrated eye-blink values directly — no custom classifier needed; run
  only on detected faces (≤ 6 per image, largest first) to keep cost bounded.
- DINOv2 embeddings separate near-duplicates far better than pHash and also feed the personal model.
- SFace lets face-level duplicate logic tell "same people, same pose" from "same scene, different people".

## Registry contract (`src/shotsieve/models/registry.py`)
```python
@dataclass(frozen=True)
class ModelSpec: name: str; filename: str; url: str; sha256: str | None; size: int; license: str
MODEL_SPECS: dict[str, ModelSpec]   # keys: "yunet", "sface", "face_landmarker", "dinov2_small"
def models_dir() -> Path            # <project>/models  (override env SHOTSIEVE_MODELS_DIR)
def model_path(name) -> Path        # raises ModelMissingError with hint to run download script
def ensure_models(names=None, download=True) -> dict[str, Path]  # download with .part + rename, verify size, record sha256 in models/manifest.json
```
Sizes are known (232589, 38696353, 3758596, 44420939 bytes). sha256: compute on first download and
store in `models/manifest.json`; verify on later loads.

## Memory budget
YuNet ~10 MB, SFace ~80 MB, MediaPipe ~60 MB, DINOv2 fp32 ~350 MB + activations (batch 8 @224 ≈ 100 MB). Measured whole-process peak RSS: ~1.4 GB for 300 images (data/benchmark.md).
Image queue 64 × (1280×853×3) ≈ 210 MB. Total well under 3 GB.

## DINOv2 variant (measured on M1 Pro, 10 cores, ORT 1.31.1, 5 fixture images, 224x224)
Per-image latency in ms, `bs1 / bs8` (cosine = min cosine to the fp32 reference over 8 images):

| Variant | Provider / options | threads | bs1 | bs8 | cosine vs fp32 |
|---|---|---|---|---|---|
| **fp32 `model.onnx` (88.5 MB)** | **CPU** | 4 / 6 / 8 | 53 / 40 / 36 | 42 / 29 / 26 | 1.0000 |
| fp16 `model_fp16.onnx` | CPU | 4 / 6 / 8 | 57 / 45 / 45 | 43 / 30 / 26 | 0.9997 |
| int8 `model_quantized.onnx` | CPU | 4 / 6 / 8 | 31 / 25 / 24 | 20 / 15 / 12 | **0.914 (fails >= 0.98)** |
| q4f16 | CPU | 4 / 6 / 8 | 58 / 45 / 47 | 43 / 32 / 29 | 0.886 (fails) |
| fp32 | CoreML NeuralNetwork, ALL / CPUAndGPU / CPUAndNeuralEngine | 4 | 111 / 111 / 103 | 91 / 91 / 87 | 1.0000 |
| fp32 | CoreML MLProgram, ALL / CPUAndGPU / CPUAndNeuralEngine | 4 | 90 / 91 / 83 | 75 / 75 / 73 | 1.0000 |
| fp16 | CoreML NeuralNetwork (any units) | 4 | 57 | 42 | 0.9997 |
| fp16 | CoreML MLProgram (any units) | 4 | 96-104 | 85-87 | 0.9998 |

Other findings: graph optimisation level (ALL / EXTENDED / BASIC) changes little (<= 5 %, ALL best);
6 threads is the sweet spot when decode and face models run concurrently (8 threads is ~7 % faster in
isolation but slows the face worker; 10 threads is slower); batch 8 ~ batch 16 > batch 1 on CPU.

**Decision: fp32 on the CPU provider** (`ImageEmbedder` defaults `use_coreml=False`, threads 6,
batch 8; registry file `dinov2_small_fp32.onnx`). Rationale: CoreML partitions the ViT graph into many
CPU/ANE fragments and is 2-3x *slower* than CPU for every format and compute-unit setting; the fp16
model has no CPU fast path (it is upcast), so it is slower than fp32 and gives nothing but a smaller
file; the int8 model is the fastest (12-20 ms) but its embeddings drift to cosine 0.91 of fp32, which
violates the >= 0.98 quality bar and shrinks duplicate separation. fp32 is the reference (cosine 1.0), so
the duplicate thresholds (0.12) keep their meaning, and the full pipeline is embedding-bound at ~21 img/s.
CoreML remains selectable via `Settings.use_coreml=True` (falls back to CPU on error).

## Phase 2 search models: SigLIP2-B/16 (T10, measured 2026-11-07)
Files (group `search`, `shotsieve models download --extra search`): `siglip2_vision_fp32.onnx` 371.8 MB,
`siglip2_text_int8.onnx` 283.4 MB (lazy, query time only), `siglip2_tokenizer.json` 34.4 MB = 689.6 MB
(<= 720 MB budget). The int8 vision alternative lives in its own group `search-int8` (94.6 MB, gate only).

**int8 vision gate** (`scripts/benchmark.py --siglip-gate`; 300 bench images x 30 fixed queries, fp32 vs int8
vision, same int8 text tower, CPU EP, 4 threads, bs 8):

| Metric | Value | Gate |
|---|---|---|
| Mean top-20 overlap | **0.365** | >= 0.90 |
| Median image cosine (int8 vs fp32) | **0.8331** (min 0.5923) | >= 0.98 |
| Embed speed fp32 / int8 | 8.9 / 23.4 img/s | info |

**Decision: fp32 vision** (`Settings.siglip_vision = "fp32"`). The dynamic-quantised ViT drifts far from fp32,
as DINOv2 int8 did (cos 0.91). fp32 still embeds at 8.3-8.9 img/s (>= 5 budget). Text tower: the shipped int8 text
model is used for queries on both variants (ranking sanity-checked in `tests/test_models_real.py`, recall@10 = 1.00
on `tests/fixtures/search_queries.json`).

## Phase 2 OCR models: PaddleOCR on ONNX (T11, 2026-11-07)
Group `ocr` (`shotsieve models download --extra ocr`), pinned to HF `monkt/paddleocr-onnx@7b02d0a` (Apache-2.0):
`ppocr_det_v3.onnx` 2.4 MB (PP-OCRv3 DB detector), `ppocr_rec_en.onnx` 7.8 MB (PP-OCRv5 en recogniser, CTC, 438 classes =
blank + 436-char dict + space), `ppocr_en_dict.txt`. CPU provider, 2 intra-op threads; 10.2 MB total.


## T18 pinning (2026-11-08)
Every `ModelSpec` carries its sha256 (`PINNED_SHA256` in `models/registry.py`); a downloaded or copied file
with the wrong size or hash is deleted ("download corrupted, run the install again"). The opencv_zoo URLs
are pinned to commit `48b10fa505aeb51fbd127fc60a5c660040e823b7` (was `raw/main`). Verified 2026-11-08 with
`curl -sIL`: both URLs 302 to the LFS host then **HTTP 200**, `content-length` 232589 (YuNet) and 38696353 (SFace);
the downloaded bytes hash to the pinned sha256 (`8f2383e4...52fa4`, `0ba9fbfa...e79`). The HF and MediaPipe
URLs were already commit/version pinned. `SHOTSIEVE_MODELS_SRC=/dir` copies from a local folder (same checks).

## T22 crop models (2026-11-08)
Group `crop` (`shotsieve models download --extra crop`; optional, the wizard lists it as "Smart crop: 36 MB"):
`yolox_s_2022nov.onnx` 35,858,002 B, **Apache-2.0** (Megvii YOLOX; opencv_zoo `models/object_detection_yolox` at the same pinned commit
`48b10fa5...`, sha256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`, verified by download). COCO-80 classes
(annotations CC BY 4.0), same posture as YuNet / DINOv2. Input `images` `[1,3,640,640]` float32 BGR 0-255 (letterbox, pad 114, top-left),
output `[1,8400,85]`; decode, NMS and the letterbox undo follow opencv_zoo `yolox.py` (read for fidelity, not run): `models/object_detect.py`.
CPU provider, 4 threads, arena off, lazy, idle unload 120 s (`models/idle.py`).
No other new weights: saliency and point prompts use DINOv2-small **patch tokens** from the already installed model, in a session of its own at
dynamic size (long edge 448, multiples of 14): `models/patch_features.py`. Crop works without YOLOX (faces + saliency + point prompts).

## T23f denoise model (view only, 2026-11-09)
Optional group `denoise` (`shotsieve models download --extra denoise`; Setup assistant / the loupe's strength menu offer it, 124 MB):
**NAFNet-SIDD-width32** (Chen, Chu, Zhang, Sun, ECCV 2022, **MIT**; trained on SIDD real sRGB noise), as ONNX `nafnet_sidd_width32_fp16.onnx`
123,972,868 B, sha256 `a9ca266868fad47ce1d19ae2677809e8e69a3c131afbe54ba24c431fe0dc069a` (verified by download), from
`mlboydaisuke/NAFNet-SIDD-width32-OpenVINO` at commit `ccea65867e9e54460335054c81eea25524edac4d` (MIT; tf2onnx 1.17.0 conversion of
`litert-community/NAFNet-SIDD-width32-LiteRT` rev `59984b52...`, itself the official weights). Input `[1,3,256,256]` float32 RGB 0..1, output the same.
No export of our own was needed. The published graph is not loadable by the CoreML provider as is (every `Conv` has an empty `group`,
the 1x1 / 2x2 ones no `pads`); `models/onnx_conv_fix.py` rewrites those attributes (protobuf wire format, no `onnx` dependency, bit-identical
on the CPU provider, streamed from a memory map into `models/nafnet_sidd_width32_coreml.onnx` once). Measured on an M1 Pro 32 GB, ORT 1.31.1,
one 256x256 tile: CPU 270 ms; CoreML MLProgram CPUAndNeuralEngine 134 ms, CPUAndGPU / ALL 49-50 ms, ALL with 3 tiles in flight 35 ms (**1.3 MP/s**
steady); CoreML vs CPU max abs difference 0.003 (fp16). `models/denoise.py`: 256 px tiles, 16 px feathered overlap, reflect padding, band-wise
accumulation (a 24 MP image needs ~35 MB of accumulators), CoreML `ALL`, CPU arena off (saves ~230 MB with 3 tiles in flight), idle unload 60 s
through `models/idle.py`. Strength = residual scaling of the output (`x + a (y - x)`), `a` = 0.45 / 0.75 / 1.0 for Low / Medium / High times
`clip(sigma / 5, 0.25, 1)` (`sigma` = median Immerkaer noise estimate, kept within 0.5-2.5x of what the EXIF ISO predicts), so a clean photo is barely touched.
Fallback without the model: `cv2.fastNlMeansDenoisingColored`, `h = clip(k sigma, 1.5, 18)`, k = 0.7 / 1.0 / 1.4, same tiling.
Quality on real photographs + Gaussian sigma 8 / 15 / 25 (a 1600x1200 crop): PSNR +5.7 / +6.8 / +7.9 dB at Medium (AI), +3.4 / +5.3 / +7.2 dB (fallback);
edge contrast on sharp photographs >= 93 % at Medium (AI), a clean photograph keeps 72-86 % of its Laplacian variance (tests/test_denoise_api.py).
Memory (20 MP image, M1 Pro): +438 MB peak while denoising (session ~+390 MB incl. CoreML compile, arrays), **+205 MB retained after the idle unload**
(macOS keeps the freed ORT / CoreML heap, the plateau described in `models/idle.py`; a worker process would return it, not built).

## T23h recognition-only face scan (2026-11-09; no new model)
YuNet + SFace are unchanged and the cull still calls `FaceDetector.detect` (one 640 px pass, score >= 0.7, at most 6 faces): decisions use those "decision faces" only. A second `FaceDetector` instance (`get_rec_face_detector`, score >= 0.3) runs in the background scan (`pipeline/facescan.py`) over: the whole image at 640 and 1280 px long edge, native-scale 1280 px tiles (15 % overlap) and x2 tiles of 640 px source windows, on a working image of up to 4096 px. Candidates are merged by NMS (IoU 0.3, 80 % containment), then accepted when score >= 0.6 (box side < 24 px), >= 0.8 (otherwise) and the loose landmark geometry holds. The MediaPipe landmarker was measured as a second check and rejected (it succeeds on only 50-85 % of true faces, so it cost more recall than it removed false positives). Faces under 24 px are embedded after a mild blur + Lanczos x(112/side) of their neighbourhood (`FaceEmbedder.embed_small`: cosine to the same face at full size 0.377 vs 0.358 plain; no gain above 24 px, so it is not used there). Memory: the detector is 230 KB; the scan holds one 4096 px RGB image (<= 34 MB) and is RSS-budgeted (700 MB above its start, then frees pages). Numbers: TASKS T23h.

## Release bundle (T24)
`ShotSieve.app` carries every model in `Contents/Resources/models/` (read-only) and never downloads one.
`scripts/bundle_models.py` verifies each file against the registry sha256, drops `siglip2_vision_int8`
(rejected alternative) and ships `nafnet_sidd` as the CoreML-repaired graph; `bundle.json` records the shipped
filename, size and sha256 per model and marks the folder as a bundle (`registry.bundled()`).
Fetch the models for a build with `uv run python scripts/fetch_models.py`. Licences and citations:
`THIRD_PARTY_NOTICES.md`.

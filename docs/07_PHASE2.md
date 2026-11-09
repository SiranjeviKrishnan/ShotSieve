# Phase 2 spec (architect decision, 2026-11-07)

Build in this order. Each section is a task in TASKS.md (T6…T13). New files follow the 01_ARCHITECTURE layout and
add no top-level dirs. Rules from 01/03 still hold: decode once, lazy model singletons, every module has unit tests,
ruff-clean, network-free tests except `@pytest.mark.models`.

| # | Task | Why this slot | Inline in cull? | New model |
|---|---|---|---|---|
| T6 | N5 calibration report fix | open minor from Review #3 | — | — |
| T7 | AF-point focus check | biggest cull-quality win, ~1 ms/img | yes (default on) | — |
| T8 | E-shutter banding | cheap classic CV, new reject reason | yes (default on) | — |
| T9 | Multi-camera clock offset | fixes ordering for multi-shooter jobs; reuses DINOv2/SFace | post-group step | — |
| T10 | Background indexer + NL/people search | most-requested; heavy → background | **no** (lazy) | SigLIP2-B/16 |
| T11 | Jersey OCR + roster | reuses T10 indexer; niche (adult sports) | **no** (opt-in) | PP-OCR det+rec |
| T12 | Video frame-grab culling | reuses Stage A/B; opt-in | **no** (opt-in) | — |
| T13 | Editing profile | **DEFERRED to Phase 3**: editing, not culling; needs ≥200 edited frames; LR ignores JPEG/DNG sidecars, so few users would see the result | — | — |

## Budgets (acceptance for every task; `scripts/benchmark.py` gains `--phase2`)
- **Cull throughput ≥ 10 img/s** (target ≥ 18) on the 300 bench with T7+T8+T9 on. Indexer is paused while a cull runs.
- **Peak RSS ≤ 3 GB** for the whole process: cull (~0.95 GB) + indexer running SigLIP vision (+~0.6 GB) + text tower
  loaded for a query (+~0.45 GB) + OCR (+~0.1 GB) ≈ 2.1 GB measured target. Text tower unloads after 10 min idle.
- **Added model size ≤ 720 MB** (fp32 vision) or ≤ 420 MB if the int8-vision gate in T10 passes. Phase 2 models are
  an *extra* group: `shotsieve models download --extra search,ocr`; core download is unchanged (131 MB).
- Indexer: ≥ 5 img/s SigLIP embedding at 4 threads in the background; UI p95 latency < 150 ms while it runs.
- Search query latency: < 300 ms warm (text encode + kNN over 20k), < 3 s cold (text tower load).

## Phase 2 models (all verified 2026-11-07 with `curl -sIL`: HTTP 200, sizes below exact)
Pin to commit, not `main`: SigLIP2 repo `ba1f3b0843f24bc5417d38e19c37b287d719b2f4`, PaddleOCR repo
`7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65`. Base: `https://huggingface.co/<repo>/resolve/<sha>/<file>`.

| Registry key | Repo / file | Bytes | License | Runtime | Input → Output |
|---|---|---|---|---|---|
| `siglip2_vision` | onnx-community/siglip2-base-patch16-224-ONNX `onnx/vision_model.onnx` | 371807752 | Apache-2.0 (google/siglip2-base-patch16-224) | ORT CPU EP, 4 thr, bs 8 | 224×224 RGB squash-resize (bilinear), (x/255−0.5)/0.5, NCHW f32 → `pooler_output` 768-d, L2-norm |
| `siglip2_vision_int8` (alt) | same repo `onnx/vision_model_quantized.onnx` | 94553333 | Apache-2.0 | ORT CPU | same; used only if gate passes |
| `siglip2_text` | same repo `onnx/text_model_quantized.onnx` | 283438275 | Apache-2.0 | ORT CPU, lazy, query-time only | `input_ids` [1,64] (lowercase, EOS appended, pad to 64 with `<pad>`) → `pooler_output` 768-d, L2-norm |
| `siglip2_tokenizer` | same repo `tokenizer.json` | 34363039 | Apache-2.0 | `tokenizers` (Apache-2.0) | text → ids |
| `ppocr_det` | monkt/paddleocr-onnx `detection/v3/det.onnx` (PP-OCRv3 mobile det) | 2429873 | Apache-2.0 | ORT CPU | BGR, long edge 960 (÷32), ImageNet norm → prob map H×W |
| `ppocr_rec` | monkt/paddleocr-onnx `languages/english/rec.onnx` (PP-OCRv5 mobile en rec) | 7830888 | Apache-2.0 | ORT CPU | crop H=48 (read from model input), W dyn, (x/255−0.5)/0.5 → CTC logits [T, C] |
| `ppocr_dict` | monkt/paddleocr-onnx `languages/english/dict.txt` | 1416 | Apache-2.0 | — | CTC classes (index 0 = blank) |

Rejected: **MobileCLIP / MobileCLIP2** (`apple-amlr`: "exclusively for Research Purposes… does not include… use in any
commercial product"); **SigLIP v1 text fp32** (441 MB) / **SigLIP2 text fp32** (1.13 GB, 256k vocab); OpenAI CLIP B/32
(MIT, but weaker retrieval and a bigger text tower than needed); PP-OCRv5 server det (88 MB, no gain on big jersey digits);
PARSeq (Apache-2.0 but no maintained ONNX release). Fallback if SigLIP2 must be swapped: Xenova/siglip-base-patch16-224
(Apache-2.0, sha `4649052661e53c7000355844105f8a1792088239`; vision 371819850, text_quantized 111475220, tokenizer.json 2398744).
License caveats: the SigLIP2 tokenizer is Gemma's SentencePiece vocab, shipped by Google under Apache-2.0 in the SigLIP2
repo (no Gemma ToU attached). Record that in README "Third-party models". The ONNX conversions are community builds of
Apache-2.0 weights, so verify by sha256 recorded on first download (existing manifest flow).
New pip deps: `tokenizers` (Apache-2.0), `sqlite-vec` 0.1.9 (MIT/Apache-2.0). Optional external: `exiftool` (not bundled).

Registry change: `ModelSpec` gains `group: str = "core"` (`"search"`, `"ocr"`); `ensure_models(names=None)` = core only;
`ensure_models(groups=("search",))` for extras. A missing extra model disables that feature with a hint and no error.

---
## T6 — N5: repeat `audit/complete` must not clobber calibration (stream: product)
Files: `personal/calibrate.py`, `tests/test_review_fixes.py`.
- Build audit samples from `meta.audit_sample` (frozen) joined with `overrides`, not from `decisions.audit`.
- If the join gives 0 audited samples: return the **persisted** thresholds with `applied=False`, and do not write `meta.calibration`.
- Test `test_N5_repeat_audit_complete_idempotent`: cull → audit all → complete (report R1) → complete again → report == R1,
  `meta.calibration` unchanged, thresholds unchanged.

## T7 — AF-point focus check (stream: engine)
Files: `io/makernotes.py` (new), `features/af_focus.py` (new), `features/vector.py`, `scoring/quality.py`,
`scoring/reasons.py`, `io/exif.py` (`ImageMeta.af_points`), `tests/test_makernotes.py`, `tests/test_af_focus.py`.
**Extraction** `read_af_points(path, meta) -> list[AFPoint(x, y, w, h)]` (normalised 0–1 in *sensor* orientation; empty = unknown):
- Canon (CR2/JPEG: MakerNote IFD in EXIF; CR3: TIFF inside the `CMT3` box, located by scanning the first 512 KB for `b"CMT3"`):
  tag `0x0026` AFInfo2 → NumAFPoints, AFImageW/H, AFAreaW[], H[], X[] and Y[] (center-origin, Y up), AFPointsInFocus bitmask.
  Return the in-focus points (or the selected points if the mask is empty).
- Sony (ARW/JPEG: MakerNote starts with `b"SONY DSC \0\0\0"`, IFD at +12): tag `0x2027` FocusLocation (int16 ×4: W, H, X, Y,
  top-left origin) → one point, box = 6 % of the short edge. Fall back to `0x201e`/`0x2020` only if 0x2027 is missing.
- Other makes, or a parse failure → `[]`. If `exiftool` is on PATH and `Settings.af_use_exiftool`, use one stay_open process
  for files the pure parser couldn't read. Parse cost ≤ 0.5 ms/img and no full-file reads (seek only).
**Features** (append to `FEATURE_NAMES`, bump `SCHEMA_VERSION` to 2; `PersonalModel` stores its schema and slices
`to_array()[:n_features]` so v1 models keep working until retrained):
- `af_known` (0/1); `af_sharp_norm`: Laplacian variance in the union AF ROI (each box padded to ≥ 64 px at preview scale,
  orientation-mapped), normalised by the shoot p90 of the same measure; `af_peak_ratio`: max sharpness over an 8×8 tile
  grid ÷ AF-ROI sharpness; `af_on_face`: the AF ROI overlaps an important face (IoU ≥ 0.1 or center inside).
**Rule** (`af_missed`, HARD tier, never CERTAIN; it only applies when `af_known` and the AF point is not on a face, because face
sharpness already covers that case): `af_sharp_norm < 0.35` and `af_peak_ratio ≥ 2.5`, with the peak tile not adjacent to the
ROI. Reason text: "Focus missed the AF point (sharpest area is elsewhere)". Moment safeguard unchanged (best of group → review).
**Tests**: synthetic TIFF/JPEG with hand-built Canon and Sony maker-note IFDs (bytes in `make_fixtures.py`), plus orientation 6/8 mapping;
a fixture image sharp at the AF box → no flag; the same image with the AF box blurred (σ=4) and the rest sharp → `af_missed`.
**Accept**: on bench, 0 clean frames flagged; throughput drop ≤ 3 %; the real Canon and Sony sample files in
`tests/fixtures/real/` (if present, `@pytest.mark.network`) parse to a point inside the frame.

## T8 — E-shutter banding (stream: engine)
Files: `features/banding.py` (new), `vector.py` (`banding_amp_ev`, `banding_coherence`), `quality.py`, `reasons.py`, `tests/test_banding.py`.
**Algorithm** (on luma of the preview, rotated back to sensor orientation so readout bands are rows; 1–2 ms):
1. Skip if `exposure_time ≥ 1/50` (flicker integrates) → features NaN.
2. Split into 8 vertical strips. For each strip, take the row-mean of log2(luma+1) over rows `p(y)` and detrend:
   `d(y) = p(y) − gaussian(p, σ = H/25)`. Mask rows whose strip is >30 % clipped.
3. `coherence` = mean pairwise Pearson r of the strips' `d`. `amp_ev` = robust std (1.4826·MAD) of the strip-median `d`.
4. Periodicity: FFT of the strip-median `d`; `peak_frac` = the power at the peak with period in [H/60, H/3] ÷ the total power in that band.
**Flag** `banding`: `coherence ≥ 0.6 ∧ peak_frac ≥ 0.25 ∧ amp_ev ≥ 0.04` → HARD-tier reason (review). Severe
`amp_ev ≥ 0.12` → reject only if the group has a non-banded alternative; otherwise review. Text: "Light-flicker banding (electronic shutter)".
**Tests**: clean fixture × (1 + a·sin(2πy/P)) for a ∈ {0.03, 0.05, 0.1}, P ∈ {20, 60, 150} px → flags a ≥ 0.05 (a = 0.03 not
flagged); stripes on the left half only (incoherent) → no flag; a horizontal-blinds natural texture → no flag; rotated (orient 6) → still flags.
**Accept**: synthetic recall ≥ 95 % for a ≥ 0.05; bench false-flag ≤ 1 % (≤ 3/300) and 0 clean frames rejected; ≤ 2 ms/img.

## T9 — Multi-camera clock offset + suggested pair (stream: engine → product for UI)
Files: `grouping/clock.py` (new), `grouping/bursts.py` (apply offsets), `pipeline/runner.py` (post-embed step),
`pipeline/cache.py` (`meta.clock_offsets` JSON {serial: seconds, source: auto|manual|exif}), `io/xmp.py` (optional
`photoshop:DateCreated`), server, CLI, UI, `tests/test_clock.py`.
**Algorithm** `suggest_offsets(images, emb, face_emb) -> list[OffsetSuggestion(serial, offset_s, confidence, pair=(ref_path, path, cos))]`:
- Only when ≥ 2 serials with `ts_source == "exif"`. Reference = serial with the most frames. Exclude mtime timestamps.
- Prior: if both cameras have EXIF `OffsetTimeOriginal`, the difference of those offsets is a candidate (a time-zone mismatch).
- Candidates: mutual nearest neighbours across the two cameras by DINOv2 cosine, computed in chunks of 1024 rows (no n×m matrix),
  keep `cos ≥ 0.65`; `+0.05` bonus when the face sets match (≥ 1 SFace pair sim ≥ 0.40). `Δ = t_ref − t_cam`.
- Mode: histogram of Δ, 2 s bins, range ±26 h. Require support ≥ max(3, 20 % of candidates). Offset = median of Δ in
  mode ± 3 s. Confidence = support ÷ candidates. Suggested pair = the highest-cosine candidate inside the mode.
- No mode passes → return the top-5 candidate pairs (no offset) for the user to pick from.
- Apply: `t_adj = t + offset[serial]`. Bursts stay per serial (other viewpoints are not duplicates); the UI timeline,
  `/api/shoot` order and the CLI listing sort by `t_adj`. Never applied silently: `auto` suggestions need confirmation
  (`--apply` or the UI) unless `confidence ≥ 0.6 ∧ support ≥ 10`.
**API/CLI/UI**: `GET /api/clock` → suggestions + current offsets; `POST /api/clock {serial, offset_s | pair:[id_a,id_b]}`;
`shotsieve clock <folder> [--apply] [--set SERIAL=+3725] [--write-xmp-time]`. UI panel "Cameras": suggested pair side by side,
Accept / pick another pair / type an offset (±h:mm:ss).
**Tests**: copy fixture bursts into cam B (piexif new serial, ts +3725 s ± 0.3 s jitter) → |est − 3725| ≤ 1 s, confidence ≥ 0.5;
unrelated scenes → no offset; a +1 h zone prior is honoured; 10k×10k synthetic embeddings stay under 200 MB RSS delta and take < 2 s.
**Accept**: cull throughput unchanged within 3 %; offsets persisted and survive re-runs; manual overrides win over auto.

## T10 — Background indexer + natural-language & people search (stream: search)
Files: `models/text_image.py` (new: `SiglipVision`, `SiglipText`, lazy, unload-after-idle), `pipeline/indexer.py` (new),
`pipeline/search.py` (new; no new `search/` package) + `pipeline/people.py` (new), `pipeline/cache.py` (tables), `models/registry.py`,
server, CLI, UI, `tests/test_indexer.py`, `tests/test_search.py`, `tests/test_people.py`, `tests/test_models_real.py` (+siglip).
**Indexer** (`Indexer(shoot_dir, settings)`; a single daemon thread, low priority):
- Task queue per shoot: `siglip` (all images), `people` (after siglip), `ocr` (T11, opt-in). Resumable: work = rows with
  no `embeddings(kind='siglip2')` for the current `fp`. Reuses the cached preview decode (≤ 1280) → 224 squash.
- ORT 4 intra-op threads, bs 8. **Pauses** while a cull job runs (`server/app.py` job lock; it checks between batches),
  and resumes after. Auto-starts after a cull when `Settings.search_index_auto = True` (the default when the models exist).
- Progress via `GET /api/index/status` {task, done, total, paused, img_s}.
**Storage** (cull.db): `embeddings(kind='siglip2', dim=768)` as the source of truth; `vec_siglip USING vec0(id INTEGER PRIMARY KEY,
emb float[768] distance_metric=cosine)` keyed by `image_ids.id`, loaded with `sqlite_vec.load(conn)`. If loading the extension
fails (Python built without `enable_load_extension`), fall back to a numpy brute-force matmul (20k×768 f32 = 61 MB, < 20 ms).
**int8 gate** (first subtask, `scripts/benchmark.py --siglip-gate`): on bench + 30 fixed queries, int8 vision vs fp32:
mean top-20 overlap ≥ 0.90 **and** median cosine ≥ 0.98 → ship int8 (`Settings.siglip_vision = "int8"`), else fp32. Record the result in 02_MODELS.md.
**Query** `search(shoot, q, k=100) -> [Hit(id, score, why)]`: parse filters `person:"Name"`, `jersey:23`, `is:keep|review|reject`,
`camera:SERIAL`; the rest is the text → tokenizer (lowercase, EOS, pad to 64) → text tower → cosine kNN. Results: keep
`cos ≥ top1 − 0.08`, at most k. An empty text with filters only → filter results in time order. Query embeddings are
cached in an LRU (256).
**People** (gated by `Settings.people_consent=False`; T10 must ship the toggle first, per 06 legal note):
- Faces: the existing SFace 128-d (`embeddings kind='face'`), important faces only, det score ≥ 0.8, face ≥ 48 px.
- Cluster: agglomerative, cosine, average linkage, merge while sim ≥ 0.40 (SFace same-ID threshold 0.363 + margin);
  clusters with < 3 faces → "unsorted". Incremental: a new face joins a cluster if its sim to the centroid ≥ 0.45, otherwise it is
  re-clustered at the end of the task.
- Tables: `people(id, name, cover_face, n)`, `person_faces(person_id, path, face_idx, sim)`. Names are user-given only.
- `DELETE /api/people` wipes both tables + `meta.people_consent=false`. Consent off → the people task is skipped and the endpoints return 403.
**API**: `GET /api/search?q=&k=`, `GET /api/index/status`, `POST /api/index/{start|pause}`, `GET /api/people`,
`POST /api/people/{id}` {name | merge_into}, `POST /api/people/consent` {on}, `DELETE /api/people`.
**CLI**: `shotsieve index <folder> [--no-people]`, `shotsieve search <folder> "bride laughing at sunset" [--k 50] [--json]`.
**UI**: `/` focuses the search box; results replace the grid (Esc restores it); a people strip with a rename field; an indexer status pill.
**Tests**: tokenizer pad/EOS shape; vec0 vs numpy give the same top-k; the indexer resumes after a kill; pause during a cull (mock job);
consent off → no people rows; models test: "a dog" ranks the dog fixture top-3 among the fixtures, and "a red car" picks the red-car fixture.
**Accept**: budgets above; cull throughput with the indexer auto-started still ≥ 10 img/s (it pauses); recall@10 ≥ 0.8 on a 30-query
hand-labelled bench file `tests/fixtures/search_queries.json`; peak RSS ≤ 3 GB with the indexer + a query running.

## T11 — Jersey OCR + roster (stream: ocr; depends T10 indexer)
Files: `models/ocr.py` (new: `TextDetector` DB post-process, `TextRecognizer` CTC), `features/jersey.py` (new),
`pipeline/indexer.py` (`ocr` task), `io/roster.py` (new CSV), `io/xmp.py` (`dc:subject` keywords),
server, CLI, UI, `tests/test_ocr.py`, `tests/test_jersey.py`.
**Pipeline** (opt-in: `Settings.jersey_ocr=False`; it runs in the indexer, so it never touches cull throughput):
1. Det on the preview at a 960 long edge; DB: binarize 0.3, box score ≥ 0.6, unclip 1.6, min box height ≥ 2.5 % of the image height.
2. Up to 12 boxes by area → a perspective crop → H=48 → batch rec. CTC greedy decode; then the regex `^\d{1,3}$`; per-char min prob ≥ 0.80,
   mean ≥ 0.90. Reject boxes with aspect ratio > 4 (scoreboards/ads) and digits inside detected-face boxes.
3. Burst vote: a number seen in ≥ 30 % of a burst's frames (≥ 2 frames) is propagated to the burst's other frames with `source=burst`.
4. Roster (`number,name[,team]` CSV; `POST /api/roster`, `shotsieve roster <folder> roster.csv`): a match tags the name; if a roster is loaded,
   numbers not on it are kept but marked `unmatched` (not written to XMP).
Storage: `jersey(path, number, conf, x, y, w, h, source)`. XMP: `dc:subject` += "#23", "Jane Doe" (merged with existing keywords, never removed).
Search: `jersey:23` and `person:"Jane Doe"` (roster names resolve to numbers).
**Tests**: rendered digits on synthetic shirts (cv2.putText, rotated ±15°, 3–10 % height) → exact read; a scoreboard with a long box → rejected;
roster CSV with BOM/whitespace; burst propagation; the XMP keyword merge is idempotent.
**Accept**: on a user-labelled set of ≥ 100 sports frames: tag precision ≥ 0.95 (roster-matched), recall ≥ 0.70 for numbers ≥ 3 % of the frame
height; ≥ 8 img/s in the indexer; +≤ 120 MB RSS.

## T12 — Video frame-grab culling (stream: engine; depends T7/T8 features)
Files: `io/video.py` (new), `pipeline/video.py` (new), `io/formats.py` (`VIDEO_EXTS = .mp4 .mov .m4v`), cache, server, CLI, UI, `tests/test_video.py`.
**Decode**: `cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)` (hardware decode on macOS), with `CAP_FFMPEG` as the fallback; seek by `CAP_PROP_POS_MSEC`.
Never hold more than 2 frames; downscale to 1280 at once.
**Algorithm** (opt-in `cull --video` / `Settings.video=False`; runs after the photo cull, in the same job):
1. Coarse pass at 2 fps (`Settings.video_fps`): Stage A (sharpness, exposure, banding) + Stage B (faces/eyes) per sampled frame.
2. Scene cuts: HSV histogram χ² distance > 0.5 between samples → a new segment.
3. In each segment: frame score = the existing rule prior; local maxima with NMS over a 1.0 s window; keep the top `max(1, 6/min)` per segment.
   Refine each pick ±0.25 s at 10 fps and take the sharpest.
4. Candidates are virtual images `path#t=12.345` in `images`/`decisions` (the grouping key = video + segment), with decisions and reasons as for photos,
   and `burst_id` = segment. Thumbs come from the cached frame.
5. Export (UI `g` key / `POST /api/video/grab {id}` / `cull --video --export-keeps`): a full-res decode at t → `<shoot>/frame_grabs/<stem>_t012345.jpg`
   (q 95, EXIF DateTimeOriginal = creation + t). Exported JPEGs are normal images on the next cull. Originals are never modified.
**Tests**: a synthetic mp4 (cv2.VideoWriter, mp4v) with 3 scenes, blurred and sharp spans, and a closed-eye span → picks fall in the sharp,
open-eye spans of each scene; seek accuracy ± 1 frame; export file EXIF time.
**Accept**: ≥ 4× realtime on a 4K30 H.264/HEVC clip (M1 Pro); RSS +≤ 300 MB; photo-cull throughput unaffected when `--video` is off.

## T13 — Local editing profile (DEFERRED, Phase 3; spec kept so it is not re-derived)
`personal/edit_profile.py`: targets `crs:Exposure2012, Contrast2012, Highlights2012, Shadows2012, Whites2012, Blacks2012,
Temperature, Tint, Vibrance, Saturation`, read from lrcat `Adobe_imageDevelopSettings.text` and XMP `crs:`. Features: exposure stats,
gray-world RGB, Lab a/b means, ISO, flash, and DINOv2 PCA-16. Model: per-slider HistGB, shrunk to the user median with w = n/(n+100).
Needs ≥ 200 edited frames. Write `crs:` to RAW sidecars only, and only when `crs:HasSettings` is absent. Accept: held-out MAE ≤ 0.6 × the
median-baseline MAE for Exposure/Temp.

# Pipeline Spec

## 1. Scan & metadata
- `formats.py`: RAW = {.cr2,.cr3,.nef,.nrw,.arw,.srf,.sr2,.raf,.orf,.rw2,.pef,.dng,.srw,.3fr,.iiq,.erf}; RASTER = {.jpg,.jpeg,.png,.tif,.tiff,.heic,.heif,.webp}.
  If RAW+JPEG pair with same stem exists, treat as one image (prefer JPEG as preview source, RAW as the "master" path for XMP).
- `exif.read_meta(path) -> ImageMeta`: `timestamp` (DateTimeOriginal + SubSecTimeOriginal → float epoch; fallback file mtime, flag `ts_source`),
  `camera_serial` (BodySerialNumber / SerialNumber / InternalSerialNumber; fallback "make-model"), `make`, `model`,
  `flash_fired: bool|None`, `iso`, `exposure_time`, `fnumber`, `focal_length`, `orientation`, `width`, `height`.
  Use `exifread` for TIFF-based raws and JPEG; Pillow for PNG/HEIC(if pillow-heif present); optional `exiftool` subprocess if on PATH (CR3).
  Never raise on bad metadata — return defaults.

## 2. Preview
`load_preview(path, max_edge=1280) -> np.ndarray (H,W,3) uint8 RGB`, orientation applied.
- RAW: `rawpy.imread(path).extract_thumb()` → JPEG bytes → decode with `cv2.imdecode` (fast). If thumb is
  bitmap, use directly. If no thumb or thumb < 640 px long edge: `postprocess(half_size=True, use_camera_wb=True, output_bps=8)`.
- Raster: `cv2.imread` with `IMREAD_REDUCED_COLOR_2/4` when source is big (≥ 2× max_edge) for speed; HEIC via pillow-heif if installed.
- Downscale with `cv2.INTER_AREA`.

## 3. Features (per image) — `FeatureVector`
Fixed-order numeric vector (document order in `vector.py`; missing → NaN, HistGradientBoosting handles NaN):
```
sharp_global, sharp_global_norm, tenengrad_norm,
exp_mean, exp_clip_hi, exp_clip_lo, exp_ev,
n_faces, main_face_area, main_face_centrality,
face_sharp_main, face_sharp_min_important, face_sharp_norm_main,
eyes_closed_main, eyes_closed_max_important, n_eyes_closed_important,
smile_main, face_det_score_main,
flash_fired, flash_lum_delta_ev,
burst_size, dup_group_size, rank_in_group_norm,
iso_log2, shutter_log2
```
- "Important faces": faces with area ≥ 25% of the largest face area AND area ≥ 0.3% of the image. Max 6.
- Eyes closed per face: `max(eyeBlinkLeft, eyeBlinkRight)` → closed if > 0.55 (both > 0.45 → closed). Store raw values.
  If MediaPipe finds no landmarks in the crop, eye values are NaN (not "open").
- Store the 384-d DINOv2 embedding and face embeddings in the cache (BLOB float16), not in FeatureVector.
  The personal model uses FeatureVector + PCA-16 of the DINOv2 embedding (fit on training set).

## 4. Grouping
- **Bursts** (`bursts.py`): sort by (camera_serial, timestamp). New burst when gap > `burst_gap_s` (default 2.0 s).
  Images with `ts_source == "mtime"` get their own burst unless embedding-similar.
- **Duplicate groups** (`duplicates.py`): within a window of consecutive images per camera
  (bursts plus neighbours within `dup_window_s` = 30 s), agglomerative clustering (average linkage) on
  cosine distance of DINOv2 embeddings, threshold `dup_cos_dist` = 0.12. Additionally, if both images have faces,
  require main-face SFace cosine similarity ≥ 0.36 (same person) to merge — prevents merging different couples
  in the same venue. Output `group_id` for every image (singletons get their own).
  Must be O(n·w), not O(n²) over the whole shoot.

## 5. Rule-based quality (`quality.py`) — the "prior"
Sub-scores in [0,1]: `sharp` (face sharpness if faces else global), `eyes` (1 − closed prob over important faces;
1 if no faces), `exposure` (penalize |ev| > 1.5 and clipping > 5%), `expression` (smile bonus, small weight),
`subject` (faces present / main subject size). Weighted: sharp .35, eyes .30, exposure .15, subject .10, expression .10.
Hard-fail rules (produce reject + reason regardless of score):
- `blur`: global sharp_norm < 0.15 AND (no face OR face_sharp_norm < 0.25)
- `face_out_of_focus`: main face sharp_norm < 0.20 while global sharp_norm ≥ 0.35 (missed focus)
- `eyes_closed`: any important face eyes_closed (the largest face or ≥ 50% of important faces)
- `exposure_bad`: |ev| > 2.5 or clip_hi > 25% or clip_lo > 40%
- `flash_misfire`: see §7
- `accidental`: (very dark or very blurry) AND no faces AND exposure_time suggests not deliberate (e.g. mean luma < 8)

## 6. Selection (`selection.py`)
Per duplicate group (size > 1): rank by final score; the top `keep_per_group` (default 1; 2 if group ≥ 8)
are candidates; others → `reject` with reason `duplicate` ("A better version of this moment exists: <best filename>"),
unless their score is within `tie_margin` 0.03 of the best: those near-identical frames are `reject` with reason
`duplicate_tie` ("Near-identical to the best frame <name>; kept the sharper/eyes-open one"). They go to `review` only
when the best frame itself is in the uncertain band (`t_reject ≤ p < t_keep`), because then the pair needs a human.
Final score = `p_keep` = personal model probability if trained, else quality prior; blended:
`p = w·p_personal + (1−w)·prior`, `w = n_train / (n_train + 200)`.
Decision thresholds (per shoot after calibration): `p ≥ t_keep (0.60)` → keep; `p < t_reject (0.30)` → reject; else review.
Hard-fail rules override to reject (reasons attached). Every reject has ≥ 1 reason. Keeps may carry warnings.

## 7. Flash misfire (`flash.py`)
Compute per-image mean luma (in linear-ish space: (L/255)^2.2). Within its burst/time window (±10 s, same camera):
- If `flash_fired == False` but ≥ 50% of neighbours have `flash_fired == True` and image is ≥ 1.3 EV darker than their median → misfire.
- If `flash_fired == True` and image ≥ 1.5 EV darker than the median of other flash-fired neighbours → misfire (recycle/partial).
- Need ≥ 2 neighbours; else no decision. Output `flash_lum_delta_ev` feature + boolean.

## 8. Personal model (`personal/`)
- `dataset.py`: sources (any combination):
  - Lightroom catalog `.lrcat` (SQLite): label keep if pick == 1 or rating ≥ 1 (configurable `min_rating`), reject if pick == −1 or
    (rating == 0 and pick == 0 and the shoot had any picks/ratings — "seen but not selected", weak negative weight 0.5).
  - XMP sidecars in a folder: `xmp:Rating`, `xmp:Label`, `shotsieve:decision` written by the user's tool.
  - UI overrides stored in the cache DB (`overrides` table) with `propensity` of being shown.
  Features computed by running Stage A–C on those images (cached).
- **Bias correction**: each training row has `weight = base_weight / propensity`; rows from audit samples carry their sampling
  probability; plain catalog rows propensity 1.
- `model.py`: `PersonalModel.fit(X, y, w)` → `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, l2=1.0)` +
  isotonic or sigmoid calibration (CalibratedClassifierCV, cv=3 if n ≥ 60 else sigmoid on prior). `save/load` with joblib to
  `data/personal_model.joblib` with feature schema version; refuse to load mismatched schema.
  Require ≥ 40 rows and both classes, else stay on prior.
- `audit.py`: from the reject bucket sample `audit_n = min(25, max(5, 3% of rejects))` with probability proportional to
  `p_keep + 0.05` (rejects closest to the boundary are likelier). Record each sampled image's inclusion probability. Mark
  them `review` with `audit=true`. Deterministic with a seed per shoot.
- `calibrate.py`: given audit results (user kept k of n audited rejects), estimate missed-keeper rate with IPW; if estimate > 1%
  lower `t_reject` by 0.05 steps (min 0.10) and re-decide; return new thresholds. Persist per-shoot thresholds.

## 9. XMP writeback (`xmp.py`)
Write `<stem>.xmp` next to the master file (never modify the original; for JPEG also sidecar by default — option `--embed` is NOT in MVP).
Merge with an existing sidecar if present (preserve unknown elements; update only our fields). Fields:
`xmp:Rating` (keep → `rating_keep` default 3; review → 0; reject → `rating_reject` default 0, set −1 for Bridge/Capture One "rejected"), `xmp:Label`
(keep "Green", review "Yellow", reject "Red"; configurable), `shotsieve:Decision`, `shotsieve:Reasons` (rdf:Bag), `shotsieve:Score`.
Namespace `http://ns.shotsieve.app/1.0/`. Output must parse with `xml.etree` and be accepted by exiftool if present.

## 10. Runner (`runner.py`)
```python
def run_cull(folder: Path, settings: Settings, progress: Callable[[str, int, int], None] | None = None,
             write_xmp: bool = False) -> CullResult
@dataclass class ImageResult: path, master_path, group_id, burst_id, decision, reasons: list[str], score, features: FeatureVector, audit: bool, is_best_in_group: bool
@dataclass class CullResult: images: list[ImageResult], groups: dict[int, list[str]], thresholds, stats: dict (timings per stage, imgs/s, peak_rss_mb, counts)
```
Stages report progress (`"scan"|"features"|"faces"|"embed"|"group"|"score"|"write"`, done, total). Results persisted in
`<folder>/.shotsieve/cull.db` (SQLite: images, features, embeddings, decisions, overrides). Default cache location is
inside the shoot folder so moving the folder keeps state; tests use tmp dirs.

## 11. Ingest (`copier.py`)
`ingest(src, dest, backup=None, rename_template=None) -> IngestReport`: copy supported files preserving subfolders
(or flatten by date `YYYY-MM-DD/`), stream-copy with blake2b on the fly, verify dest hash (re-read), write backup in parallel,
skip files already present with same size+hash, never delete source. Report copied/skipped/failed + bytes + seconds.

## 12. Implementation notes (T2 engine; deviations from the text above)
- `Settings.personal_model_path` (default `data/personal_model.joblib`); `run_cull(..., *, personal_model=None)`.
  The model object needs `n_train: int` and `predict_proba(X)` (X = (N, 25) matrix in `FEATURE_NAMES`
  order; return (N,) or (N,2)). If `predict_proba` also accepts an `embeddings` keyword it receives the
  (N, 384) DINOv2 embeddings. Protocol: `scoring.selection.PersonalModelLike`. Blend `w = n/(n+200)`.
- Audit hook: `shotsieve.personal.audit.sample_audit(images, seed)` is called with `list[ImageResult]`
  (after decisions, before overrides/persist); it mutates `decision="review"`, `audit=True`,
  `audit_prob=<inclusion probability>`. Missing module or exceptions are ignored.
- Per-shoot thresholds are read from `cull.db` (`Cache.get_thresholds()/set_thresholds()`), else Settings.
- `n_faces` counts *important* faces; `face_sharp_min_important` is already normalised.
- Sharpness normalisation: per-shoot p90 with absolute floors; inside a duplicate group the reference is the
  sharpest sibling clamped to [0.25, 1] x shoot p90 (blur is judged against the same scene). Singletons use p90.
- `exposure_bad` (dark side / clip_lo) is suppressed when real highlights exist (clip_hi >= 1 %), so night
  scenes are not rejected. Blur/face_out_of_focus are not reported on frames that are exposure_bad-dark or flash misfires.
- Extra reason codes: `duplicate_tie`, `uncertain`, `audit` (review), `low_quality`, `no_subject`, `unreadable`.
  Undecodable files get decision `review` with `unreadable`.
- Stage overlap: decode+EXIF+Stage A in a 4-thread pool; bounded queues feed one face worker (YuNet, then
  SFace + MediaPipe on important faces only) and one embedding worker (batches of up to 8). Near-black
  frames (mean luma < 8) skip the face stage. Cache hits skip all of it (loads no models).
- Tie rule (T3 fix): `duplicate_tie` is a reject, not a review, unless the best frame is borderline (see §6). On the
  300-image benchmark this took review from 140 (47 %) to 10 (3 %: 6 audit spot-checks + 4 uncertain).
- T3 modules: `personal/model.py` (`PersonalModel`; untrained models report `n_train == 0` so the runner blend weight is 0;
  `SchemaMismatchError` on a feature-schema/format mismatch, which the runner turns into "prior only"),
  `personal/dataset.py` (label rules above, weight = base / max(propensity, 0.02); features come from a prior cull of the
  folder when its `cull.db` covers every label, else from running the stages), `personal/audit.py` (systematic PPS sampling
  in a seeded random order, so the recorded inclusion probability is exact; chosen images keep their reject reasons behind an
  `audit` reason), `personal/calibrate.py` (Horvitz-Thompson estimate of missed keepers = sum of 1/pi over audited rejects the
  user kept; rate = estimate / (estimate + keepers found); above 1 % `t_reject` drops 0.05 per step: one step plus one per
  further 5 points of rate, floor 0.10; thresholds persist in `cull.db`, then the shoot is re-decided from cached features).
- `Settings.audit_enabled` (default True) switches audit sampling off (tiny shoots would otherwise audit most rejects).
- `cull.db` additions: `image_ids` (stable integer image ids for the API) and `overrides.rating`.
- Rating-only overrides (keys 1-5) imply `keep` unless an override already exists.

## Review-fix addendum (behaviour changes)

- Face sharpness is measured at one canonical scale (crop area-downscaled to 96 px, never
  upscaled); faces under 96 px are not judged (NaN). The shoot blur reference is capped at
  `LAP_REF_CAP` (300) so smooth-content frames are not "blurry".
- Hard fails have tiers: certain (black frame, extreme blur, flash misfire, deep under-exposure),
  firm (both eyes closed, mild blur, large-face focus: overridable by a trained model with
  p >= t_keep or by a calibration margin) and soft (single closed eye, high-key exposure,
  small-face focus, long-exposure dark frame: review when p >= `t_soft` - `hard_margin`).
- Calibration levers per audit stratum: `t_reject` (score), `hard_margin` (hard fails),
  `dup_margin` (duplicates / failed siblings). Applied from defaults, idempotent, refused while
  audits are pending. Persisted only if changed.
- `decisions` stores the model decision; overrides live in `overrides` and are applied at read
  time. The audit sample is frozen per shoot (`--redraw-audit` to redraw) and excludes overridden
  images.
- Cache keys are shoot-relative paths + (size, mtime) or content fingerprint + extractor
  fingerprint (settings, model files, code version), so a moved folder keeps its cache.
- Duplicate components over 1000 frames use time-ordered leader clustering (groups <= 400).
- The personal model derives a per-user `t_reject` from the 1st percentile of its keepers.

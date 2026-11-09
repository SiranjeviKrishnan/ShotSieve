# T22 spec: subject-aware crop suggestions (architect, 2026-11-08)

User request: "select one or more subjects by pointing, have AI recommend a crop; the user accepts or rejects it; preset and
custom ratios; after a crop the whole image stays available for re-cropping."

Rules from 01/03/07/08/10 still hold: decode once, lazy model singletons, unit tests per module, ruff-clean, network-free tests
except `@pytest.mark.models` / `network`, no new top-level dirs, **originals never modified**, sidecar writes keep the r3 backup
rule and the per-sidecar lock, no GET writes, local Origin only (M5). Crop never runs inside `cull` (cull throughput unchanged).
**One new model (YOLOX-s, Apache-2.0, 35.9 MB, optional group `crop`). No new pip dependencies** (onnxruntime, OpenCV, numpy
are already in use).

| # | Item | Decision | Reason |
|---|---|---|---|
| T22 | Subject picking + AI crop suggestion + accept/reject + versions + `crs:Crop*` write | **build** | high user value; every input except an object detector is already in the repo |
| — | Rotation / straighten (`CropAngle ≠ 0`) | **out of scope** | needs rotated-rect geometry and LR's angle convention; a tilted horizon is only *reported* ("horizon tilted 2.1°") |
| — | SAM-family point segmentation (MobileSAM / EfficientSAM, Apache-2.0) | **defer** | +45–60 MB weights and two sessions; crops need boxes, not masks. The point-prompt interface below is model-agnostic so a SAM backend can drop in later |
| — | Learning the user's crop taste | **defer** | accept/reject/adjust events are logged (`crop_events`) so a later task can train on them |
| — | Rendered cropped full-res exports (RAW) | **defer** | needs a RAW render that matches LR (cut in 08); JPEG cropped export is a stretch item (§Stretch) |

## 1. What already exists (grounding)
- **Faces**: YuNet (MIT) at cull time; `FaceObs.bbox` (x, y, w, h in *preview px*, orientation applied, long edge 1280) is cached in
  `features.json` (≤ 6 faces, `mark_important`). Landmarks are **not** cached → T22 re-runs YuNet on demand (1–3 ms) to get eyes/nose.
- **Previews**: `io/preview.load_preview()` returns RGB with EXIF orientation applied (RAW: embedded thumb orientation or
  `raw.sizes.flip`); `/api/preview/{id}` serves a cached 1600-px JPEG (`.shotsieve/thumbs/p_{id}_1600_{mtime}.jpg`).
- **DINOv2-small fp32** (Apache-2.0) is in core models; its ONNX takes **dynamic H×W** and returns all tokens
  (`last_hidden_state [1, ⌊H/14⌋·⌊W/14⌋+1, 384]`). Measured here (M-series, CPU EP, 4 threads): 448×308 → 154 ms, 448×448 → 248 ms.
  Patch tokens give saliency and point-prompt regions with **no new weights**.
- **No person/object detector and no saliency model** exist. `features/jersey.torso_region` is a face→torso heuristic only.
- **`io/crs.py`**: `write_develop()` is the only crs writer (merge-safe, `shotsieve:EditProfile/EditKeys/EditDigest` provenance,
  never-overwrite, `.bak` per content hash, sidecar lock, embed-copy for JPEG/DNG). The carry set already excludes all `Crop*` keys
  (`personal/edit_dataset._NO_CARRY`). **Two conflicts T22 must fix** (see §5): (a) `EditDigest` hashes the *whole* crs set, so adding
  crop keys after an edit write makes that edit look like the user's and `edit apply` would then skip the image forever;
  (b) `writable()`/`write_develop()` treat any `HasSettings=True` not tagged by an edit write as the user's, so a crop-only sidecar
  would block `edit apply`.
- Free shortcut: **R** (Lightroom's crop key). In crop mode only, keys are scoped (X flips orientation as in LR; global X = reject is
  suspended while crop mode is open).

## 2. UX (loupe crop mode)
Entry: **R**, or a "Crop" button in the detail panel → opens the full-screen loupe (`#overlay-loupe`) in crop mode. The **full
frame is always shown**; the crop is an overlay (outside darkened 55 %, rule-of-thirds grid inside, 8 handles). Re-cropping always
starts from the full frame with the current crop drawn on it.

- **Subjects layer** (SVG over the image): every detected subject as a thin dashed box with a small label (`face`, `person`, `dog`…);
  hover highlights. Selected subjects: solid accent box + number badge.
- **Pointer selection**
  - Click = select exactly that subject (replaces the selection). Shift-click = toggle add/remove. A click maps to the **smallest
    detected box containing the point**, with these rules: a click inside a face that is linked to a person box selects the *person*
    with the face as must-keep core (`select_mode="person"`); Option/Alt-click on a face selects the face only (`"face"`, tight head
    shot). A click on a person box outside the face selects the whole person.
  - Click where no box contains the point → **point prompt** (§3.3): a region box is computed (≤ 250 ms cold) and added as a
    subject of kind `point` (dotted box). Shift-click adds more points/subjects.
  - Esc: first clears the selection; second Esc leaves crop mode without changes. "Clear" button does the same as the first Esc.
  - No selection = **auto**: all `important` faces + YOLOX persons/animals (score ≥ 0.5, area ≥ 1 % of frame) + the saliency peak
    region if nothing else exists.
- **Ratio picker** (chips): `Original`, `1:1`, `4:5`, `5:4`, `2:3`, `3:2`, `5:7`, `7:5`, `16:9`, `9:16`, `Free`; plus **Custom W:H**
  (two numeric inputs, decimals allowed, e.g. `1.91:1`; allowed range 1:4 … 4:1; invalid input shows inline error). **Flip** button
  (and X) swaps W:H. Default orientation = the image's (landscape image → 3:2 not 2:3) unless the user flips; `Original` = full-frame
  W:H. Ratios are exact in **full-resolution pixels** (§4.3), not preview pixels.
- **Suggestion**: runs automatically on entry and on every selection/ratio/flip change (debounced 150 ms; stale responses dropped by
  a request counter). Shows the top suggestion as the crop overlay and **up to 3 alternatives** as a strip of mini previews
  ("1 · 2 · 3", keys `[` / `]` cycle); each shows its warnings (e.g. "subjects don't fit at 9:16 — widest possible shown").
- **Accept** (Enter / button) → stores a new crop version, writes XMP per §5, toast with the write outcome. **Reject** (Backspace /
  button) → dismisses the suggestion; the image keeps its *previous* crop state (none or the last accepted); logs a reject event.
- **Adjust**: drag inside the crop = move; handles = resize, ratio-locked unless `Free` (Shift while dragging temporarily unlocks →
  ratio switches to `Free`). Live red outline + warning if a selected subject leaves the crop. Accepting an adjusted crop records
  `source="ai_adjusted"` (or `"manual"` when no suggestion was used). Rects snap to exact ratio on pointer-up.
- **Reset crop** (button, anytime) → new version with no crop (full frame); **History** dropdown lists every version
  (time, ratio, source) with **Restore**. Nothing is ever deleted.
- **Show crops** toggle in the toolbar (default on): grid thumbs and the loupe preview show the cropped view client-side (CSS: a
  wrapper with `overflow:hidden` + percentage-positioned `<img>`; **not** `object-view-box`, which Safari lacks). Cropped images get a
  small ⧉ badge. Off = full frames everywhere.
- **Apply to burst**: button "Apply to burst (N)" → transfers the accepted crop to the other frames of the same `burst_id` (§3.6);
  results arrive as *pending suggestions* in a filmstrip with "Accept all" / per-frame accept/reject. Never auto-accepted.
- Works with mouse and trackpad via Pointer Events; UI fits at 1280×800; no console errors. Video virtual items (`path#t=`) have the
  Crop button disabled ("not for video frames").

## 3. AI: detection, saliency, scoring
### 3.1 Models
| Use | Model | File / size | Licence | Runtime |
|---|---|---|---|---|
| Faces + 5 landmarks | YuNet 2023mar (existing) | 232,589 B | MIT | `cv2.FaceDetectorYN`, on the 1600 preview scaled to long edge 640 |
| People, animals, objects (COCO-80) | **YOLOX-s 2022nov (new)** | `yolox_s_2022nov.onnx` **35,858,002 B** | **Apache-2.0** (Megvii YOLOX; opencv_zoo `models/object_detection_yolox/LICENSE` = Apache-2.0 at the pinned commit) | onnxruntime CPU EP, 4 intra-op threads, arena off, lazy, idle unload 120 s |
| Saliency + point regions | DINOv2-small fp32 (existing) patch tokens | 88.5 MB (already installed) | Apache-2.0 | onnxruntime CPU EP, 4 threads, **own** session at dynamic size (not the cull embedder's), idle unload 120 s |

**YOLOX pin** (verified 2026-11-08 with `curl -sIL` → HTTP 200, downloaded and hashed):
- URL `https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/object_detection_yolox_2022nov.onnx`
  (same opencv_zoo commit as YuNet/SFace)
- size `35858002`, sha256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`
- I/O: input `images` `[1,3,640,640]` float32; output `output` `[1,8400,85]` (cx, cy, w, h raw, obj, 80 class scores). Measured
  97–102 ms per 640² frame (CPU, 4 threads), process RSS 203 MB with the session loaded (python + ORT baseline included).
- Registry: `ModelSpec("yolox_s", "yolox_s_2022nov.onnx", url, sha, 35858002, "Apache-2.0", group="crop")`; add to
  `PINNED_SHA256`; `shotsieve models download --extra crop`; the T19 wizard models step lists it as optional ("Smart crop: 36 MB").
  Record in 02_MODELS.md under "T22 crop models". Training data is COCO (annotations CC BY 4.0) — same posture as YuNet/DINOv2.
- Pre/post-processing **must match opencv_zoo `yolox.py`**: letterbox keep-ratio to 640 with pad value 114, BGR, float32 **0–255
  (no mean/std)**, HWC→CHW; decode with grids for strides 8/16/32 (`xy = (out + grid)·stride`, `wh = exp(out)·stride`),
  `score = obj · cls`, conf ≥ 0.35, class-wise NMS IoU 0.45, undo letterbox. The builder verifies the decode against the opencv_zoo
  demo on one CC0 image (boxes within 2 px) and freezes that as a golden test.
- **Graceful degradation**: without the `crop` group, crop mode works with faces + saliency + point prompts and shows "Download the
  object detector (36 MB) for better people/animal detection" (calls the T19 `/api/models` download lane).

### 3.2 Subject list (`pipeline/crop.py: detect_subjects(image_id) -> SubjectSet`)
Input: the cached 1600-px preview (orientation applied). Output in **normalized display coordinates** (§4.1):
- `face` (YuNet, score ≥ 0.6, all faces, landmarks kept), `person` / `animal` / `object:<coco name>` (YOLOX, score ≥ 0.45, area ≥ 0.3 %).
- Face↔person link: a face is linked to the person box that contains ≥ 90 % of it and whose top is within 0.25·person_h of the face top.
- Each subject: `sid` (stable within a fp), `kind`, `box`, `score`, `source`, optional `landmarks`, `linked`.
- Cached per image in cull.db `subjects` (§6), keyed by file fingerprint + `CROP_DET_VERSION`; plus a saliency map (§3.3, float16
  gh×gw ≤ 32×32 ≈ 2 KB). Patch tokens are **not** stored (≈ 540 KB/img); an in-memory LRU keeps the last 8 images' tokens (≈ 4 MB).

### 3.3 Saliency and point prompt (DINOv2 patch tokens, no new weights)
- Input: the preview resized so the long edge is 448 and both sides are multiples of 14 (short side ≥ 224), ImageNet norm (reuse
  `image_embed._MEAN/_STD`), aspect kept (no squash). Tokens `P` (gh×gw×384), L2-normalised.
- **Saliency** `s` (background-contrast on deep features): border ring patches → k-means k = 4 → centroids `c_k`;
  `d_i = 1 − max_k cos(P_i, c_k)`; `s = blur_σ=1patch(d)` normalised to [0, 1]. Saliency peak region = flood fill of `s ≥ 0.6`
  from the argmax.
- **Point prompt** at patch `p`: 4-connected flood fill over patches with `cos(P_i, P_p) ≥ τ_pt` (0.60) **or**
  (`cos ≥ 0.45` and `s_i ≥ 0.5`); dilate 1 patch; if the region exceeds 50 % of the frame, fall back to a box of 15 % frame area
  centred on the click. Box = patch bbox ± ½ patch. τ values are fitted on the train split only and frozen in code.
- The interface `point_region(image_id, x, y) -> Subject` is backend-agnostic (a future SAM backend replaces only this).

### 3.4 Crop scoring (`pipeline/crop_score.py`, pure numpy, vectorised)
Notation: crop `C` (x0, y0, x1, y1) in display-normalized coords, `w_C, h_C`; selected subjects `S`; unselected detected subjects `U`.

**Safe boxes** (must-keep core of each selected subject): face → bbox expanded 25 % of w each side, 45 % of h on top (hair), 20 % below
(chin); person (`select_mode="person"`) → person box + 2 % of its size each side; `face` mode → the face safe box only (body may be cut,
but only at allowed bands below); animal/object/point → box + 3 %. `R` = union of selected safe boxes.

**Hard constraints** (a candidate violating any is discarded):
1. `C` inside [0, 1]²; exact ratio in full-res px (§4.3).
2. Every selected face safe box fully inside `C`; every other selected safe box ≥ 98 % inside.
3. Full-res crop long edge ≥ `crop_min_long_edge_px` (default 1200) and area ≥ 15 % of the frame (prevents absurd zooms).
If no candidate satisfies 2 at this ratio → **infeasible**: return the largest crop at that ratio maximising covered safe-box area
(faces first), with `warnings=["subjects_do_not_fit"]`; the UI shows it in amber. Never a silent cut.

**Soft terms** (each in [0, 1]):
- `E_thirds`: anchor = eye line for faces (mean of YuNet eye landmarks; else box top + 0.38 h) / head point for persons (top + 0.07 h)
  / centroid for others; multi-subject anchor = area-weighted mean. `dx` = distance of anchor x (in crop-normalized units) to the
  nearest of {1/3: w 1.0, 1/2: w 0.85, 2/3: w 1.0}; `dy` to {1/3: 1.0, 1/2: 0.7} for faces/persons, {1/3, 1/2, 2/3} for others.
  `E_thirds = max_lines(w_l · exp(−d²/(2·0.07²)))` averaged over x and y.
- `E_head` (faces/persons): headroom `hr` = (top of topmost selected safe box − y0)/h_C; 1 for hr ∈ [0.04, 0.18], linear to 0 at 0 and 0.40.
- `E_lead` (single face): yaw sign from nose vs eye-midpoint offset (|offset| > 0.08 face w); 1 if space on the facing side ≥ behind,
  else 0.5; 1 when frontal.
- `E_cut` (no cut limbs, upright person boxes with h > 1.4 w and h ≥ 0.4 h_C): for each crop edge crossing the person box at relative
  height t (0 = top), penalty bands neck [0.12, 0.20], knees [0.70, 0.80], ankles/feet [0.92, 1.0) → 0; elsewhere 1; also left/right
  edges crossing a selected person box → 0.3. Term = min over crossings (1 if none).
- `E_strangers`: for each unselected face/person in `U`, inside-fraction `f`; cut if 0.05 < f < 0.95; `E = 1 − 0.5·n_cut`, floored at 0.
- `E_clear`: min gap between `R` and any crop edge / crop dimension; `min(1, gap/0.04)`.
- `E_sal`: saliency mass inside `C` / total (context kept) × (1 − saliency mass in a 2 %-wide border strip / mass inside) (no salient
  stuff sliced at the edge).
- `E_size`: `sqrt(area(C))` (prefer less zoom, keep resolution).
- `E_horizon`: if a dominant near-horizontal line exists (Canny + HoughLinesP on the preview, length ≥ 50 % width, |θ| < 5°):
  0.3 if it lies within ±4 % of the crop's vertical centre, 1 otherwise (and the UI reports tilt when |θ| ≥ 1°; no rotation).

`score = 0.24 E_thirds + 0.12 E_head + 0.06 E_lead + 0.12 E_cut + 0.12 E_strangers + 0.08 E_clear + 0.12 E_sal + 0.10 E_size + 0.04 E_horizon`
(`CROP_WEIGHTS`, frozen in code; may be re-tuned **once** on the train split, never on held-out). Terms and weights are returned per
candidate (`terms`) for debugging and tests.

### 3.5 Search
Ratio `r` in full-res px → preview-normalized ratio `r_n = r · (fh/fw)` (normalized coords are per-axis). Widths `w` from `w_max`
(largest at `r` inside the frame) down to the smallest allowed by constraint 3 or `w_R` (R's need), 24 log-spaced steps; for each,
25 × 25 positions over the interval that keeps `R` inside (clamped to the frame). ≈ 15 k candidates scored vectorised; refine the
top 10 by a ±½-step local search; snap (§4.3). **Alternatives**: greedy pick by score with diversity IoU < 0.70 to every pick
already chosen, up to `n` (default 3). Deterministic (no RNG; k-means seeded).

### 3.6 Burst transfer
For each frame B in the burst: match each selected subject of A to B (faces/persons: SFace cosine ≥ 0.5 when embeddings are cached,
else box IoU ≥ 0.3 after the burst's median shift; objects/points: DINOv2 token similarity at the anchor, best patch within 15 % of
frame). If all matched → re-run §3.4 on B with the same ratio, adding a term `E_same = exp(−‖offset_B − offset_A‖²/(2·0.05²))` (weight
0.15, others renormalised) that keeps the subject-relative framing; else the frame is returned as `needs_selection`.

## 4. Coordinates and geometry (`pipeline/crop_geom.py`)
### 4.1 Canonical form
`Rect(x0, y0, x1, y1)`, floats in [0, 1], `x0 < x1`, `y0 < y1`, **relative to the displayed image** = after EXIF orientation (and after a
`tiff:Orientation` in the sidecar when present, which is how LR stores a user rotation). Every API, DB row and UI value uses this form.
Each crop row also stores the `orientation` (1–8) it was computed under and the full-res `full_w × full_h` *display* dims.

### 4.2 To / from `crs:`
`crs:CropLeft/Top/Right/Bottom` are absolute fractions (Right/Bottom are coordinates, not margins) of the image **in stored
(sensor) orientation**, i.e. before EXIF orientation is applied. `to_crs(rect, orientation)` / `from_crs(crs, orientation)` apply the
inverse / forward of the 8 dihedral EXIF transforms (1 id, 2 mirror-x, 3 rot180, 4 mirror-y, 5 transpose, 6 rot90 CW, 7 transverse,
8 rot90 CCW). Written values: 6 decimals; `CropAngle="0"`, `CropConstrainToWarp="0"`, `HasCrop="True"`. The stored-frame convention is
the main correctness risk: confirm it with a real LR-written sidecar of a portrait RAW (CC0, `network`) **and** the user's manual LR check
(Acceptance 9). If LR turns out to use display orientation, only `to_crs`/`from_crs` change.

### 4.3 Exact ratio snapping
Full-res display dims `fw × fh`: RAW from `rawpy` `sizes.width/height` (visible area, swapped for orientations 5–8); JPEG/HEIC/PNG
from the decoded header; DNG from `DefaultCropSize` when present. Given the chosen rect and ratio `a:b`, `H = round(h·fh)`,
`W = round(H·a/b)`; if `W > fw` reduce H until it fits; keep the rect centre, clamp inside; return pixel rect → normalized. Guarantee:
`|W·b − H·a| ≤ max(a, b)` → width within **1 px** of the exact ratio for the integer height. `Original` = (fw, fh); `Free` = no snap.

### 4.4 Ratio parsing
`parse_ratio("4:5" | "1.91:1" | "original" | "free")`; numbers > 0, ≤ 2 decimals, max/min ≤ 4; else `ValueError` → HTTP 422 / CLI exit 2.

## 5. Non-destructive storage and XMP (`io/crs.py` additions)
- **Pixels are never modified.** The crop lives in cull.db (versions) and optionally in XMP as `crs:Crop*`. The full frame is always
  recoverable: Reset writes a new version, and the XMP is set back to LR's no-crop defaults.
- `CROP_KEYS = ("HasCrop", "CropTop", "CropLeft", "CropBottom", "CropRight", "CropAngle", "CropConstrainToWarp")`.
- New `write_crop(master, rect_crs | None, *, mode="sidecar"|"embed", out_dir=None, catalog_has=False, catalog_has_crop=False) -> WriteResult`
  (status adds `skipped_user_crop`), same lock / atomic replace / `backup_by_hash` / embed-copy machinery as `write_develop`:
  - **Crop provenance is separate from edit provenance**: `shotsieve:CropDigest` = sha256 of the canonical CROP_KEYS lines only,
    `shotsieve:CropVersion` = DB version. Crop is "ours" iff the digest matches.
  - **Edit digest v2**: `EditDigest` is computed over the crs set **minus CROP_KEYS**, marked `shotsieve:DigestV="2"`. Readers: when
    `DigestV == "2"` compare against the crop-excluded digest, else the legacy full digest (old sidecars keep their meaning). When
    `write_crop` touches a sidecar whose edit write is currently ours under v1, it re-signs it as v2. Test: edit apply → crop accept →
    edit apply again ⇒ second edit is `replaced_own`, crop unchanged.
  - **User develop settings present** (`HasSettings=True`, not ours): **merge** — only CROP_KEYS (+ provenance) change; every other
    attribute/child stays byte-for-byte equal; `.bak` taken once per content hash. This differs from `write_develop`, on purpose: a
    crop must coexist with the photographer's edits.
  - **User crop present** (`HasCrop=True` and CropDigest absent/mismatched): skip `skipped_user_crop` (UI: "Lightroom crop kept";
    shown read-only as the image's crop, source `lightroom`).
  - **Catalog** (when an lrcat is known via T20): image has a crop in the catalog → `skipped_user_crop`; image has other develop settings
    in the catalog but the sidecar does not carry them → `skipped_catalog` with "Lightroom holds edits for this photo; writing the
    sidecar could replace them on 'Read Metadata'. Crop saved in ShotSieve only." (Catalog stays read-only, T20 rule.)
  - **New sidecar** (no file): `HasSettings="True"`, `ProcessVersion` via `_process_version()` (T15 rule), CROP_KEYS, provenance;
    plus the edit-profile carry set when a profile exists (prevents LR from treating the image as "custom settings = Adobe defaults"
    instead of the user's import preset; see Risk 3).
  - `rect_crs=None` (Reset) writes `HasCrop="False"`, `CropTop="0"`, `CropLeft="0"`, `CropBottom="1"`, `CropRight="1"`, `CropAngle="0"`
    (LR's defaults, explicit so a later "Read Metadata" in LR also clears the crop), tagged as ours.
- `_describe()` gains `crop_ours`, `has_crop`, `crop` (raw values) and `only_ours` = every crs key except
  HasSettings/ProcessVersion/CROP_KEYS is absent or ours. `writable()` / `write_develop()` treat `HasSettings=True` with `only_ours`
  as **not** the user's (fixes conflict (b) in §1).
- JPEG/HEIC/PNG/DNG: LR ignores sidecars → default **DB only** + `lightroom_warning`; with `--embed-copies DIR` / UI checkbox, embedded-XMP
  copies into `<shoot>/shotsieve_edits/` exactly as T15 (original sha asserted). An existing copy with ShotSieve's edit is merged, not
  replaced.
- `read_crop(master) -> CropXmp(rect_display, ours, source)` uses `from_crs` with the sidecar/EXIF orientation.
- Settings: `crop_write_xmp: bool = True` (RAW sidecars on accept), `crop_min_long_edge_px = 1200`, `crop_threads = 4`,
  `crop_idle_unload_s = 120`.

## 6. Schema (cull.db, `CREATE TABLE IF NOT EXISTS`, no migration of old tables)
```sql
CREATE TABLE IF NOT EXISTS crops (            -- append-only history; current = max(version)
    image_id INTEGER NOT NULL, version INTEGER NOT NULL,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,       -- display-normalized; all NULL = full frame (reset)
    ratio TEXT, source TEXT NOT NULL,         -- ai | ai_adjusted | manual | reset | restore | lightroom | burst
    subjects_json TEXT, score REAL, terms_json TEXT, alt_rank INTEGER, det_version TEXT,
    orientation INTEGER, full_w INTEGER, full_h INTEGER,
    written_mode TEXT, write_status TEXT, written_at REAL, created REAL NOT NULL,
    PRIMARY KEY (image_id, version));
CREATE TABLE IF NOT EXISTS subjects (         -- detection cache
    path TEXT PRIMARY KEY, fp TEXT NOT NULL, det_version TEXT NOT NULL, json TEXT NOT NULL,
    sal_h INTEGER, sal_w INTEGER, saliency BLOB, created REAL);
CREATE TABLE IF NOT EXISTS crop_events (      -- for a future taste model; never read by T22 logic
    image_id INTEGER NOT NULL, ts REAL NOT NULL, action TEXT NOT NULL,   -- suggest|accept|reject|adjust|reset|restore
    rect_json TEXT, ratio TEXT, rank INTEGER);
```
Pending suggestions are not persisted (except burst-transfer results, kept in memory per job until accepted/rejected). Restore =
new version copying the old row (`source="restore"`). `build_shoot` adds `crop: [x0,y0,x1,y1] | null` per image (current version,
one indexed query; the T21 `build_shoot ≤ 1.0 s at 10k` budget still holds).

## 7. API (`server/crop.py`, one `register()` line in app.py, like onboarding)
All take `folder` like the existing endpoints; ids are `image_id`; POSTs only for writes; local Origin only.
- `GET /api/crop/{id}` → `{current, history[], full:{w,h}, orientation, xmp:{has_crop, ours}, ratios[]}`
- `GET /api/crop/{id}/subjects` → `{subjects[], saliency_ready, detector:"yolox_s"|null}` (computes + caches on first call; a cache
  fill inside a GET is a derived-data cache like thumbs, not a user-visible write — same rule as `/api/thumb`)
- `POST /api/crop/{id}/point {x, y}` → `{subject}`
- `POST /api/crop/{id}/suggest {subjects:[sid…], points:[{x,y}], select_modes:{sid:"person"|"face"}, ratio, flip:bool, n:3}` →
  `{candidates:[{rect, rect_px, score, terms, warnings}], ratio_px, infeasible}` (logs a `suggest` event)
- `POST /api/crop/{id}/accept {rect, ratio, source, subjects, alt_rank, write:"auto"|"none"|"embed"}` → `{version, write:{status, reason, path}}`
- `POST /api/crop/{id}/reject {alt_rank}` · `POST /api/crop/{id}/reset {write}` · `POST /api/crop/{id}/restore {version, write}`
- `POST /api/crop/burst {id, ids?}` → job (one-job lock) → `{results:[{id, rect|null, status:"suggested"|"needs_selection"}]}`;
  `POST /api/crop/accept_many {items:[{id, rect, ratio}]}`.
- Errors: unknown id 404, video item 422, bad ratio / rect outside [0,1] / rect not matching ratio (> 1 px) 422, model missing →
  works without YOLOX (`detector: null`).

## 8. CLI (`shotsieve crop …`)
- `crop detect FOLDER [--ids …|--keeps]` — fill the subject cache (job; ≥ 6 img/s cold).
- `crop suggest FOLDER --ratio 4:5 [--flip] [--ids …|--keeps] [--n 3] [--json]` — prints candidates; **writes nothing**.
- `crop apply FOLDER --ratio R [--ids …|--keeps] [--rect x0,y0,x1,y1 --id N] [--write auto|none|embed --embed-copies DIR] [--dry-run] [--json]`
  — an explicit CLI command is the user's acceptance: stores top-1 (or `--rect`) as a version, writes XMP.
- `crop reset FOLDER [--ids …|--all] [--write auto|none]` · `crop status FOLDER [--json]` · `crop history FOLDER --id N`.
- Exit codes: 0 ok, 2 bad args/ratio, 3 nothing to do. Document in 04_API_UI.md.

## 9. Budgets (M-series 16 GB; `scripts/benchmark.py --crop` adds the lines)
| Metric | Target |
|---|---|
| Hit-test a click (client) | < 5 ms |
| `subjects` cold (preview cached; YuNet + YOLOX + DINOv2 448) | ≤ 450 ms p50, ≤ 700 ms p95 |
| `point` warm (tokens in LRU) / cold | ≤ 30 ms / ≤ 300 ms |
| `suggest` warm (subjects cached), 3 alternatives, any ratio | **≤ 60 ms p95** (server); UI overlay update ≤ 100 ms after response |
| Drag handles | 60 fps (client-only; no server call while dragging) |
| `crop detect` batch / `crop apply` warm | ≥ 6 img/s cold, ≥ 30 img/s warm (300 bench) |
| Accept incl. sidecar write | ≤ 50 ms |
| RSS | **≤ +300 MB** over an idle server with YOLOX + DINOv2-crop sessions loaded; back to ≤ +30 MB after 120 s idle (unload); whole process (cull + indexer + query + OCR + T15–T17 + crop) ≤ 3 GB |
| Cull | unchanged: ≥ 10 img/s and within ±5 % of the last bench (crop is not in cull) |
| `build_shoot` 10k with crops | still ≤ 1.0 s |

## 10. Fixtures and evaluation
- `scripts/make_crop_fixtures.py` → `tests/fixtures/crop/synth/`: rendered scenes (ellipse "faces" with eye dots, rect "bodies",
  textured objects, a horizon line) with exact boxes — deterministic unit tests for geometry/scoring with no models.
- `tests/fixtures/crop/real/`: **≥ 40 CC0 / public-domain photos** (≤ 1280 px long edge, ≤ 8 MB total; sources: existing CC0
  fixtures, Wikimedia Commons CC0/PD, Unsplash only for photos dated before 2017-06-05, the existing rule) listed with source URL +
  sha256 in `tests/fixtures/crop/LICENSES.md`. Mix: 10 single portraits (both orientations), 8 groups (2–6 people), 6 animals,
  6 objects outside COCO (point-prompt cases: statue, lighthouse…), 5 horizon landscapes, 5 off-centre action/sports; plus 4 copies
  re-encoded with EXIF orientation 3/6/8 (pixels rotated, tag set) and 1 CC0 portrait-orientation RAW (raw.pixls.us, `network`).
- `crop_labels.json`: per image `subjects[{kind, box (display-normalized), must}]`, optional `horizon_y`, and for 20 images a
  human **reference crop** at 4:5 and 16:9. Labels are written by someone **other than the scorer's builder** (lesson from T17 N1),
  sha256 frozen in TASKS.md; split 24 train (any tuning) / 16 held-out, **held-out scored once** at acceptance.

## 11. Acceptance criteria
1. **Containment**: over held-out images × 8 preset ratios, on every feasible (image, ratio) pair, all must-keep labelled subjects
   are ≥ 98 % inside the top-1 crop in **≥ 95 %** of pairs; labelled **faces are never cut** by a crop edge (0 faces with 2–98 %
   inside) in top-1 and in every alternative; infeasible pairs are flagged `subjects_do_not_fit` in **100 %** (no silent cut).
2. **Ratio**: every returned/stored crop's full-res pixel rect is within **1 px** of the exact ratio (property test, 1000 random
   dims × ratios incl. custom, both orientations, all 8 EXIF orientations); rect always inside the frame.
3. **Unselected people**: an unlabelled-as-must face/person is cut (5–95 % inside) in ≤ 10 % of top-1 crops where an alternative
   without the cut exists.
4. **Reference crops** (held-out 20 × 2 ratios): best-of-3 IoU with the human reference ≥ 0.75 in ≥ 70 % of cases, and top-1 mean
   IoU beats a centre-crop baseline by ≥ 0.10.
5. **Detection / pointing**: YOLOX person recall (IoU ≥ 0.5) ≥ 0.85 on labelled people; a click inside a labelled detected box selects
   it in 100 %; point prompts on non-COCO objects give box IoU ≥ 0.5 with the label in ≥ 70 %.
6. **Round-trip**: `write_crop → read_crop` equals the rect to 1e-6 for all 8 orientations; reset → reads back "no crop"; history keeps
   every version and Restore reproduces it exactly; original file sha256 unchanged in every test; every non-crop crs attribute and
   foreign XMP content byte-equal after a crop write onto a user-edited sidecar; `.bak` exactly once per content hash.
7. **Coexistence**: edit apply → crop accept → edit apply ⇒ `replaced_own`, crop unchanged; crop-only sidecar → `edit apply` writes
   (not `skipped_user_settings`); user's own crop never overwritten (`skipped_user_crop`); legacy v1 EditDigest sidecars still "ours".
8. Budgets of §9 met; recorded in `data/benchmark.md` and STATUS.
9. **Manual (user)**: in Lightroom Classic, "Read Metadata from Files" on one landscape and one portrait-orientation RAW with a ShotSieve
   4:5 crop shows the same framing as ShotSieve's overlay and the user's import preset/other edits are intact; Reset clears it. Recorded
   in STATUS like T18 acceptance 7.
10. ruff clean, full suite green, no Playwright console errors, existing UI tests unchanged.

## 12. Test plan
- `tests/test_crop_geom.py`: 8 orientation transforms (forward∘inverse = id; `to_crs` golden values for 1/3/6/8), ratio parsing
  (valid/invalid → errors), snapping property test (Acc. 2), Original/Free.
- `tests/test_crop_score.py` (synthetic, no models): face on the left third stays on a third; must-keep never cut; infeasible flagged
  and maximal; stranger at the edge avoided; neck/knee band penalised (crop bottom not at 0.15·person h); headroom window; horizon not
  centred; alternatives diverse (IoU < 0.7) and deterministic; terms sum with `CROP_WEIGHTS`.
- `tests/test_crop_detect.py`: YOLOX decode/NMS on a canned `[1,8400,85]` tensor (network-free); letterbox undo; face↔person linking;
  cache invalidation on fp change and `CROP_DET_VERSION` bump; without the model → faces + saliency only. `@models`: golden boxes on
  one CC0 image vs opencv_zoo demo (≤ 2 px); saliency peak on a single-object image inside the object box.
- `tests/test_crop_xmp.py`: every case in §5 (new sidecar, foreign keys, user develop settings merge, user crop skip, catalog skips,
  reset defaults, v1/v2 digest, re-sign, concurrent `write_decision` + `write_crop` under the lock, embed copy for JPEG/DNG with
  Pillow `getxmp()` reading `crs:CropTop`, original sha unchanged).
- `tests/test_crop_api.py` / `test_crop_cli.py`: endpoints and errors (404/422), versions/reset/restore, burst job under the job lock,
  `suggest` writes nothing, `apply --dry-run` writes nothing, no GET writes to user-visible state, local Origin enforced.
- `tests/test_crop_quality.py` (`models`): Acceptance 1, 3, 4, 5 on the labelled set (held-out run behind an explicit flag so it is
  scored once; the result is recorded in TASKS.md).
- **Playwright** (`tests/test_crop_ui.py`, `ui` marker, shared `browser` fixture): R opens crop mode with subject boxes on a two-face
  fixture; click face → selected badge; shift-click second face → both selected and the suggested rect (exposed as
  `data-rect` on the overlay) contains both safe boxes; click empty background → a `point` subject appears; 1:1 chip → overlay aspect
  1:1 within 1 CSS px; custom 7:5 + Flip → 5:7; `]` switches alternative; drag a corner → ratio kept, source becomes `ai_adjusted`;
  Accept → ⧉ badge in grid, `GET /api/crop/{id}` version 1, RAW fixture sidecar contains `crs:HasCrop="True"`; Reject leaves state
  unchanged; Reset → full frame shown, history 3 entries, Restore v1; "Show crops" toggle changes the grid thumb box; Esc/Enter/Backspace;
  video item has Crop disabled; 1280×800 fit; no console errors.
- Bench: `scripts/benchmark.py --crop` (subjects cold/warm, suggest p95, detect/apply img/s, RSS loaded/idle-unloaded, cull unchanged).

## 13. Build order (one builder, lane "product"; engine helps on models)
1. `crop_geom.py` + tests → 2. `crop_score.py` on synthetic fixtures → 3. registry YOLOX + `models/object_detect.py` + DINOv2 patch
session (`models/patch_features.py`) + `detect_subjects` + cache → 4. `io/crs.py` `write_crop`/`read_crop` + digest v2 + `only_ours`
(+ regression tests for T15/T16 write paths) → 5. schema + `pipeline/crop.py` versions → 6. API + CLI → 7. UI crop mode + Playwright →
8. burst transfer → 9. fixtures/labels (independent labeller) + quality + bench → 10. docs (02_MODELS, 04_API_UI, STATUS), then an
independent review.

## 14. Risks
1. **crs crop frame vs orientation** (stored vs display) is the biggest correctness risk; isolated in `to_crs/from_crs`, covered by
   the 8-orientation tests, a real LR-written portrait sidecar fixture and the user's manual check (Acc. 9).
2. **Lightroom reads XMP only on import or "Read Metadata from Files"**, which replaces catalog develop settings with the sidecar's.
   Mitigation: merge into the user's sidecar instead of skipping; skip + explain when the catalog holds edits the sidecar lacks; never
   write the catalog.
3. A crop-only new sidecar with `HasSettings=True` + PV 11.0 may make LR apply Adobe defaults instead of the user's import preset on
   import. Mitigation: include the edit-profile carry set when available; manual LR check; if confirmed bad, a setting
   `crop_new_sidecar=False` (DB only for never-edited RAWs) is the fallback.
4. **Edit/crop provenance collision** (§1 conflicts a/b): fixed by digest v2 + `only_ours`; regression tests on T15/T16 paths are
   mandatory before merging.
5. Saliency/point regions from DINOv2 patches are coarse (≈ 3 % of the frame per patch at 448) and weak on low-contrast subjects;
   the user can drag, and the backend is swappable (SAM deferred).
6. "Good crop" is subjective: metrics are geometric guarantees + reference-crop IoU on an independently labelled held-out set
   scored once; weights frozen.
7. Memory: YOLOX (~140 MB resident) + a second DINOv2 session at 448 (~150 MB); both lazy with 120 s idle unload; RSS test.
8. RAW full-res dims from rawpy can differ by a few px from LR's default crop → ratio exact in ShotSieve's dims, ≤ 0.2 % in LR.
9. COCO-trained weights (YOLOX) — annotations CC BY 4.0, images Flickr-licensed; same posture as existing models (Apache-2.0 weights).

## Stretch (only after acceptance passes)
- `crop export FOLDER --out DIR` for JPEG masters: cropped q95 JPEG from the full-res decode, EXIF copied, original untouched.

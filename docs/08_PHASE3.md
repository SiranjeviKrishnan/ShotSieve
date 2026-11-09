# Phase 3 spec (architect decision, 2026-11-07)

Build in this order. Each section is a task in TASKS.md (T14…T17). Rules from 01/03/07 still hold: decode once, lazy model
singletons, unit tests per module, ruff-clean, network-free tests except `@pytest.mark.models` / `network`, no new top-level dirs,
originals never modified, sidecar writes keep the r3 backup rule.

| # | Item | Decision | Reason |
|---|---|---|---|
| T14 | Benchmark HEVC budget → 350 MB | **build first** | Review #5 ruling; the script still says 300 |
| T15 | Local editing profile (old T13) | **build** | top user ask; every input already cached (DINOv2, EXIF, previews); rawpy gives as-shot WB |
| T16 | Reference-frame colour match | **build (thin)** | reuses T15 writer + WB estimator; works with zero edit history, so it covers users under 200 edits |
| T17 | Shot-list coverage alerts | **build** | pure reuse of the T10 SigLIP2 index; small code, high value for weddings and events |
| — | Album sequencing | **defer** | depends on taste and layout, there is no ground truth to test against, and album tools already own this step; revisit using T17 coverage + people data |
| — | School roster + compliance | **cut** | different segment (picture day: barcode cards, packages), and minors' biometrics under COPPA/BIPA raise the legal bar beyond a local culler |
| — | HSL / tone curve / profile prediction in T15 | **defer** | 24 HSL sliders + curve points need ≫ 200 edits and global stats carry little signal for them; they are passed through as "carry" values only (T15 §write) |
| — | Reverse-engineered LR rendering preview | **cut** | an approximate render that differs from LR misleads more than it helps |

## Budgets (every task; `scripts/benchmark.py --phase3`)
- **Cull ≥ 10 img/s, unchanged within ±5 %** on the 300 bench. Nothing in Phase 3 runs inside `cull`; edit/match/shotlist are
  separate jobs under the existing one-job-at-a-time lock.
- **Peak RSS ≤ 3 GB**, 16 GB Apple Silicon. `edit train` ≤ +400 MB over an idle server; `edit apply` ≤ +150 MB; `shotlist`
  uses the already-budgeted SigLIP2 text tower (+~450 MB, idle unload). Whole process with indexer running stays ≤ 3 GB.
- `edit apply` ≥ 50 img/s on cached embeddings; featurising uncached catalog images ≥ 15 img/s (DINOv2 + stats); `edit train` ≤ 60 s
  for 10k edited rows (featurisation excluded); profile file ≤ 10 MB. `match` ≥ 30 img/s. Shot list of 100 items over 20k frames < 1 s warm.

## Models (verified 2026-11-07 with `curl -sIL`, HTTP 200, exact sizes)
**No new weights in Phase 3.** Reused:
| Model | URL | Bytes | Licence |
|---|---|---|---|
| DINOv2-small fp32 | `https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/model.onnx` | 88,532,934 | Apache-2.0 (facebook/dinov2-small) |
| SigLIP2 vision / text int8 / tokenizer | `…/siglip2-base-patch16-224-ONNX/resolve/ba1f3b0843f24bc5417d38e19c37b287d719b2f4/{onnx/vision_model.onnx, onnx/text_model_quantized.onnx, tokenizer.json}` | 371,807,752 / 283,438,275 / 34,363,039 | Apache-2.0 |
- T15 housekeeping: registry pins DINOv2 to commit `8b1f705…` instead of `main` (same bytes, so the sha256 is unchanged).
- Learners: scikit-learn `HistGradientBoostingRegressor` (BSD-3, already a dependency). RAW metadata: rawpy (MIT, already a dependency;
  metadata only, no demosaic on the hot path).
- Optional, developer-only eval: MIT-Adobe FiveK (`https://data.csail.mit.edu/graphics/fivek/`, HTTP 200). Its licence is
  **non-commercial research**: never vendored, never in CI, never used to ship weights. It can only be run locally via `scripts/eval_fivek.py`.

---
## T14 — Benchmark HEVC RSS budget 350 MB (stream: engine) · deps: none
- `scripts/benchmark.py`: add `HEVC_RSS_BUDGET_MB = 350`; use it in `hevc_lines()` (the PASS/OVER test and text currently hardcode 300). H.264 stays at 300.
- Re-run `--n 300`; `data/benchmark.md` shows "target <= 350 MB"; STATUS row unchanged otherwise. Test: a unit test calls
  `hevc_lines({"rss_added_mb": 320, …})` → contains "PASS".

## T15 — Local personal editing profile (stream: product; engine for io/)
### Modules
- `io/lrcat.py` + `read_develop(path) -> list[DevelopEntry(path, fmt, settings: dict[str, float|str], pv)]`: join
  `Adobe_imageDevelopSettings.image = Adobe_images.id_local` and parse `text` (Lua `s = { Key = value, … }`): top-level scalar keys only, with
  nested tables (`ToneCurvePV2012 = {…}`) skipped by a brace-depth scanner. Keep rows with `Exposure2012` present (PV2012+, `processVersion` ≥ 6.7)
  and with ≥ 1 target ≠ default. Virtual copies are skipped, the read is read-only, and the locked-copy fallback is reused.
- `io/crs.py`: `CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"`; `read_develop_xmp(master)` (sidecar or embedded packet);
  `format_value(key, v)` (Exposure2012 `+0.35`, two decimals; Temperature int K clamped to 2000–50000; Tint −150…150; −100…100 sliders as signed ints);
  `write_develop(master, values, carry, provenance, mode)`.
- `io/jpeg_xmp.py`: copy a JPEG and insert/replace the APP1 `http://ns.adobe.com/xap/1.0/\0` segment (≤ 65,502 B, no extended XMP); all
  other segments are byte-identical.
- `io/dng_xmp.py`: copy a DNG, append the new XMP packet at EOF, and patch IFD0 tag 700 (offset/count) in place. If tag 700 is absent, append a
  rewritten IFD0 with the extra entry and patch the header offset. Strip/tile offsets are never moved.
- `features/color_stats.py`: `edit_stats(rgb256, exif, raw_wb) -> np.ndarray` (≤ 3 ms, NaN-tolerant), `EDIT_STATS_VERSION`:
  luma (linear 16-bin log hist, p1/p5/p50/p95/p99, clip_hi/lo, `exposure.analyse` ev, RMS contrast); colour (grey-world and shades-of-grey p=6
  ln R/G, ln B/G; midtone Lab a*/b*; chroma mean/p90); EXIF (log2 ISO, log2 t, N, EV100, flash, focal, hour, camera idx); RAW
  (as-shot mired + Δuv, see WB; NaN for JPEG); faces (count, mean face luma, if boxes are cached).
- `features/raw_wb.py`: `as_shot(path) -> (mired, duv) | None` via `rawpy` metadata only: neutral = 1/`camera_whitebalance`
  → XYZ through inverse `rgb_xyz_matrix` → xy → CCT (McCamy) + Δuv. The absolute value differs from Adobe's Kelvin, but it is used only as a
  **difference**, the same way in train and predict. DNG: `AsShotNeutral` (tag 50728) when present.
- `personal/edit_dataset.py`: catalog/XMP rows → (features, targets). Featurise from the **embedded/camera preview**, which matches what predict
  sees. Skip DNG whose `PreviewApplicationName` (tag 50971) contains "Adobe": the preview has the edit baked in, a label leak. `--dng-render` uses a rawpy
  half-size render instead. Feature cache: `~/.shotsieve/edit_features.db` keyed by `cache.fingerprint`; shoots already culled reuse their DINOv2 rows.
- `personal/edit_profile.py`: `EditProfile.fit / predict / save / load` (same save mechanism and dir as `PersonalModel`).
- `pipeline/edit.py`: train / apply jobs; `edits` table in cull.db `(image_id, values json, confidence json, written_mode, written_at, profile_hash)`.

### Targets
`Exposure2012, Contrast2012, Highlights2012, Shadows2012, Whites2012, Blacks2012, Vibrance, Saturation`, plus WB:
- RAW: target = `mired(Temperature) − as_shot_mired` and `Tint − as_shot_tint_proxy`. Rows with `WhiteBalance = "As Shot"` give target 0.
- JPEG: `IncrementalTemperature`, `IncrementalTint` (no physical units, own head). If no JPEG rows are present, the JPEG WB head is "off".
- Two WB heads per format: `P(user changes WB)` (HGB classifier), then the regressor. If p < 0.5, write nothing for WB.

### Algorithm
- Design matrix: edit_stats (~45) + DINOv2 PCA-24 (fit on training rows) + format class.
- Per slider: HGB point model (`loss="squared_error"`, max_iter 300, lr 0.05, max_leaf_nodes 15, l2 1.0, early stopping), plus q10/q90 models
  (`loss="quantile"`). Chain: tone sliders and WB get the **out-of-fold** predicted Exposure as a feature.
- Output shrinkage: ŷ = w·model + (1−w)·user_median, w = n/(n+100).
- Minimum: n ≥ 200 edited frames, else refuse with a message ("need 200 edits, have N; use `match` instead").
- Train-time slider mode, from 5-fold CV **grouped by shoot folder** (bursts must not leak):
  `learned` if MAE_cv ≤ 0.85 × MAE_median; else `constant` (write the median) if the user's IQR ≤ tol_s; else `off`.
  tol: Exposure 0.15 EV, WB 10 mired / 5 tint, sliders 5.
- Per-image confidence gate (both must hold, else that slider is omitted and the image is badged "needs your edit"):
  (a) q90−q10 ≤ W_s, where W_s = the 80th percentile CV width (~80 % written by design); (b) OOD: the mean cosine distance from the image's DINOv2 to its
  5 nearest training rows ≤ the p95 of the same statistic in CV. Couplings: Exposure withheld → withhold Highlights/Shadows/Whites/Blacks; Temp and Tint
  are written together or not at all.

### Write policy
- RAW: sidecar `crs:` attributes, `crs:HasSettings="True"`, `crs:ProcessVersion="11.0"`, `crs:WhiteBalance="Custom"` only when WB is written,
  provenance `shotsieve:EditProfile=<hash>`, `shotsieve:EditConfidence`.
- **Carry set**: every crs key (profile, lens corrections, sharpening, NR, HSL…) where ≥ 80 % of the user's edits share one value is written verbatim.
  Without it, `HasSettings=True` would replace the user's import preset with Adobe defaults.
- Never overwrite: skip if the sidecar has `crs:HasSettings=True` not written by ShotSieve, or the lrcat (when given) shows develop settings for the image.
  ShotSieve's own previous write may be replaced (provenance match).
- JPEG/DNG: Lightroom ignores sidecars here, so the default is report-only plus a warning (`lightroom_warning`). With `--embed-copies DIR` (UI checkbox), write
  embedded-XMP **copies** into `DIR` (default `<shoot>/shotsieve_edits/`) for re-import. Originals are never modified (sha256 asserted).

### CLI / API / UI
- `shotsieve edit train --catalog X.lrcat [--xmp DIR…] [--dng-render]` → prints per-slider mode, CV MAE vs median baseline, n.
- `shotsieve edit apply FOLDER [--scope keeps|all] [--dry-run] [--embed-copies DIR] [--json]` · `shotsieve edit status`.
- `GET /api/edit/profile` · `POST /api/edit/train` · `POST /api/edit/apply {scope|ids, dry_run, embed_copies}` (job) ·
  `GET /api/edit/predictions?ids=` · `DELETE /api/edit/profile`.
- UI: an "Edit profile" sheet (train, per-slider table learned/constant/off with MAE), "Apply to keeps" (dry-run diff first), a loupe badge with predicted
  values + confidence, and a "needs your edit" filter.

### Evaluation without user data: `scripts/make_edit_styles.py` → `tests/fixtures/edit_styles/`
- Base corpus: the 300 bench images + CC0 fixtures, at 5 perturbations each. Known Δev ~ U(−1.5, 1.5) as a linear-light gain (re-gamma, clip), WB cast
  ~ U(−60, 60) mired by von Kries on linear RGB, tint cast ~ U(−0.01, 0.01) Δuv. Scene cluster = DINOv2 k-means (k = 6) on the base image.
- Ground-truth sliders = the photographer *undoes* the perturbation plus a personal style:
  - **A "bright & airy"**: Exp = −Δev + 0.4 + 0.3·[indoor cluster]; WB = −cast − 8 mired; Shadows = 30 + 20·shadow_frac; Highlights = −40·clip_hi_norm;
    Contrast −10; Vibrance +15; Saturation −5.
  - **B "moody"**: Exp = −Δev − 0.3; Contrast +25; Blacks −20; Whites +10; WB = −cast + 10; Saturation −15 except one cluster at +5.
  - **C "random"** (control): iid sliders.
  Noise σ: Exposure 0.08 EV, WB 4 mired, sliders 3. The lrcat + XMP fixtures are written in real Lightroom text format, so the readers are exercised end to end.
- Split grouped by base image. Targets on held-out data (A and B each):
  | Slider | MAE target | and vs median baseline |
  |---|---|---|
  | Exposure2012 | ≤ 0.15 EV | ≤ 0.5× |
  | Temp (mired Δ) / Tint | ≤ 10 mired / ≤ 5 | ≤ 0.6× |
  | Contrast, Highlights, Shadows, Whites, Blacks, Vibrance, Saturation | ≤ 6 | ≤ 0.6× where the style has signal; ≤ 1.1× for constant sliders |
  Also: q10–q90 empirical coverage 0.70–0.90; ≥ 75 % (image, slider) written on A/B; on C ≤ 10 % written and every slider `off`/`constant`; gated MAE ≤ ungated MAE;
  n = 200 passes the Exposure target; n = 150 refuses to train.
- Tests: Lua parser (nested tables, strings, negatives, PV filter); as-shot WB on a CC0 full RAW (`network`, raw.pixls.us) gives a monotonic mired vs a daylight/tungsten pair;
  crs round-trip (write → `read_develop_xmp` equal; golden sidecar string); never-overwrite cases; carry set; JPEG copy (pixels and non-APP1 segments identical, Pillow
  `getxmp()` reads crs); DNG copy (rawpy `raw_image` identical, tag 700 parses, original sha unchanged); DNG Adobe-preview skip; API/CLI; Playwright: sheet + badge; bench
  cull throughput unchanged.

## T16 — Reference-frame colour match (stream: product) · deps: T15 (io/crs, raw_wb, color_stats)
- `pipeline/match.py`: `match(folder, ref, targets) -> list[MatchResult(values, skipped_reason)]`. The reference crs comes from its sidecar, embedded XMP or lrcat;
  if it has none, refuse ("edit the reference in Lightroom first").
- Per target t vs reference R (256-px previews):
  - Gate: DINOv2 cos(R, t) ≥ 0.60 ("same scene"), else skip "different scene". Clip_hi > 25 % → skip exposure.
  - Exposure: Δev = log2(L_R / L_t), with L = linear-light median over pixels in luma p5–p95; |Δev| ≤ 2 else skip; Exp_t = Exp_R + Δev.
  - WB (RAW): scene mired s = as_shot_mired + c, where c = residual cast of the preview (shades-of-grey p=6 on low-chroma, unclipped pixels → mired).
    δ = mired(K_R) − s_R; mired_t = s_t + δ; Tint likewise in Δuv with a constant `TINT_PER_DUV` fitted once on the synthetic set and frozen in code.
    WB on JPEG targets: off in T16 (incremental units are not physical).
  - Every other crs key of R is copied (an LR "sync"), with Exposure/WB replaced by the matched values.
- Writing uses the T15 policy (never-overwrite, carry, embed-copies for JPEG/DNG).
- CLI `shotsieve match FOLDER --ref IMG [--targets keeps|group|camera:MODEL|ids] [--dry-run]`; `POST /api/edit/match`; UI: loupe "Set as reference" and "Match
  selected" (the builder picks unused keys and documents them in 04_API_UI.md).
- Accept (synthetic, from T15 base corpus): with a known gain/cast between R and t, Exposure error MAE ≤ 0.10 EV and WB ≤ 8 mired; different-scene pairs are 100 % skipped;
  a two-body test with a CC0 Canon + Sony full RAW (`network`) matches within 15 mired of a hand-measured grey patch; ≥ 30 img/s.

## T17 — Shot-list coverage alerts (stream: search) · deps: T10 index (independent of T15/T16)
- `pipeline/shotlist.py` + `data/shotlists/{wedding,sports,corporate}.txt` (≥ 15 items each). Line syntax:
  `[!]text [person:"Name"] [xN]`, where `!` = must-have and `xN` = min count (default 1).
- Algorithm: for each item, run the T10 `search()` core restricted to keep+review frames (rejects excluded; person filter via T10 people/roster). Raw SigLIP2 cosine s.
  Per-item z = (s_top − median_s) / (1.4826·MAD_s) over the shoot. **covered** if ≥ N frames have s ≥ τ and z ≥ 3; **weak** if the top z ≥ 3 but fewer than N reach τ;
  **missing** otherwise. τ is fitted on `search_queries.json` + 20 new absent queries (max F1 on "any relevant present", frozen in code). If the index is < 100 %, status is
  `pending (indexed X %)`, never "missing".
- Storage: `shotlist(item_id, text, must, min_count, status, best_ids json, updated)` in cull.db. Opt-in XMP keyword `shotlist:<item>` on matched keeps via `merge_keywords`.
- CLI `shotsieve shotlist FOLDER --list shots.txt [--json] [--tag-xmp]`; `GET/POST /api/shotlist`, `GET /api/shotlist/coverage`; UI "Shot list" panel (✓ / ~ / ✗,
  best-frame thumbs, click → frames; a banner when a must-have is missing after a cull or index completes).
- Accept: on the bench + fixtures, item classification accuracy ≥ 0.90 (30 present + 20 absent); must-have false-"covered" ≤ 5 %; pending never reported
  missing; 100 items < 1 s warm; RSS within the T10 budget.

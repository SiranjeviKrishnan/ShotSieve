# Phase 4 spec: usable by a photographer who is not us (architect, 2026-11-08)

Source: `docs/09_NEXT.md` § Phase 4. Tasks T18–T21 in TASKS.md. Rules from 01/03/07/08 still hold: decode once, lazy model
singletons, unit tests per module, ruff-clean, network-free tests except `@pytest.mark.models` / `network`, no new top-level
dirs except the two named in T18 (`.tools/`, `ShotSieve.app`, both gitignored), originals never modified, sidecar writes keep the r3
backup rule, no GET writes, local Origin only (M5). **No new models, no new pip dependencies** (everything below uses the stdlib,
Starlette's built-in `GZipMiddleware`, and macOS system tools: `curl`, `tar`, `shasum`, `plutil`, `osascript`, `open`, `sips`, `iconutil`).

| # | Task | Builder lane | Build order | Main files |
|---|---|---|---|---|
| T18 | One-command install + `ShotSieve.app` launcher | A "packaging/UX" | 1 (parallel with T21) | `install.sh`, `Install ShotSieve.command`, `scripts/build_app.sh`, `scripts/launcher.sh`, `models/registry.py`, `cli.py` (`doctor`), `server/app.py` (health, quit) |
| T21 | 10k-frame shoots | B "engine/store" | 1 (parallel with T18) | `pipeline/runner.py`, `pipeline/cache.py`, `server/app.py` (`build_shoot`, gzip, thumbs), `static/app.js` (grid), `scripts/benchmark.py`, `tests/large_synth.py` |
| T20 | Lightroom Classic sync-back | B | 2 (after T21) | `io/lrcat.py`, new `pipeline/lrsync.py`, `pipeline/cache.py` (2 tables), `personal/dataset.py`, `io/xmp.py` call sites, `cli.py` (`sync`), `server/app.py` (`/api/lr/*`), small UI |
| T19 | First-run onboarding wizard | A | 3 (after T18; LR card + catalog step last, after T20 lands) | new `static/onboarding.js`, `index.html`, `style.css`, `server/app.py` (`/api/onboarding`, `/api/fs/*`, `/api/models*`, `/api/train`), `models/registry.py` (progress) |

Why this order: T18 and T21 touch disjoint code (only `app.py` health vs `build_shoot`; trivial merge). T20 and T21 both touch `cache.py`
and the runner's XMP write path, so the same builder does them back to back. T19 is the UI on top of everything: it needs T18's model
download with progress and T20's sync/train endpoints, so it goes last. Then **Independent review #10** over all four.

**Two builders in parallel (lane A ∥ lane B) must not share a pytest temp dir** (pytest basetemp rotation deletes the other run's tmp dirs, see
RESUME). Lane A runs `uv run pytest --basetemp=.pytest_tmp_a …`, lane B `--basetemp=.pytest_tmp_b …` (add both to `.gitignore` and ruff
`extend-exclude`). Benchmarks (`scripts/benchmark.py`) are run by one lane at a time with the other lane idle, because throughput numbers
are compared at ±5 %.

## Budgets (every task; `scripts/benchmark.py --phase4` adds the lines, `--large 10000` the T21 run)
- Unchanged: **cull ≥ 10 img/s and within ±5 % of 25.1** on the 300 bench; peak RSS ≤ 3 GB whole process (cull + indexer + query + OCR +
  T15–T17); decisions identical warm == cold; review ≤ 20 %.
- New (T21, synthetic 10k shoot, M-series 16 GB): see the T21 table. New (T18/T19/T20): see each task.

---
## T18 — One-command install + launcher (lane A) · deps: none

**Goal.** A photographer with no terminal knowledge gets from a downloaded/cloned folder to the open UI with one double-click (or one
command), and removes it by deleting the folder.

**User story.** "I unzip ShotSieve into `~/ShotSieve`, double-click `Install ShotSieve.command`, wait a few minutes, then double-click
`ShotSieve.app`; my browser opens the ShotSieve page. Next time I just click the app in my Dock."

### Design
- `install.sh` (bash, `set -euo pipefail`, repo root; idempotent; re-running = repair/update). Steps, each printing one plain line and
  logging to `data/logs/install.log`:
  1. Platform check: `uname -s` = Darwin (else exit 2 "macOS only for now"); `uname -m` arm64 (x86_64 allowed with a "CPU fallback, slower"
     warning); macOS ≥ 13 (`sw_vers -productVersion`); free disk ≥ 3 GB in the repo volume (`df -k`), else exit 3 with the number.
  2. **uv bootstrap into the repo**: download the pinned release tarball
     `https://github.com/astral-sh/uv/releases/download/<UV_VERSION>/uv-<arch>-apple-darwin.tar.gz` with `/usr/bin/curl -fL --retry 3`,
     verify with `shasum -a 256` against the sha256 pinned in the script (builder takes it from the release's `.sha256` asset of the same
     version, e.g. 0.11.4 which is already in use; uv is MIT/Apache-2.0), extract to `.tools/uv/`. Never use a system uv (reproducible);
     `SHOTSIEVE_UV_TARBALL=/path` overrides the download (tests, offline installs).
  3. Every uv call runs with `UV_CACHE_DIR=$ROOT/.uv-cache UV_PYTHON_INSTALL_DIR=$ROOT/.uv-python UV_PYTHON_PREFERENCE=only-managed
     UV_PROJECT_ENVIRONMENT=$ROOT/.venv UV_NO_MODIFY_PATH=1 UV_TOOL_DIR=$ROOT/.tools/uv-tools`; then `uv python install 3.11` and
     `uv sync --frozen --no-dev --inexact` (`--inexact` so running it in a dev checkout does not uninstall pytest/playwright; `--dev` flag
     syncs dev deps too).
  4. Core models: `.venv/bin/shotsieve models download` (131 MB). **T18 pins `sha256` in every `ModelSpec`** (values below) and pins the
     opencv_zoo URLs to a commit instead of `raw/main` (builder resolves the commit that serves these exact bytes and records it with a
     `curl -sIL` check, HTTP 200 + size, in 02_MODELS.md). A mismatch deletes the file and fails with "download corrupted, run install again".
     `SHOTSIEVE_MODELS_SRC=/dir` copies (APFS clone `cp -c`) from a local folder instead (tests, offline installs; same sha check).
  5. `scripts/build_app.sh` → `$ROOT/ShotSieve.app`; `.venv/bin/shotsieve doctor` (new): Python 3.11, core models sha ok, data dir writable,
     free port in 8765–8775, disk ≥ 1 GB free → exit 0 or one plain sentence per failure.
  6. Final line: "Done. Open ShotSieve.app (drag it to your Dock to keep it)." With `--open`, open it.
- `Install ShotSieve.command` (repo root, `chmod +x`): `cd "$(dirname "$0")" && ./install.sh --open; read -n1 -p "Press a key to close"`.
  Finder runs `.command` files in Terminal, so there is nothing to type.
- `scripts/build_app.sh` writes a plain bundle (no third-party packer, **unsigned**; `codesign -s -` ad-hoc signing is optional, never
  required): `Contents/Info.plist` (`CFBundleExecutable=ShotSieve`, `CFBundleIdentifier=local.shotsieve.launcher`,
  `CFBundlePackageType=APPL`, `LSMinimumSystemVersion=13.0`, `CFBundleShortVersionString` = package version), `Contents/MacOS/ShotSieve`
  = `scripts/launcher.sh` with `ROOT` substituted as an absolute path at build time, `Contents/Resources/ShotSieve.icns` (drawn with
  Pillow into a PNG, converted with `sips` + `iconutil`). Built locally, so it carries no quarantine attribute.
- Launcher (`scripts/launcher.sh`, bash): if `$ROOT/.venv/bin/shotsieve` is missing → `osascript` dialog "ShotSieve's folder moved or is
  incomplete. Run Install ShotSieve.command again." and exit 1. Read `data/server.json` `{pid, port, version, started}`; if the pid is alive and
  `curl -fs http://127.0.0.1:$port/api/health` returns `"app": "shotsieve"` → `open http://127.0.0.1:$port/` and exit (no second server).
  Else pick the first free port in 8765–8775, start `.venv/bin/shotsieve serve --port $P --no-open` detached (`nohup … >> data/logs/serve.log 2>&1 &`),
  write `server.json` atomically, poll health every 0.2 s for up to 20 s, then `open` the URL (skipped when `SHOTSIEVE_NO_BROWSER=1`).
  On timeout: dialog naming `data/logs/serve.log`. Rotate `serve.log` at 5 MB (keep 1).
- Server: `/api/health` gains `"app": "shotsieve", "version": …, "pid": …`. New `POST /api/quit` (local Origin only) shuts uvicorn down
  gracefully after the response and removes `server.json` when its pid is ours; the UI "?" menu gets **Quit ShotSieve**. `serve` writes/removes
  `server.json` itself as well (so a terminal `shotsieve serve` is also found by the launcher).
- Footprint rule: install and launcher write **only inside `$ROOT`** (`.tools/`, `.uv-cache/`, `.uv-python/`, `.venv/`, `models/`, `data/`,
  `ShotSieve.app`). The shoot folders get `.shotsieve/` (and XMP if asked), as today; README "Uninstall" says: delete the ShotSieve folder; each
  shoot's `.shotsieve/` folder can be deleted too (it only holds ShotSieve's cache and decisions).

Pinned sha256 (from `models/manifest.json`, recorded 2026-11; builder re-verifies): yunet `8f2383e4…52fa4`, sface `0ba9fbfa…e79`,
face_landmarker `64184e22…c9ff`, dinov2_small `f22797ea…c20a`; extras: siglip2_vision `c0573e3f…bcad`, siglip2_text `3a0603d3…6400`,
siglip2_tokenizer `cb9140fa…8322`, siglip2_vision_int8 `5f2b401c…0070`, ppocr_det `ee40e800…b935`, ppocr_rec `4e16deb2…cd7`, ppocr_dict
`e025a66d…70d6` (take the full 64-hex values from the manifest).

### Acceptance
1. Sandbox install (automated, `tests/test_install.py::test_full_install_sandbox`, markers `install` + `network`): copy the repo without
   `.venv/.uv-*/models/data/.tools` to a temp dir (`cp -c` clones), `HOME=<tmp>/home`, `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, run
   `./install.sh` → exit 0; then `ShotSieve.app/Contents/MacOS/ShotSieve` with `SHOTSIEVE_NO_BROWSER=1` → health ok. Asserts:
   the `$HOME` tree (names, sizes, mtimes) is **identical before and after**; nothing new under `/Users/Shared`, `~/Library`; installed size
   (excluding `.uv-cache`) ≤ 1.5 GB; `.uv-cache` may be deleted afterwards and the app still starts.
2. Offline variant (no `network` marker, runs in the normal suite only with `-m install`): same but `SHOTSIEVE_UV_TARBALL`, `SHOTSIEVE_MODELS_SRC`,
   `UV_OFFLINE=1` with `.uv-cache`/`.uv-python` cloned from the dev checkout → exit 0 in ≤ 120 s.
3. Re-run on a complete install: no network access (prove with `UV_OFFLINE=1` and models from disk), exit 0 in ≤ 60 s, nothing re-downloaded.
4. Corrupt model (flip a byte) → re-run detects, re-downloads (or re-copies), sha ok. Wrong uv tarball sha → exit non-zero, nothing extracted.
5. Launcher: cold start to health ok ≤ 8 s (first launch after install ≤ 15 s); a second launch while running starts **no** second server
   (exactly one `shotsieve serve` pid) and returns in ≤ 2 s; port 8765 busy (test binds it) → server on 8766 and `server.json` says so;
   stale `server.json` (dead pid) is replaced; `POST /api/quit` → process exits ≤ 3 s, `server.json` gone; `ROOT` moved → exit 1 with the dialog
   text (dialog stubbed via `SHOTSIEVE_DIALOG=echo`).
6. `plutil -lint ShotSieve.app/Contents/Info.plist` ok; `bash -n` on every script; executable bits set; `shotsieve doctor` reports each failure
   case (models missing, port range busy, unwritable data dir) with exit 1.
7. **Manual, run by the user (not by the builder): a clean macOS user account** on a home connection, unzip → double-click
   `Install ShotSieve.command` → ≤ 10 min to the open UI, **zero prompts** other than macOS's own first-run Terminal permission (if any);
   record time, disk and prompts in TASKS STATUS. The builder must not create macOS accounts.

### Tests
`tests/test_install.py` (new markers `install` registered in pyproject; skipped by default unless `-m install`): plist/bash lint, build_app
structure and ROOT substitution, launcher behaviours 5 (real `.venv`, `SHOTSIEVE_DATA_DIR=<tmp>`), sandbox installs 1–4.
`tests/test_registry_runtime.py`: every `ModelSpec.sha256` is set, a mismatching file is rejected and re-fetched, `SHOTSIEVE_MODELS_SRC` copy
path. `tests/test_api.py`: health has `app`, `/api/quit` refuses foreign Origin, `serve` writes/removes `server.json`.

### Risks
- Gatekeeper: a **downloaded zip** is quarantined, so the `.command` may need right-click → Open once ("unidentified developer"). Mitigation:
  README one-liner; never strip quarantine automatically. The `.app` is built locally and is not quarantined. A `git clone` needs Xcode CLT
  (macOS prompts for it); README recommends the zip.
- uv / python-build-standalone download hosts (GitHub) unreachable → clear message + `SHOTSIEVE_UV_TARBALL` path. HF outage → models step fails
  with retry advice; `SHOTSIEVE_MODELS_SRC` for offline.
- Some wheels (mediapipe, opencv-contrib) are large; the 10-min target assumes ≥ 25 Mbit/s (~500 MB total download). Record the measured size.
- A user drags `ShotSieve.app` to /Applications: the absolute ROOT still works; deleting the repo folder leaves a dead app → the dialog explains.

### Out of scope
Signing/notarisation (needs an Apple developer account: user decision), auto-update, Windows/Linux packaging, launch-at-login, menu-bar app,
a bundled Python inside the .app.

---
## T21 — 10k-frame shoots (lane B) · deps: none (independent of T18)

**Goal.** 5–10k-frame weddings feel instant: the grid opens fast, adding a second card re-culls only what changed, and nothing grows past
the 3 GB budget.

**User story.** "I culled 9,500 frames last night. This morning I copy the second shooter's card (500 frames) into the same folder and press
Cull: it's done in under a minute, and none of the picks I already reviewed moved."

### Design
1. **Incremental re-cull with stable decisions** (`runner.py`). Extraction is already incremental (`lookup_many`). What is not: every
   re-run re-scores the whole shoot with shoot-global quantities, so adding files can flip old frames. Change:
   - Persist `meta.shoot_norm = {lap_ref, tenen_ref, face_ref, af_ref, n, extractor}` after a full decide. On a re-run where
     `n_new + n_removed ≤ 20 %` of the stored `n` and the extractor fingerprint is unchanged, reuse the stored norm; otherwise recompute,
     store, and set `stats.redecide = "full"` (reason in `stats.redecide_reason`). Stats always report `n_new`, `n_removed`, `n_extracted`,
     `n_affected`.
   - **Affected set** A = frames in a duplicate group or burst that contains a new/removed frame (groups computed after the change, plus the
     pre-change groups of any of their members), plus frames on the same body within `max(dup_window_s, flash_window_s)` of a new/removed frame.
     Frames outside A keep their stored decision row **verbatim** (decision, reasons, score, is_best, group membership); grouping and scoring
     still run on the whole shoot (they are cheap, see table) but the result for frames outside A must equal the stored one; the runner
     asserts this in debug mode and logs a count otherwise.
   - Known shoot-global couplings the builder must handle explicitly: `compute_norm` (frozen as above), `dup_review_budget < 1` after
     calibration (the cut is shoot-wide: allowed count recomputed, but frames outside A keep their status; overflow only within A), clock
     offsets (`clock.analyse(auto_apply=True)` changing an offset → full re-decide, reported), the frozen audit sample (new frames never enter it
     until `redraw_audit`), personal model / thresholds (unchanged within a run). Group ids may renumber; the API keeps `group_id` stable for
     unaffected groups (map new → old id by member overlap).
   - `save_decisions` becomes an upsert of changed rows plus delete of removed paths (not DELETE-all + insert) inside one transaction.
2. **Thumbnails at cull time** (decode once): while the preview is decoded for extraction, a worker writes the 400-px grid thumbnail
   `thumbs/t_{id}_400_{mtime}.jpg` (the exact name `_image_file` serves; ids assigned via `ids_for` before extraction). Cost ≤ 5 % of cull
   throughput (300 bench stays ≥ 23.8 img/s, i.e. within the ±5 % rule). Missing thumbs are still rendered on demand as today.
3. **Fast shoot payload** (`server/app.py`): `build_shoot` at 10k ≤ 1.0 s server time; no per-request `INSERT` (move the `ids_for` insert to
   cull time; the GET only reads, M5); reason texts cached per (code, ctx) or computed lazily by `/api/image/{id}`; `GZipMiddleware(minimum_size=4096)`;
   payload ≤ 8 MB raw / ≤ 1.5 MB gzip at 10k. Thumb endpoint stays sync (threadpool) and sets `Cache-Control: max-age=86400, immutable`
   (name carries mtime).
4. **Grid at 10k** (`static/app.js`): layout/filter recompute ≤ 100 ms at 10k (measure `performance.now()` around `layout()`), no O(n²)
   (e.g. `indexOf` inside loops over images); thumbnails for the visible window + 1 screen are requested first (in viewport order), older
   requests are aborted on fast scroll (`img.src = ""` / `AbortController` for fetches); DOM stays bounded as in the 5k test.
5. **Memory**: runner keeps `CachedImage` per frame (scene 384 f32 + faces + RawFeatures); at 10k that is ≲ 150 MB. `lookup_many` reads in
   chunks of 2,000 rows instead of one `fetchall` over the 4-table join.

### Acceptance (synthetic 10k shoot `data/bench/shoot_10000_7`, `benchmark.py --large 10000`; 16 GB M-series)
| Metric | Budget |
|---|---|
| Cold full cull of 10k | ≥ 10 img/s (≤ 1,000 s), peak RSS ≤ 3 GB (cull only ≤ 1.3 GB) |
| Warm re-cull, nothing changed | ≤ 15 s wall, 0 frames extracted, decisions identical |
| First grid, warm (server running, shoot culled, thumbs from cull) | ≤ 2.0 s from navigation to every visible thumb loaded (Chromium, 1500×900) |
| First grid, thumbs not yet rendered (`thumbs/` deleted) | ≤ 5.0 s, same definition |
| `GET /api/shoot` at 10k | ≤ 1.0 s server, ≤ 1.5 MB gzip |
| Add 500 files (450 as a new second-shooter block + 50 interleaved in time) | exactly 500 extracted; wall ≤ 500 / measured img/s + 20 s; frames outside A: decision, reasons, score, is_best identical (100 %); `n_affected` reported |
| Remove 200 files | 0 extracted, ≤ 15 s, their rows gone, frames outside A identical |
| > 20 % new files | `redecide = "full"`, reported in stats and the UI toast |
| UI at 10k | per scroll step < 0.10 s (was 0.25 at 5k), ≤ 400 `.cell`, filter switch ≤ 150 ms, no console errors |
| Server RSS while serving the 10k grid | ≤ +300 MB over idle |
| 300 bench | unchanged: ≥ 10 img/s and within ±5 % of 25.1, decisions identical warm == cold |

### Tests
- **Cheap 10k synthesis for pytest** (`tests/large_synth.py`, no models, ≤ 20 s): write N tiny JPEGs (64×43, ~1 KB, distinct EXIF
  `DateTimeOriginal` laid out as bursts/moments like `benchmark.synthesize`, 2 camera serials), then populate `cull.db` directly with
  `Cache.put(...)` using the files' real `(size, mtime_ns)`, the real `extractor_fingerprint(settings)`, synthetic `RawFeatures` (sharpness drawn
  per burst, a few blurred/dark), and scene embeddings as unit vectors = moment centroid + small noise (so `group_duplicates` forms realistic
  groups). `run_cull` is then fully warm and model-free. New files for the incremental tests go through a monkeypatched
  `runner._process_uncached` that yields synthetic jobs and **counts the paths it was given**.
- `tests/test_large.py` (marker `large`, in the default run): warm 10k run (no extraction, identical decisions, wall ≤ 15 s); add 500
  (extraction count == 500, stability outside A); remove 200; > 20 % → full; norm reuse rule; `save_decisions` upsert; `build_shoot` ≤ 1.0 s
  and gzip size; GET `/api/shoot` performs no write (sqlite `total_changes` unchanged).
- `tests/test_ui.py`: the 5k virtualised test parametrised to 10,000 with the tighter step budget; a first-grid timing test against a real
  server on the `large_synth` shoot (1-KB thumbs pre-rendered; asserts ≤ 2.0 s; the ≤ 5.0 s cold-thumb number is a benchmark line, not a
  pytest assert, because 64-px sources are not representative).
- `scripts/benchmark.py --large 10000`: synthesize (reuse if present; ~4.6 GB at ~460 KB/frame, ~10 min once; gitignored), cold cull,
  warm re-cull, +500/−200 incremental (copies 500 extra synthesized frames with a second serial and a +2 h offset into the folder), first-grid
  timing via Playwright (both thumb states), RSS per stage → `data/benchmark.md` T21 section. `--phase4` appends the T18–T20 lines.
- Regression: `tests/test_warm_cold.py` still identical; existing grouping/scoring tests unchanged.

### Risks
- Freezing the norm means a re-cull after a big change differs from a fresh cull of the same folder. Accepted and reported (`redecide`);
  `cull --full` forces a recompute.
- Stability vs `dup_review_budget` and clock auto-apply (above): the builder lists every place a shoot-global quantity is read; the stability
  test must fail on today's code first.
- Playwright timing on a loaded machine: run the UI timing test with nothing else heavy running; benchmark is authoritative.
- 4.6 GB bench folder: check free disk first (`df`), refuse below 10 GB free.

### Out of scope
Multi-folder "projects" and cross-shoot search, a native thumbnail cache outside the shoot, GPU decoding, 50k+ shoots, server-side pagination
(only if the gzip/1-s budget cannot be met; then the reviewer decides).

---
## T20 — Lightroom Classic sync-back (lane B) · deps: T21 (same files)

**Goal.** What the photographer finally picked in Lightroom flows back into ShotSieve as overrides, so the personal model and per-shoot
calibration learn from real use, and a re-cull never fights the user.

**User story.** "I culled in ShotSieve, imported into Lightroom, and fixed a few picks there (flags, stars). I click *Sync from Lightroom*:
ShotSieve tells me it missed 3 keepers out of 412 and sent 9 % to review; next week it does better."

### Safety rules (hard requirements; each has a test)
1. **The user's catalog is never written.** The only open of the user's file is `sqlite3.connect("file:<abs>?mode=ro", uri=True)`
   (path URI-escaped with `urllib.parse.quote`), never a plain path, never `immutable=1` on the user's file (immutable skips the lock and the
   `-wal`, so it can read a torn page or miss Lightroom's latest changes). No `PRAGMA` that writes, no `ATTACH`, no checkpoint.
2. **Locked / WAL / read-only directory → copy first.** On `SQLITE_BUSY`/"locked", or `SQLITE_CANTOPEN` / "unable to open" (WAL catalog in a
   read-only dir without `-shm`), copy `.lrcat` + `-wal` (+ `-shm`) into a private temp dir under `data/tmp/` (APFS `cp -c` clone when possible,
   else `shutil.copy2`), open **the copy** normally (so its WAL is folded into the copy), run `PRAGMA quick_check`; if it fails, retry the copy
   up to 3× 0.5 s apart, then refuse: "Lightroom is writing the catalog; quit Lightroom and sync again." The temp dir is deleted in `finally`.
   This extends the existing `_read_locked` (today it only reacts to "lock"/"busy").
3. The original's bytes and its sidecar files are untouched: tests assert sha256 + mtime of `.lrcat`, `-wal`, `-shm` (and their presence or
   absence) before == after, for every case (plain, WAL with un-checkpointed frames, locked by another connection, chmod 0444 file, read-only dir).
4. The UI recommends "Quit Lightroom before syncing" (not required). `.lrcat` paths must be absolute files ending `.lrcat`; `..`/NUL refused.

### Design
- **Reader** (`io/lrcat.py`): `read_catalog` gains optional columns when present (`PRAGMA table_info`): `Adobe_images.touchTime` (seconds since
  2001-01-01 UTC → Unix), `captureTime`, `AgLibraryFile.originalFilename`, `sidecarExtensions`; `CatalogEntry` gets `touch_ts: float | None`,
  `capture: str | None`, `id_local: int`. A `folder_prefix` filter pushes `WHERE rf.absolutePath || f.pathFromRoot LIKE ?` into SQL so a
  100k-image catalog is not fully materialised.
- **Baseline of ShotSieve's own output** (`cache.py`, new table): `lr_written(path PK, rating INTEGER, label TEXT, decision TEXT, ts REAL)`,
  upserted wherever `write_decision` actually writes a rating/label (runner write phase, `/api/xmp/write`). Backfilled once from sidecars'
  `shotsieve:WrittenRating/WrittenLabel` when the table is empty. This is what Lightroom imported from ShotSieve, so equal values are
  ShotSieve's opinion, not the user's.
- **Sync state** (`cache.py`, new table): `lr_sync(path PK, lr_id INTEGER, rating INTEGER, pick INTEGER, label TEXT, touch_ts REAL,
  synced_at REAL)` = last seen Lightroom state; `meta.lr_sync_last = {catalog, ts, n_catalog, n_matched, n_unmatched, n_new, n_changed,
  n_conflicts_ui_won, n_conflicts_lr_won}`; `meta.lr_catalog` remembers the path.
- **Matching** (`pipeline/lrsync.py`): for each decided master path in cull.db: exact `norm_path` match; else fallback key
  (`casefold(basename)`, capture time to the second) unique in the catalog subset (covers LR "Copy" imports to another folder); RAW+JPEG pairs
  match on the RAW master (LR lists the JPEG in `sidecarExtensions`). Unmatched counted, never guessed. Video frames (`#t=`) skipped.
- **User signal and mapping** (order: pick > rating > label; `min_rating` from Settings):
  | Lightroom state (vs `lr_written`) | Override decision |
  |---|---|
  | `pick = -1` (rejected flag) | reject |
  | `pick = 1` (picked flag) | keep |
  | rating ≥ min_rating and ≠ written rating | keep (rating stored) |
  | rating lowered below min_rating from a written keep rating, no flag | reject (note `lrcat:downgrade`) |
  | label Red ≠ written label | reject |
  | label Green ≠ written label | keep |
  | everything equal to what ShotSieve wrote, or no signal | **no override** (agreement is not a label: never learn from our own output) |
  Label names come from `Settings.lr_label_map` (default `{"Red": "reject", "Green": "keep"}`; others ignored), because Lightroom stores
  the label-set *name* (localised/custom sets differ).
- **Precedence / idempotency.** New overrides are written with `note="lrcat"`, `propensity=1.0`, `rating=lr rating`, `ts=touch_ts or now`.
  An existing `lrcat` override is replaced only when the LR state differs from `lr_sync` (the user changed it again in LR). An existing UI
  override (`note="ui"`): the later one wins (`touch_ts` vs override `ts`; without `touch_ts`, LR wins only if its state changed since the last
  sync); every conflict is counted and listed. Running sync twice with no LR change → 0 writes (`total_changes` unchanged apart from meta).
- **Re-cull and XMP never fight Lightroom.** Overrides already survive re-culls (separate table); additionally `write_decision` skips the
  rating/label of any image with an `lrcat` override (LR owns those values; counted as `kept_lr`), with or without `--force-ratings`.
- **Training uses it.** `labels_from_overrides` already reads every override (`source="override"`); `build_training_set(lrcat=…)` additionally
  **drops catalog ratings/labels equal to `lr_written`** for images under culled folders (today `train --lrcat` after an XMP round-trip
  learns ShotSieve's own 3★/Green as user keeps: a real bug, fix and regression-test it). `shotsieve train --from-overrides FOLDER` and the
  T19 train step pick up synced rows. Recalibration: `audit/complete` and `calibrate_shoot` already join the frozen audit sample with
  overrides, so synced audit items count; test it.
- **"This shoot" report** (reuses `pipeline/evaluate.evaluate`): truth = the user's final LR state for matched frames (kept = pick 1 or
  rating ≥ min_rating, not rejected; frames with no user signal and no flag in a folder where the user picked something count as not kept,
  as in `truth_from_catalog`), predictions = the stored **model** decisions (not overrides). Output: group gate rate, frame missed-keeper
  rate, review fraction, keepers in review, reject precision, n evaluated, plus the missed file names. Stored in `meta.lr_report`.
- **CLI** `shotsieve sync FOLDER --lrcat CAT.lrcat [--dry-run] [--json] [--no-report]` → table of counts + report; exit 0, 2 on bad args,
  4 when the catalog is busy and the copy fails the integrity check.
- **API** `POST /api/lr/sync {folder, catalog, dry_run}` → job (kind `lr_sync`, under the one-job lock; progress stages `read|match|write|report`);
  `GET /api/lr/report?folder=` (read-only); `GET /api/lr/catalogs` (auto-detect `~/Pictures/Lightroom/**/*.lrcat` depth ≤ 3, plus
  `meta.lr_catalog`; listing only). **UI**: "Sync from Lightroom" button (topbar "…" menu or the Lightroom card), dry-run summary first
  ("12 changes: 9 keeps, 3 rejects; 2 conflicts, your ShotSieve choice kept"), then Apply; report card with the numbers; grid badges show
  `LR` on synced overrides.

### Acceptance
1. All safety tests (rule 3) green for: plain catalog, WAL catalog with ≥ 1 un-checkpointed change (the change is seen), catalog held with
   `BEGIN EXCLUSIVE` by another process (copy path taken, committed data seen), chmod 0444 + read-only dir, corrupted copy (monkeypatched)
   → exit 4 / refusal. A spy on `sqlite3.connect` proves every connection to the user's path is a `mode=ro` URI.
2. Fixture round-trip (`tests/test_lrsync.py`): cull the fixture shoot with `--write-xmp` → `lr_written` filled → build a catalog of that
   shoot whose ratings/labels equal what ShotSieve wrote → simulate LR edits on 12 frames (4 pick=1 on rejects, 3 pick=-1 on keeps, 2 stars
   added, 1 star removed from a keep, 1 Red, 1 Green; `touchTime` bumped) → sync → **exactly those 12 overrides**, decisions per the mapping
   table, 0 overrides for untouched frames. Second sync → 0 writes. Re-cull (with and without `--write-xmp --force-ratings`) → the 12
   overrides unchanged, their sidecar rating/label unchanged (sha), stored model decisions refreshed.
3. Conflicts: a UI override newer than the LR touch survives; an older one is replaced; counts reported.
4. Copy-import fallback: catalog paths under a different root with the same file names and capture times → ≥ 95 % matched on the fixture
   (100 % expected), ambiguous names (same basename + time on two bodies) left unmatched.
5. Training: after sync, `train --from-overrides` includes the 12 rows; `train --lrcat` on a catalog equal to ShotSieve's output gives **0**
   user labels from those frames (regression test fails on today's code).
6. Report: on a hand-made truth (known missed keepers) the numbers equal `evaluate.evaluate` on the same inputs; written to `meta.lr_report`.
7. Perf: synthetic catalog of 100k images (built with `executemany` in ≤ 5 s) + 10k-frame `large_synth` shoot: sync ≤ 3 s without copy;
   with the locked-copy path on a 1 GB catalog ≤ 10 s; RSS +≤ 200 MB.
8. Playwright: dry-run → apply → badges and report card render, no console errors.

### Test fixtures
`scripts/make_fixtures.build_lrcat` gains optional columns (`touchTime`, `captureTime`, `originalFilename`, `sidecarExtensions`) and a
`wal=True` mode; a test helper `lr_edit(cat, path, *, pick=None, rating=None, label=None)` updates rows the way Lightroom would and bumps
`touchTime` (always on the test's own copy). Helpers for the lock case run a child `python -c` that holds `BEGIN EXCLUSIVE` until signalled.
No real Lightroom catalog is committed; the reader's column handling stays tolerant (`PRAGMA table_info`), matching how T15 tests work.

### Risks
- Real catalogs differ from the fixture schema (column types, `colorLabels` text, `pick` as REAL). Mitigation: optional columns, cast in
  Python, and the user's own catalog is the first real-world check (dry-run first). The builder documents the assumed columns in 01_ARCHITECTURE.
- Lightroom "Copy as DNG" imports change extension and file name; the fallback key does not cover renames → reported as unmatched.
- Learning from LR picks has selection bias (the user only looked at what ShotSieve kept): propensity stays 1.0 here; the audit sample remains
  the unbiased signal for calibration. Documented in the report card.
- "Downgrade → reject" may be too strong for photographers who remove stars casually; it is one table row and easy to soften after real data.

### Out of scope
Writing anything to Lightroom (no catalog writes, no plugin), Lightroom CC/cloud catalogs, Capture One sessions, syncing develop settings back
(T15 owns edits), automatic background sync.

---
## T19 — First-run onboarding in the UI (lane A) · deps: T18 (download progress, health), T20 (sync/catalog endpoints, for the LR card)

**Goal.** The first shoot succeeds without reading docs or touching a terminal.

**User story.** "The page opens with a short setup: it downloads what it needs, I choose my wedding folder, optionally point it at my
Lightroom catalog, decide about face recognition, press Cull, and it tells me exactly what to click in Lightroom afterwards."

### Design
- Wizard = full-screen overlay in a new `static/onboarding.js` (keeps `app.js` from growing; loaded after it, uses its `api`/`toast`).
  Shown when `GET /api/onboarding` says not completed **or** core models are missing; "Skip setup" always visible; reopen via "?" → "Setup
  assistant". State in `data/onboarding.json` `{version: 1, completed_at, skipped, last_step}`. ≤ 6 screens, each one primary button:
  1. **Welcome + models**: if core models are missing, "Download (131 MB)" → `POST /api/models/download {groups:["core"]}` → job with byte
     progress, MB/s and ETA; failure → plain message + Retry (resumes the `.part` with an HTTP `Range` request).
  2. **Shoot folder**: in-app folder browser (`GET /api/fs/list?dir=`: sub-directories with image counts, restricted to `$HOME` and `/Volumes`,
     hidden dirs skipped, never file contents), a "Choose…" button using the native dialog (`POST /api/fs/pick {kind:"folder"}` → `osascript -e
     'POSIX path of (choose folder)'`, 120 s timeout, cancel → no change), paste a path, recent folders (`data/recent.json`, last 10). After a
     choice: `GET /api/fs/scan?folder=` → `{n_images, by_format: {raw, jpeg, dng, heic, video}, culled: bool}` shown in one line.
  3. **Lightroom catalog (optional)**: auto-detected list (`GET /api/lr/catalogs`) + browse (`.lrcat` only); "Train my personal model" →
     `POST /api/train {catalog, xmp_folders, from_overrides}` → job kind `train` with **per-image** progress and ETA (featurise passes a
     progress callback into `run_cull` instead of counting folders); result line "Learned from 1,240 of your picks" or "Not enough picks yet
     (need 40); ShotSieve uses its built-in judgement". Skip is fine.
  4. **Privacy & extras**, three explained toggles, all **off by default**: *Recognise people* (face clusters stay in this shoot's folder;
     delete any time) → `POST /api/people/consent`; *Search by description* (download 690 MB, indexing runs in the background after the cull);
     *Read jersey numbers* (download 10 MB). Sizes come from `GET /api/models` (bytes per group from `MODEL_SPECS`, installed or not).
  5. **Cull**: "Write ratings for Lightroom" checkbox (on when the shoot has RAW files) → existing `POST /api/cull` + progress with img/s and ETA.
  6. **Lightroom step card** (also reachable later from the "?" menu), built from `by_format` and whether XMP was written:
     RAW: "In Lightroom Classic: *Add* this folder in Import (ratings and colour labels come with it). Already imported? Select all →
     Metadata → Read Metadata from Files." JPEG/DNG/HEIC: "Lightroom does not read ShotSieve's ratings for these files. Review them here (Keep
     tab); after you finish in Lightroom, use *Sync from Lightroom* so ShotSieve learns." Plus the Sync button (T20).
- **Download lane**: model downloads are network-bound, so they run as a separate single-slot lane (`state.download`), not under the CPU job
  lock: a cull can run while search models download. `ensure_models(..., progress=cb)` reports bytes (Content-Length) per file and total.
- New endpoints, all honouring M5 (GET never writes, POST only from the local Origin): `GET/POST /api/onboarding`, `GET /api/models`,
  `POST /api/models/download`, `GET /api/fs/list`, `POST /api/fs/pick`, `GET /api/fs/scan`, `POST /api/train`. `fs` paths: absolute, no `..`/NUL,
  `realpath` must stay under `$HOME` or `/Volumes` (symlinks out are refused), results capped at 500 entries.
- Accessibility: every control reachable with Tab/Enter, `aria-live` on progress, visible focus; works at 1280×800 and 1500×900.

### Acceptance
1. **Playwright end-to-end** (`tests/test_ui.py::test_onboarding_full_flow`, fresh `SHOTSIEVE_DATA_DIR` and models dir; the server's
   `_download` is monkeypatched to stream from the real local model files in 1-MB chunks with a short sleep, so progress is observable and no
   network is used): wizard appears → core download reaches 100 % (≥ 3 distinct progress values seen) → folder chosen in the fs browser (a temp
   copy of the fixture shoot) → scan line shows the right counts → catalog step with `catalog.lrcat` (fixture) → train job finishes and shows the
   result line → people toggle on (`people_consent` true for that folder via API) → cull → grid shows every fixture image → Lightroom card
   lists the shoot's formats → reload: wizard not shown. **Zero console errors, zero unexpected 4xx/5xx**, test ≤ 120 s.
2. Click budget: with models present, from first page load to "cull started" ≤ 5 clicks when the folder is picked from Recent (counted in the test).
3. Skip path: "Skip setup" → normal UI, `skipped: true`; reopen from "?" works. Failure path: a download raising `URLError` → message + Retry;
   Retry resumes from the `.part` (Range header seen in the stub).
4. Concurrency: search-model download running while a cull runs: both finish; the cull lock still blocks a second cull (409).
5. API tests (`tests/test_api.py`): `/api/fs/*` refuse `..`, NUL, relative, symlink escaping `$HOME`, non-directories; list never returns
   non-`.lrcat` files; GETs leave `data/` unchanged (mtime snapshot); `/api/fs/pick` returns 501 off macOS and is stubbed via
   `SHOTSIEVE_PICKER=<script>` in tests; `/api/train` job lifecycle and the < 40 rows message; `GET /api/models` sizes equal `MODEL_SPECS`.
6. No regression: all existing Playwright tests pass with the wizard suppressed (`onboarding.json` completed in their fixture).

### Risks
- Browsers cannot give absolute paths from `<input type=file>`; hence the server-side browser + `osascript` picker. `osascript` may trigger a
  macOS "Terminal/python wants to control…" prompt the first time; the browse list works without it.
- Training a large catalog takes minutes (it runs the cull stages on every labelled folder): show the ETA, allow "continue in background",
  and never block step 4–5 on it (the job lock means the cull waits for training: the wizard says so and offers "Skip training for now").
- People consent semantics (per-folder, default off) must not change; the toggle only calls the existing endpoint.

### Out of scope
Editing-profile training in the wizard (stays in the Edit sheet), multi-language UI, tutorials/video, telemetry of any kind.

---
## After Phase 4
reviewer **Review #10** over T18–T21 (lane B's stability test must be shown failing on the pre-T21 runner; the T20 safety tests must be shown
failing against a deliberately non-`mode=ro` reader). The clean-account install (T18 acceptance 7) is run by the user. Album sequencing
can be revisited with real T20 pick data.

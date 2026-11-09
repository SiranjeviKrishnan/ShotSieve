# Next work: two clean-up batches, then Phase 4 (2026-11-08)

State: MVP, Phase 2 and Phase 3 all **SHIP** (Review #8, 583 passed, ruff clean). Below are the open minors from Reviews #4/#5 plus two known gaps, grouped so that **one builder per batch** touches one set of files. Each batch ends with: full `uv run pytest -q`, `uv run ruff check .`, `uv run python scripts/benchmark.py --n 300` (cull ≥10 img/s and within ±5 % of 24.1, RSS ≤3 GB, decisions identical warm == cold), plus a tick in TASKS.md and an updated STATUS. Then one reviewer **Review #9** covers both batches.

General rules (unchanged): every fix gets a regression test that **fails on the old code first** (say so in the Resolution table). No new models. Permissive licences only. Nothing is installed outside the repository folder.

---

## Batch 1: correctness of written data and consent (one builder, "store")
Files: `pipeline/cache.py`, `pipeline/people.py`, `io/xmp.py`, `pipeline/search.py`, `personal/dataset.py`, `cli.py` (train table only). Tests: `test_people.py`, `test_xmp.py`, `test_jersey.py` (or `test_search.py`), `test_personal.py`/`test_cli.py`.
Priority order: r2 > r3 > r4 > m3.

### [x] B1.1 r2: consent epoch check and face writes in one `BEGIN IMMEDIATE` (cross-process)
Problem: `people_lock` is a per-process RLock (cache.py:810). Today `people_write` (cache.py:821) reads epoch + consent, then the caller writes and commits separately. A CLI `shotsieve index` running in another process than `shotsieve serve` can therefore write face rows after the server revoked consent.
Do:
- `people_write(epoch)` opens `BEGIN IMMEDIATE` on the Cache connection, **re-reads `people_epoch` and `people_consent` inside that transaction**, yields `ok`, then COMMITs (or ROLLBACKs on exception / `ok=False`). Callers in people.py:70/84 and `put_jerseys` (cache.py:732) must not commit on their own inside the block.
- `revoke_people()` runs its consent=false write, the epoch bump and `wipe_people()` in **one** `BEGIN IMMEDIATE` transaction.
- `set_meta`/`get_meta`/`wipe_people` currently call `commit()`. Give them a "inside a transaction, don't commit" path (e.g. a `self._in_tx` flag), or the transaction gets split in the middle.
- Set `PRAGMA busy_timeout` (≥5000 ms) on connect (cache.py:136) so that the second writer waits instead of failing with `database is locked`. Keep the in-process RLock as well (threads).
Acceptance:
- Two-**process** test (multiprocessing `spawn`, synchronised with a `multiprocessing.Event`, no sleeps as the mechanism): process A reads the epoch and then blocks; process B calls `revoke_people()`; A is released and tries `cluster()`/`finalize()`/`put_jerseys(epoch=…)`. Result: 0 rows in `people`/`person_faces`, every `jerseys.face_idx/person_id` NULL, `people_consent` False. The control run without the revoke writes rows.
- Reverse order: A is inside `people_write` (transaction open) when B revokes. B waits (busy_timeout), then wipes, so the end state is still 0 face rows.
- Running all the existing test_people.py revoke tests (mid-cluster, mid-finalize, DELETE, consent-off, indexer) still passes. Bench unchanged.

### [x] B1.2 r3: XMP unreadable-sidecar backup, one `.bak` per content hash
Problem: xmp.py:172-188 writes `<name>.xmp.bak-<timestamp>` on **every** call, so the files grow with every cull.
Do: reuse the hash scheme in crs.py:324 (`.bak-<sha256[:12]>`, write only if it is missing). Move it into one shared helper and call it from both places. Leave old timestamp backups alone.
Acceptance (test_xmp.py): a broken sidecar run through `write_sidecar` ×2 + `merge_keywords` + `write_decision` gives **exactly 1** `.bak-*`, the original bytes are unchanged, `skipped_unreadable=True`. Changing the broken bytes and running again gives exactly 2 backups. Existing `.bak-<timestamp>` files are not touched or counted.

### [x] B1.3 r4: `person:"Name"` roster search honours the team
Problem: search.py:66-74 turns a roster name into its jersey number(s) without the team, so `person:"Home Jane"` also returns away-#23 frames.
Do: resolve a name to `(team, number)` pairs. Use the same rule the tagging uses (roster M6): a number matches in a folder only if the folder team (`cache.roster_team()`) equals that pair's team. If there is **no folder team and the number is on more than one team**, it is ambiguous: return no frames for it and add a warning (`"#23 is on 2 teams; set the folder team"`) in the `/api/search` response, the CLI stderr and a UI notice. If the number is on exactly one team, it matches as it does today.
Acceptance (test_jersey.py): a two-team roster with home Jane #23 and away Bob #23, and frames tagged #23.
- Folder team Home: `person:"Jane"` returns the #23 frames, `person:"Bob"` returns none.
- No folder team: both return none plus the warning.
- A single-team roster with no folder team returns the #23 frames (no regression).
- Case-insensitive team match (`home`/`Home`).
- Without consent this still never touches face data (assert that `image_ids_for_person` is not called).

### [x] B1.4 m3: train says what happened to video-frame overrides
Problem: dataset.py:188-192 counts `clip.mp4#t=12.3` overrides as "labelled files missing on disk".
Decision: **exclude them explicitly** and do not train on them (frame grabs have no EXIF, flash or AF data and a different blur profile, so they would skew a photo model). Add `n_video` to the training set and a CLI row `video frame labels (not used for training) | N`. `n_missing` no longer counts them.
Acceptance (test_personal.py / test_cli.py): overrides on 3 video frames + 5 photos → `n_video=3`, `n_missing=0`, 5 rows, and the CLI table shows the row. Labels whose base clip has been deleted still count as video, not as missing.

---

### Batch 1 resolution (builder, 2026-11-08)
- r2: `Cache._immediate()` (re-entrant `BEGIN IMMEDIATE`, `_in_tx` flag; `_commit()` is a no-op inside it) used by `people_write`, `revoke_people`, `wipe_people`, `put_jerseys`; `PRAGMA busy_timeout=10000`. Tests `tests/test_people_xproc.py` (spawn + Events). The writer-holds-transaction ordering fails on the old code; the epoch-read-then-revoke ordering and the control also pass on the old code (the old code already re-checks the epoch after the release), they are kept as guards.
- r3: `xmp.backup_by_hash` shared by xmp.py and crs.py (old `crs._backup_foreign` removed); old test that demanded a backup per call replaced.
- r4: `search(..., warnings=[...])`; `/api/search` returns `warnings`; CLI prints `warning:` on stderr; UI toast shows it.
- m3: `TrainingSet.n_video`, CLI row, video labels are skipped before the missing-file check.
- Deviations: (1) `people_write` commits on a clean exit even when `ok` is False (such a block writes nothing by contract) instead of an explicit ROLLBACK; (2) the two-process test uses one bounded `Event.wait(1.0)` in the "B is blocked" ordering to prove the revoke is waiting; (3) the m3 and r4 regression tests were verified failing by construction (new attribute/kwarg), not by running them against reverted code.

## Batch 2: memory and real-data validation (one builder, "engine")
Files: `pipeline/indexer.py`, `models/text_image.py`, `models/ocr.py`, `features/jersey.py` (only if the det size changes), `scripts/benchmark.py`, `tests/test_indexer.py`, `tests/test_makernotes.py`/`test_af_focus.py`, `tests/test_video.py`, `tests/fixtures/real/` (+ LICENSE txt), `scripts/make_fixtures.py` (download entries, pinned sha256).
Priority order: m6 > OCR memory > exiftool stub > m1 > closed-eye video.

### [x] B2.1 m6: release SigLIP while the indexer is paused
Do: in `_wait_if_paused` (indexer.py:148), on entering a pause (cull gate or user pause) call the same release as `_release_models()` (indexer.py:186): `_embedder=None`, `release_vision()`, `release_ocr()`, `gc.collect()`. `_get_embedder()` (indexer.py:201) already reloads lazily on resume. Release once per pause, not once per 0.1 s loop.
Acceptance:
- test_indexer.py: the indexer runs a batch, the cull gate is set, and within 1 s the vision session and OCR sessions are gone (assert through the module globals and the `_embedder` hook). After the gate is cleared, indexing finishes and the stored embeddings are **bit-identical** to an uninterrupted run.
- Benchmark: add a stage "cull with indexer paused mid-run" whose RSS is ≤ cull-only + 100 MB (today it is about +1.5 GB when SigLIP stays resident). Record it in benchmark.md and STATUS.
- Index drain still ≥5 img/s. Cull throughput unchanged.

### [x] B2.2 OCR memory: +200 MB isolated vs 120 MB budget
The CPU arena is already off (ocr.py:34). Do, in this order:
1. Add `scripts/bench_ocr_rss.py`, a reproducible isolated-process measurement (fresh process, load det+rec, run 50 bench frames, report peak RSS minus baseline). Record the number before any change.
2. Set `so.enable_mem_pattern = False` on the det session (dynamic shapes). Then try a det long-side limit of 736 instead of 960.
3. Rule: keep a change only if the synthetic labelled set stays **P ≥ 0.95 / R ≥ 0.70** with the same outcomes on the 3 CC0 photos (no new miss), and OCR indexer speed stays ≥8 img/s.
4. If no setting that keeps accuracy reaches ≤120 MB: keep the best setting, report the measured number, and ask Review #9 to **amend the budget** to that number. Precedent: the HEVC +350 MB ruling. Whole-process headroom under 3 GB is over 600 MB.
Acceptance: the script's number is in STATUS (OCR row), plus either ≤120 MB or an explicit amend request with the accuracy table.

### [x] B2.3 exiftool AF fallback: test it without installing exiftool
`which exiftool` finds nothing. **Do not install it.** `brew install` writes outside the repository folder. exiftool is also Perl-licensed (Artistic/GPL dual), not MIT/Apache, so it must stay an optional user-installed tool and never become a dependency. The user may install it themselves; the code already finds it on PATH.
Do: add a test-only fake `exiftool` (a small Python script in `tests/fixtures/bin/`, put first on PATH through `monkeypatch.setenv("PATH")`). It speaks the `-stay_open True -@ -` protocol used by makernotes.py:168-205 and prints canned `-j -n` JSON followed by `{ready}`.
Acceptance (test_makernotes.py):
- (a) Canon canned values (AFImageWidth/Height, area W/H/X/Y, AFPointsInFocus "1,3") → the expected normalised points, including the centre-origin/Y-up conversion.
- (b) Sony `FocusLocation "6000 4000 4500 1000"` → (0.75, 0.25).
- (c) Malformed JSON or a crashed process → `[]` with no exception, and the next call restarts the process.
- (d) `af_use_exiftool=False` never spawns it.
- (e) io/exif.py `_from_exiftool` with the same fake: a RAW meta read fills timestamp, serial and orientation.
- Remove "untested" from TASKS T7 / RESUME.

### [x] B2.4 m1: one real off-centre AF sample
Do: from raw.pixls.us (CC0) add one Canon or Sony file whose AF point is clearly off-centre, ideally portrait orientation 6 or 8. Commit only the first 512 KB head, as with the existing heads, and put the licence in a `.LICENSE.txt`. Ground truth without exiftool:
- (i) Hand-decode the maker-note bytes once and record the raw values in the test as constants.
- (ii) Independent check: the parsed point, after orientation, lies on the sharpest tile of the embedded preview (Laplacian variance, tile within the top 10 % of the 8×8 grid).
- If the user has installed exiftool locally, the builder may also cross-check against it, but the test must not depend on it.
Acceptance (`models`-free test): the point is ≥0.15 from centre in at least one axis, the X/Y sign and orientation mapping are confirmed by (ii), and `af_known=True`. If no suitable sample can be found in 30 minutes, record that and close m1 as "accepted, SOFT tier only". Do not fake it.

### [x] B2.5 Real closed-eye video test
Do: download one CC0/public-domain Wikimedia Commons portrait with clearly closed eyes, store it as ≤1024 px JPEG, record source URL, author, licence and sha256. Encode it into a short mp4 with cv2's `VideoWriter` (mp4v): 2 s open-eye frames, 1 s closed, 2 s open. Use the existing open-eye fixture for the open frames, or the same person if one is available.
Acceptance: `@pytest.mark.models` test in test_video.py with the **real** landmarker. The closed span's frames get `eyes_closed`, and the picked frame-grab candidate is not from the closed span. The existing open-eye test stays. If MediaPipe does not flag the real photo as closed on the still image itself, choose another photo: the test is meant to check the video path, not the eye model.

---

### Batch 2 resolution (builder, 2026-11-08)
- **m6** `_wait_if_paused` releases once per pause (`_release_models`: vision + OCR sessions, `gc`, macOS `malloc_zone_pressure_relief`); a test-hook embedder is kept. Test `test_pause_releases_models_once_and_resume_is_bit_identical` (release within 1 s, once not per tick, embeddings byte-identical to an uninterrupted run). New benchmark stage "cull with indexer paused mid-run" runs cull-only and paused as two fresh subprocesses: **old behaviour (no release) +465 MB, new +209 MB** (target +100: NOT met, see deviations). Release itself works (RSS 985 -> 731 MB, sessions gone in ~0.9 s).
- **OCR memory** `scripts/bench_ocr_rss.py` (fresh process, det + rec, 50 frames: 3 CC0 jersey photos + bench frames). Before: **+210 MB** (3 runs 173-210 depending on the frames, the CC0 photos included: 210). `enable_mem_pattern=False` and the CPU arena off were already set in `_session` (no change possible). det long edge 960 -> **608**: **+121 MB** (2 runs, budget 120, i.e. 1 MB over). ORT threads 1/2/4 make no difference. Accuracy table (synthetic labelled set, 100 frames / 3 CC0 photos / small digits 8 cases):

  | det edge | MB added | P | R | CC0 18/47/64 | small digits | test_ocr+test_jersey |
  |---|---|---|---|---|---|---|
  | 960 (before) | 210 | 1.00 | 0.971 | same | 7/8 | pass |
  | 736 | 150-168 | 1.00 | 0.952 | same | 7/8 | **fails** `rendered_digits[0.0-88]` |
  | 704 | 159 | 1.00 | 0.962 | same | 6/8 | pass |
  | **608 (kept)** | **121** | **1.00** | **0.942** | **same** | **5/8** | pass |
  | 576 | 108 | 1.00 | 0.923-0.95 | same | 4/8 | **fails** `single_digits` (>=5) |

  So 960 -> 608 is the cheapest setting that keeps every existing test; the cost is small-digit recall (7/8 -> 5/8 on the hardest synthetic cases, still never a wrong number). Review #9: keep 608 and **amend the budget to +125 MB**, or accept 576 and fix the small-digit test threshold.
- **exiftool** fake `tests/fixtures/bin/exiftool` (Python, `-stay_open` protocol, canned JSON per file, CRASH trigger, spawn log); 7 tests in test_makernotes.py (Canon, Sony, malformed, crash + restart, disabled, `_from_exiftool`). "Untested" removed from TASKS T7 / RESUME.
- **m1** `tests/fixtures/sony_rx100m5a_cc0/` (Sony DSC-RX100M5A, first 512 KB of raw.pixls.us 20181014_0082.ARW, CC0 by the site's upload declaration, sha256 pinned in the test and the LICENSE file). Maker note hand-decoded: FocusLocation `[5472, 3648, 1872, 2424]` -> (0.342, 0.664): 0.158 / 0.164 from centre; on rank 4 of 64 Laplacian tiles of the embedded 160x120 thumbnail (top 10 %); mirrored/swapped variants fall outside. **Deviation: orientation 1 (landscape).** No portrait sample (orientation 6/8) with an off-centre point among ~60 Canon/Sony heads scanned in the time box; the 6/8 mapping is checked by rotating the real thumbnail (`apply_orientation`) and following the point. `af_known` is covered by the same parse path.
- **closed eyes** `tests/fixtures/closed_eyes_cc0/` (Commons "Closed Eyes (Unsplash)", Toa Heftiba, CC0; blink 0.74/0.71 from the real landmarker, still checked first). mp4v clip 2 s open (portrait2) / 1 s closed / 2 s open; frames of the closed span get `eyes_closed` (real YuNet + MediaPipe), the open ones do not, no pick and no `run_cull` result lies in the closed span. **Deviation: `pv.CUT_CHI2` is monkeypatched to 1.01 in this test**: the open frames are a different person (no CC0 same-person pair found), so the cut would put the closed clip in its own scene segment and force a pick there.

**Deviations / for Review #9**: (1) m6 mid-run pause is +209 MB over cull-only, not +100: ORT's freed SigLIP weights stay resident in macOS's "Malloc Large (empty)" cache (273 MB measured with vmmap; `malloc_zone_pressure_relief`, `use_device_allocator_for_initializers`, disable_prepacking all return 0). Only a SigLIP worker subprocess would fix it; ruling needed (amend to +210 MB or a follow-up). (2) OCR +121 MB at 608, see the table. (3) m1 sample is landscape. (4) closed-eye test patches CUT_CHI2. (5) no `make_fixtures.py` download entries: the new fixtures are committed with sha256 in the LICENSE/SOURCE files and pinned in the tests, like the existing heads.

## Review #9 (reviewer, after Batch 2)
Check that every regression test fails on the old code, check the r2 two-process test for timing dependence, and rule on the OCR budget. Expected verdict: SHIP of the maintenance pass, then start Phase 4.

---

## Phase 4: make it usable by a photographer who is not us (small)
Rationale: the culling engine is feature-complete for the target segment (wedding, events, sports) and every budget holds, but today using it means a terminal, `uv`, script names and reading docs. The remaining user value is in **getting it running, getting through the first shoot, and closing the Lightroom loop**, not in more models. Keep it to 4 tasks, no new models, the same budgets.

| # | Task | User value | Acceptance (sketch; the architect writes `10_PHASE4.md`) |
|---|---|---|---|
| T18 | **One-command install + launcher** | No terminal knowledge needed | `./install.sh` from a fresh clone (uv bootstrap into the repo folder, `uv sync`, core models with sha check) followed by `ShotSieve.app` (a plain shell-script bundle built by a repo script, no third-party packer) that runs `shotsieve serve --open`. Everything stays inside the repo folder, and uninstall = delete the folder. Tested on a clean macOS user account: from clone to open UI in ≤10 min on a home connection, no other prompts. Unsigned (signing/notarisation needs an Apple developer account: the user's decision, out of scope). |
| T19 | **First-run onboarding in the UI** | First shoot succeeds without docs | Wizard: pick the shoot folder → optional Lightroom catalog (train personal model, with progress/ETA) → explained consent toggles (people, OCR, search downloads with sizes) → first cull → a "Lightroom step" card (Read Metadata from Files for JPEG/DNG; sidecars for RAW). Models download from the UI with progress (no CLI). Playwright test drives the whole flow on fixtures with no console errors. |
| T20 | **Lightroom Classic round-trip "sync back"** | The personal model improves from real use | `shotsieve sync --lrcat` (and a UI button) reads the ratings/flags/labels the user changed in LR after the cull and stores them as overrides. The next `train` and per-shoot recalibration use them. A re-cull never overwrites a value the user set in LR (user wins, like the crs digest rule). Also a local "this shoot" report: missed-keeper rate and review fraction vs the user's final LR picks (reuses `eval`). Fixture catalog test: cull → modify the catalog the way LR would → sync → overrides match; re-cull leaves them alone. |
| T21 | **Large-shoot performance** | 5–10k-frame weddings feel instant | Synthetic 10k shoot: time to first grid ≤5 s on a warm cache; adding 500 files re-culls only the new ones plus their affected groups; UI scroll stays smooth at 10k (virtualised grid test extended from 5k); peak RSS ≤3 GB; full cold cull ≥10 img/s. |

Explicitly **not** in Phase 4: album sequencing (still no ground truth; revisit after T20 gives real pick data), HSL/tone-curve prediction, a Windows/Linux installer (the CPU fallback stays tested, packaging is macOS only for now), cloud anything, and new models.

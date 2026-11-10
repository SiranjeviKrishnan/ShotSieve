# CLI, API and Review UI

## CLI (`shotsieve` console script)
```
shotsieve models download              # fetch all models into ./models
shotsieve models check                 # verify presence + sha
shotsieve ingest SRC DEST [--backup DIR] [--by-date]
shotsieve cull FOLDER [--write-xmp] [--json out.json] [--config cfg.toml]
shotsieve train --lrcat CAT.lrcat [--xmp-folder DIR ...] [--from-overrides FOLDER ...]
shotsieve eval FOLDER --truth (xmp|lrcat) [--lrcat CAT]   # prints missed-keeper rate etc.
shotsieve serve [FOLDER] [--port 8765]  # opens review UI at http://127.0.0.1:8765
shotsieve match FOLDER --ref IMG [--targets keeps|group|camera:MODEL] [--dry-run]
                [--embed-copies [DIR]] [--catalog CAT.lrcat] [--json]   # T16 reference colour match
shotsieve shotlist FOLDER --list (FILE|wedding|sports|corporate) [--json] [--tag-xmp]   # T17
```
Exit code non-zero on failure; progress bar on stderr; summary table on stdout.

## HTTP API (FastAPI, bind 127.0.0.1 only)
- `POST /api/cull {folder, write_xmp}` → `{job_id}` (runs in background thread; one job at a time)
- `GET /api/jobs/{id}` → `{state, stage, done, total, error}`
- `GET /api/shoot?folder=` → images (path id, decision, reasons, score, group_id, audit, best) + groups + stats
- `GET /api/thumb/{image_id}?size=400` → JPEG (written at cull time from the decoded preview, rendered on demand otherwise; `.shotsieve/thumbs/`; `Cache-Control: max-age=86400, immutable`). Responses are gzipped above 4 KB; `GET /api/shoot` never writes (ids are assigned at cull time)
- `GET /api/preview/{image_id}?w=|bw=&bh=` → the picture for the loupe / compare at the size the screen needs (T23f): `w` = long edge in device px, or `bw`×`bh` = the box it is shown in (device px; the server works out the long edge at Fit from the image's own aspect), snapped up to 1280 / 2048 / 2880 / 3840 (above: the native-size image as `/api/full`; none: 2048). Never upscaled; JPEG q92, 4:4:4, an embedded sRGB profile (wide-gamut sources are converted); `.shotsieve/thumbs/v_{id}_{edge}_{mtime}_v2.jpg`, immutable cache headers. The cull's analysis preview (`load_preview`) is not this image.
- `GET /api/full/{image_id}` → JPEG at native size (T23c, for zoom): RAW = the embedded JPEG when it is >= 90 % of the sensor's long edge, else a full demosaic (camera WB); JPEG/HEIC at native size; EXIF orientation applied, same display frame as the preview (a same-stem JPEG next to a RAW is used for both). Cached in `<shoot>/.shotsieve/full/f_{id}_{mtime_ns}.jpg` (written atomically, LRU-pruned to 2 GB per shoot, a hit refreshes the file's mtime), one decode at a time (lock; buffers freed, the colour swap is in place). Never touches originals. 404 unknown image / missing file, 422 undecodable or a video frame, 400 bad folder / Host. `GET /api/full/{image_id}/size` → `{w, h}` the same image will have (no pixel decode; needed for the "100 %" maths before the full image has loaded).
- `POST /api/override {image_id, decision, rating?}` → stores override (with propensity: 1.0 if image was in review/keep shown, audit probability if audit)
- `POST /api/audit/complete {folder}` → runs calibrate, returns new thresholds and estimated missed-keeper rate
- `POST /api/xmp/write {folder}` → writes sidecars for all images (final decisions = override ?? model)
- `POST /api/lr/sync {folder?, catalog, dry_run}` → `{job_id}` (job `kind: "lr_sync"`, stages `read|match|write|report`; `result` = counts, `changes`, `conflicts`, `report`); the catalog must be an absolute existing `.lrcat` path (400 otherwise); it is only ever read (T20)
- `GET /api/lr/report?folder=` → `{report, last, catalog}` (stored `meta.lr_report` / `lr_sync_last` / `lr_catalog`; read-only)
- `GET /api/lr/catalogs?folder=` → `{catalogs: [...~/Pictures/Lightroom/**/*.lrcat to depth 3 + the remembered one], remembered}` (listing only)
- `GET /api/health` → `{ok, models: {...}, coreml: bool}`
Optional `folder` query/body parameter on every endpoint after `POST /api/cull` (defaults to the last folder used).
`POST /api/override` also accepts `decision: "model"` (clear the override) and a rating-only body. Foreign `Host` headers are refused.
image_id = stable integer id from cull.db (`image_ids` table). Paths in requests must be validated (exist, are dirs); no path traversal on thumbs.

## Review UI (`server/static`, vanilla JS, no build)
Layout: top bar (folder input, progress, stats: keep/review/reject counts), left filter tabs
(All | Keep | Review | Reject | Audit), main grid of thumbnails grouped by duplicate group (group = horizontal strip,
best image badge ★), right panel: loupe of selected image + reasons list in plain language + score.
Keyboard (must all work; show cheat-sheet with `?`):
- `←/→` prev/next image, `↑/↓` prev/next group
- `P` keep, `X` reject, `U` back to model decision, `1–5` rating
- `Space` toggle loupe/fullscreen, `C` compare best vs current side by side, `G` collapse/expand groups
- `Tab` cycle filter tabs, `A` jump to next audit item, `W` write XMP, `?` help
Zoom (T23c, `static/zoom.js`, shared by the full-screen loupe (`Space`) and compare, so both behave the same):
- State per view = zoom relative to Fit + the image point at the view centre (image-relative, so it survives a frame change). 100 % = one image pixel per device pixel (`devicePixelRatio`); max 16x Fit or 800 %.
- Toolbar control in the loupe bar and in the compare header: `−`, a badge (`Fit` / `NN %`, spinner while the full-res image loads) with a menu (Fit, 50, 100, 200, 400 %), `+`.
- Input: pinch (WebKit `gesturestart/change/end`, and ctrl+wheel for Chromium) zooms around the cursor; Cmd/Alt+wheel too; two-finger scroll and click-drag pan while zoomed (a plain scroll at Fit does nothing and never changes frame); double-click or `Z` toggles Fit and 100 % at the cursor; `+ = - 0` and `Cmd/Ctrl + = - 0` (0 = Fit). A dragged click never closes the loupe; clicking the picture no longer closes it (Esc, Space, the Close button or the backdrop do).
- Corner navigator thumbnail (shown when zoomed) with the visible region; click/drag moves it.
- Progressive: the 1600 px preview is shown at once; when zoom > Fit (and after 180 ms of calm) `/api/full/{id}` is loaded in the background and swapped in (`data-full="1"` on the view). Videos have no full view (the frame-grab preview is magnified).
- Zoom is remembered per photo (T23g; the "Keep zoom" checkbox and `shotsieve.keepZoom` are gone): each view (loupe, compare left, compare right) keeps an in-memory LRU (300) image_id -> {zoom, position}; moving to a photo shows its own state (Fit if never zoomed). Compare with Link on: the right side takes the left side's state.
- Compare: zoom/pan synced by default, `L` / the Link button unlinks (each side on its own; relinking snaps to the side last used); the toolbar buttons and keys drive both sides.
- Crop mode: zoom is **disabled** (the crop overlay is not scaled); entering crop resets to Fit, zoom keys are ignored there.
- DOM hooks for tests: `#lz-view`, `.cmp-view` carry `data-z`, `data-pct`, `data-full`, `data-loading`.

Thumbnails lazy-load (IntersectionObserver). Grid virtualized enough to handle 5,000 images without lag
(render only visible rows). Colors: green keep, yellow review, red reject outline.

- `POST /api/audit/complete` returns 409 while audited rejects are unjudged; `POST /api/xmp/write`
  takes `force_ratings`, returns `kept_user_rating/label` and `warnings` (Lightroom JPEG/DNG note).
- `GET /api/shoot` never changes the server's current folder.

## Phase 3 additions (T16, T17)
- `GET /api/edit/reference?id=&folder=&catalog=` -> `{ok, source: sidecar|embedded|catalog, n_settings, exposure, temperature}`
  or `{ok: false, message: "the reference has no develop settings: edit the reference in Lightroom first"}`.
- `POST /api/edit/match {folder, ref: image id, targets: keeps|group|camera:MODEL, ids?, dry_run, embed_copies, catalog}` -> `{job_id}`
  (job kind `edit_match`; `result` = counts + a per-image diff: `values`, `delta_ev`, `cos`, `wb`, `notes`; 400 when the reference has no
  develop settings). Same writing policy as `/api/edit/apply`.
- `GET /api/shotlist` -> the stored list with the last statuses `{items: [{item_id, text, must, min_count, person, status, detail,
  n_hits, top_score, top_z, best_ids}], counts, must_missing, templates, has_list}` (no model work).
- `POST /api/shotlist {folder, list | text, tag_xmp}` loads a list (template name or the text: `[!]text [person:"Name"] [xN]` per
  line), scores it and stores it in `cull.db` (`shotlist` table); `GET /api/shotlist/coverage` re-scores the stored list against the
  current index (also `?tag_xmp=true`); `DELETE /api/shotlist` clears it. 503 without the search models. `status` is `covered | weak |
  missing | pending`; `pending` carries `detail: "pending (indexed X %)"` and is used whenever the index is incomplete.
- UI: loupe panel "Colour match" (Set as reference / Match selected: this image, its group or all keeps; dry run, then Write);
  header button "Shot list" (templates, paste a list, per-item ✓ ~ ✗ …, thumbnails, click an item to filter the grid to its frames) and a
  red banner listing the missing must-have shots after a cull or an index run finishes.

## Phase 5 additions (T22 subject-aware crop; spec docs/11_CROP.md)
All under `/api/crop/` (`server/crop.py`), `folder` as everywhere; display-normalised rects `[x0,y0,x1,y1]` (after EXIF orientation). GETs never
write user-visible state (the `subjects` cache fill is derived data). Errors: unknown id 404, video item 422, bad ratio / rect outside [0,1] / rect
not matching the ratio (> 1 px) / unknown subject 422.
- `GET /api/crop/{id}` -> `{current, history[], full:{w,h}, orientation, xmp:{has_crop, ours, source}, ratios[], kind}`; `current.source` is `lightroom`
  (read-only) when the sidecar holds the photographer's own crop.
- `GET /api/crop/{id}/subjects` -> `{subjects[{sid, kind, box, score, label, landmarks, linked}], saliency_ready, detector: "yolox_s"|null, full, orientation, peak, horizon}`.
- `POST /api/crop/{id}/point {x, y}` -> `{subject}` (kind `point`); `POST /api/crop/{id}/suggest {subjects[sid], points[{x,y}], select_modes{sid: person|face},
  ratio, flip, n}` -> `{candidates[{rect, rect_px, score, terms, warnings}], ratio, ratio_px, infeasible, auto, selected, horizon_tilt}` (no selection = auto;
  a portrait preset on a landscape image is taken in landscape unless `flip`; `infeasible` = `subjects_do_not_fit`, never a silent cut).
- `POST /api/crop/{id}/accept {rect, ratio, source: ai|ai_adjusted|manual|burst, subjects, alt_rank, write: auto|none|embed, embed_copies, catalog}` ->
  `{version, rect, write: {mode, status, reason, path}}` (the stored rect is snapped to the exact ratio in full-resolution pixels; `status` written |
  replaced_own | skipped_user_crop | skipped_catalog | db_only (JPEG/DNG: Lightroom ignores sidecars, use `write: embed`) | none);
  `POST .../reject {alt_rank}`; `POST .../reset {write}` (a new version with no crop; the sidecar goes back to Lightroom's defaults);
  `POST .../restore {version}` (a new version copying the old one). Versions are append-only.
- `POST /api/crop/burst {id, ids?, rect, ratio, subjects, points, select_modes}` -> `{job_id}` (kind `crop_burst`, one-job lock; `result.results[{id, rect,
  rect_px, status: suggested|needs_selection}]`: proposals only); `POST /api/crop/accept_many {items[{id, rect, ratio}]}`.
- `/api/shoot` images carry `crop: [x0,y0,x1,y1] | null`. UI: R opens crop mode in the loupe (subject boxes, click / Shift-click / Option-click, point prompt,
  ratio chips + custom W:H + Flip (X), 3 alternatives (`[` `]`), Enter accepts, Backspace rejects, drag handles ratio-locked, Reset, History + Restore,
  "show crops" toggle, Apply to burst).

- T23b: `suggest` also takes `boxes[{x0,y0,x1,y1}]` (user-drawn must-keep subjects, display frame, 0..1, x0<x1, y0<y1, max 6, else 422); `GET|POST /api/crop/{id}/boxes`
  (`{boxes:[{x0..y1}]}` -> `{boxes:[[x0,y0,x1,y1]]}`, append-only versions in `crop_user_boxes`); `GET /api/crop/{id}` carries `user_boxes`. `/api/shoot` images carry
  `duplicate_of: {image_id, name} | null` and `metrics {sharp, eyes_open, face, n_faces}` (compare view). UI: drag draws a box in crop mode; the crop moves by its grip.

## T23d additions (faces, People view, Open in Lightroom; details in TASKS T23d)
- `GET /api/image/{id}/faces` -> `{faces:[{idx, box:[x0,y0,x1,y1] (display frame, 0..1), important, main, person_id, name, face_idx, det_score, sharp, eyes_open, smile, drove, drove_reason}], people_consent}`; `GET /api/image/{id}/face/{k}?size=` and `GET /api/people/{id}/face?size=` (cached face JPEG).
- `GET /api/people` -> `{people:[{id, name, n, photos, keeps, cover_id, cover_idx}], unsorted, ignored}`; `GET /api/people/{id}/images` -> `{ids}`; `GET /api/people/names`; `GET /api/people/chips` -> `{chips:{id:[names]}}`; `POST /api/people/{id} {name?, merge_into?, ignore?}` (404 for an unknown merge target); `POST /api/people/restore_ignored`; `POST /api/image/{id}/faces/{k}/detach` ("Not this person", persists across re-clustering); `POST /api/image/{id}/faces/{k}/assign {person_id? | name?}`. All people endpoints need the people consent (403).
- `GET /api/lightroom/apps?folder=` -> `{apps:[{id: cloud|classic, name, bundle_id, path}], preferred, xmp_choice, classic_sync}`; `POST /api/lightroom/open {folder?, ids (1..500), app?, xmp?: write|skip, remember?}` -> `{ok, app, name, bundle_id, n, missing, xmp, xmp_choice}` or `{ok:false, needs_consent:true, app, n}` (nothing opened, nothing written); runs `open -b com.adobe.lightroomCC <RAW masters>` (Classic: `com.adobe.LightroomClassicCC7`) with an argument list; 422 empty / > 500 / bad app, 404 unknown or outside-folder id, 409 nothing installed, 502 `open` failed. Same Origin / Host rules as every POST.
- UI: toolbar "People" (Shift+P), View menu (Faces in the loupe `F`, Name chips), More menu (Sync with Lightroom Classic catalog..., Jersey numbers...), "Open in Lightroom" (Cmd+L; loupe bar, detail panel, grid right-click). The old toolbar "Lightroom" button and the jersey bar are gone. zoom.js exposes `view.frame()` (image rect in view px) and `view.onRender(fn)` for overlays.

## T23e additions (filter bar, colour labels, Key faces; details in TASKS T23e)
- `POST /api/label {folder?, image_ids (1..5000), label: red|yellow|green|blue|purple|none}` -> `{ok, n, label}` (422 bad label / count, 404 unknown id). Stored per image in `labels` (newest wins, "" = none). Written to `xmp:Label` (Red/Yellow/Green/Blue/Purple, "" for none) by `POST /api/xmp/write` and Open in Lightroom with the sidecar rules unchanged (user label wins, not recorded as `shotsieve:WrittenLabel`; images without a user label behave as before).
- `GET /api/shoot/facets?folder=` -> `{n, pending, scanning, vals: {make, model, lens, lens_serial: [{label, raw: [spellings], n}]}, rows: {id: [make_i, model_i, lens_i, serial_i, orient]} (index -1 = missing; orient 0 landscape / 1 portrait / 2 square / -1), labels: {id: key}}`. `POST /api/shoot/facets/scan {folder?}` starts the background back-fill of lens fields and sidecar labels (`exif_extra`; only reads EXIF and sidecars, never touches decisions or features). `GET /api/people/map` -> `{people: {person id: [image ids]}}` (403 without the people consent).
- UI: filter bar above the grid (People, Picked / Unflagged / Rejected, colour dots, stars 0-5, More, count, Clear all filters, chips), Key faces in the side panel, rating pill + label dot + pick mark on grid cells, label picker in the side panel and the grid context menu. Keys: `0` clears stars, `6` `7` `8` `9` red / yellow / green / blue, Cmd/Ctrl+F search, Shift+F show / hide the bar. Filters are client-side and AND across groups, OR inside one (people any / all); remembered per shoot (`localStorage shotsieve.filters:<folder>`, `shotsieve.filterbar`, `shotsieve.kf.*`).

### CLI `shotsieve crop`
`crop detect FOLDER [--ids N,N|--keeps]` (fill the subject cache) - `crop suggest FOLDER --ratio W:H [--flip] [--n 3] [--box x0,y0,x1,y1 ...] [--json]` (writes nothing) -
`crop apply FOLDER --ratio W:H [--flip] [--rect x0,y0,x1,y1 --id N] [--write auto|none|embed] [--embed-copies [DIR]] [--catalog CAT] [--dry-run] [--json]`
(an explicit apply is the acceptance: top-1 or `--rect` stored as a version, XMP written) - `crop reset FOLDER [--ids|--all]` - `crop status FOLDER` -
`crop history FOLDER --id N`. Exit codes: 0 ok, 2 bad arguments / ratio, 3 nothing to do.

## T23f additions (viewer quality + denoise; details in TASKS T23f)
- `GET /api/thumb/{id}?size=` now renders through the same colour-managed pipeline (q85, sRGB) unless the cull already wrote the 400 px thumb and the source has no wide-gamut profile; the UI asks for x2 sizes on a Retina display (`devicePixelRatio >= 1.5`).
- `GET /api/denoise/status` -> `{model: {name, installed, bytes, license, group}, backend: ai|basic, loaded, levels}`. `SHOTSIEVE_DENOISE_BACKEND=basic` forces the OpenCV fallback.
- `GET /api/denoise/{id}?level=low|medium|high&w=|bw=&bh=` -> the Fit picture (the viewer image of that bucket) denoised; header `X-Denoise-Backend: ai|basic`. Above the top bucket = `/full`.
- `GET /api/denoise/{id}/region?level&x0&y0&x1&y1` (fractions 0..1 of the full-resolution frame) -> JPEG of that box (snapped outward to a 64 px grid, at most 6 MP, 32 px of context discarded) denoised on the NATIVE pixels with the noise level of the whole image; headers `X-Region: px0,py0,px1,py1`, `X-Image-Size: w,h`. A waiting region request pauses a running whole-image job between tiles.
- `GET /api/denoise/{id}/full?level` -> the whole full-resolution image denoised (one at a time; a superseded job answers 409). 404 unknown id, 422 bad level / video / undecodable.
- `POST /api/denoise/warm` (202) -> `{warming}`: loads the model and runs one tile in the background (the UI calls it when the loupe / compare opens); `warming:false` with a `reason` when there is no model, it is loaded already or a cull runs.
- The noise level is one value per photo (full-resolution pixels + EXIF ISO, cached) and is used by Fit, region and full alike, so the strength looks the same at every zoom. The URLs carry `dv=2` (engine version) so immutable browser caches never serve an older look.
- Cache `<shoot>/.shotsieve/denoise/{v,r,f}_{id}_{mtime}_v{VERSION}_{level}_{backend}...jpg` (1 GB LRU, hits touch mtime); a changed original, strength or backend is a new key. Originals are never written.
- UI (`static/denoise.js`, `zoom.js` view state `v.dn`): toolbar control in the loupe and compare bars: **Denoise** button + state badge (`Off`, a spinner while working, `AI`, `Basic`; `#lz-denoise` / `#cmp-denoise` carry `data-state`, `data-on`, `data-sides`, `data-level`) + strength menu (Low / Medium / High, remembered in `localStorage shotsieve.denoiseLevel`; "Get the AI denoiser (124 MB)" through `/api/models/download {groups:["denoise"]}`). **D** toggles (compare: both sides, **Shift+D** only the side under the pointer, same for `L`-unlinked views); the state follows frames. Zoomed in: viewport region first (a 1.5 MP centre window, then the whole viewport) as an overlay that fades in (150 ms) once loaded, the denoised full image replaces it. Changing the strength keeps the previous denoised picture / overlay on screen until the new one has loaded (never a flash of the noisy original). Crop mode and the faces overlay are unaffected (D is ignored in crop mode).


## T23g additions (menu bar, automatic XMP; details in TASKS T23g)

- Top-right of the window: only the `?` button (plus a `☰` menu button in the plain browser, `html:not(.native)` only). The old Keep/Review/Reject totals are not repeated: the tabs carry the counts. The toolbar buttons live on, hidden, in `#menu-actions` so every handler is the existing one.
- `static/menu.js`: `window.shotsieveMenu(id)` runs an id of a fixed allow-list (`choose autoxmp crops video lropen faces chips filterbar denoise compare cameras people shotlist edit jersey lrsync auditdone help setup lrcard`, plus `recent:0-9`; anything else returns false). It also reports the page state (toggles, optional items, recent folders) to `window.pywebview.api.menu_state` every 2 s and after each menu action.
- Native menu (`src/shotsieve/desktop_menu.py`, pywebview `Menu`): File (Choose Folder, Open Recent, Open in Lightroom, Close Window Cmd+W), Edit (Undo/Redo/Cut/Copy/Paste/Select All, native), View (Faces, Show crop suggestions, Filter bar, Denoise, Compare, Enter Full Screen), Tools (Include videos, Write XMP sidecars automatically, Cameras, People, Shot list, Edit profile, Jersey numbers / Sync with Lightroom Classic catalog / Finish audit when they apply), Help (Keyboard Shortcuts, Setup Assistant, What to do in Lightroom). Menu actions call `window.evaluate_js("window.shotsieveMenu(<json id>)")`. Check marks are a "✓ " title prefix; the bar is rebuilt on the main thread when the page reports a change. Shortcut ownership: the page owns every key (P X F Shift+P Shift+F Cmd+O Cmd+L ? ...); menu items have no key equivalents except Cmd+W on Close Window, which the page never handled (no double firing).
- Detail panel: Keep, Reject, Crop, Open in Lightroom only. P on a photo you already kept (X on a rejected one) clears your override (`decision: "model"`); the button shows a pressed state. `U` and the Model/Compare buttons are gone; the grid context menu has "Use AI decision" for an overridden photo; Compare is in the View menu and on `C`.
- Main-view denoise: `D` / the "Denoise" button under the preview swaps the preview to `/api/denoise/{id}` (same strength as the loupe, badge Off / ... / AI / Basic); disabled in crop mode.
- Automatic XMP (`server/autoxmp.py`): ON by default for every shoot (standing consent; meta `auto_xmp` = "0" switches it off, menu "Write XMP sidecars automatically"). After a cull job (after the decisions are committed) and after every override, star, colour label and crop accept/reset/restore the sidecars are written by a background thread, debounced 0.8 s and batched, through `prepare_xmp` (same writers, same safety rules). `GET /api/xmp/status` -> `{auto, pending, written, skipped, failed, warnings, at}`; `POST /api/xmp/auto {folder, on}`. The status bar shows "XMP: N skipped (your Lightroom edits kept)" (click for details). `POST /api/lightroom/open` has no consent step any more (`needs_consent` never returned): it flushes pending writes, then opens. `/api/xmp/write` stays for scripts. The Write XMP button, the "write XMP" checkbox and the Lightroom consent sheet are removed.
- Filter bar / people strip (T23g follow-up): the filter bar no longer has a people button or popover; people are filtered from the people strip (click a face: toggles the one People filter, the face gets a ring, a removable chip appears; Key faces and the People view use the same state). The Any of them / All of them switch is in the strip (shown once 2+ faces are selected). The strip is one row that scrolls sideways (wheel / trackpad / arrow buttons at the edges), lists every person with lazy thumbnails, and keeps "All people (N)…" and "Delete people data" pinned at the right.
- Colour match (T16): the detail-panel section is gone; Tools menu -> "Colour match: set this photo as reference" (`shotsieveMenu('matchref')`). The API (`/api/edit/reference`, `/api/edit/match`) and `shotsieve match` are unchanged.
- People re-cluster on open: `people.PEOPLE_CLUSTER_VERSION` (cull.db meta `people_cluster_version`). `/api/shoot` and `/api/people` start a background re-cluster once for a shoot with an older version (or none) and stored face vectors: no detection, no cull re-run; names, "Not this person" / "Not a person" exclusions and merged names carry over by face overlap, `unsorted` becomes 0.
- Test isolation: `tests/conftest.py` points `SHOTSIEVE_DATA_DIR` at a temp dir for the whole run (server subprocesses inherit it), so tests never touch the real `data/` (recent.json etc.).
- Checkboxes removed: "show crops" -> View "Show crop suggestions" (`localStorage shotsieve.showCrops`), "videos" -> Tools "Include videos" (default on, `shotsieve.includeVideo`; the UI sends `video: true`; a photo-only folder is unaffected: no videos are found).
- People: every face is a person (`people.MIN_FACES = 1`): single faces are "Unknown" entries; naming a person with a name another person has merges the two (`people.rename` returns the surviving id). The "N unsorted" texts are gone. Key faces panel: one fixed face size (64 px), the slider and `shotsieve.kf.size` are gone.

## T23h additions (all-faces recognition + subject mode; details in TASKS T23h)
- `GET /api/people/scan?folder=` -> `{running, done, total, faces, error, peak_delta_mb, pending, version}`: the background all-faces recognition scan. It starts by itself when a shoot is opened (`GET /api/shoot` / `GET /api/people`, consent on) if some image has no current scan (`FACE_SCAN_VERSION` bump, new file, changed face count); it re-clusters when done (names and exclusions survive as in every re-cluster). The cull and its decisions never wait for it and it pauses while a cull runs.
- Recognition faces are numbered after the decision faces in `person_faces.face_idx` (0..n-1 = the cull's important faces, n.. = scan faces); `/api/people/{id}/face` serves their thumbnails. `/api/image/{id}/faces` (decision faces) is unchanged.
- `POST /api/subject/{id}/score` `{folder, boxes:[{x0,y0,x1,y1}]}` (display-normalised, the same boxes as `/api/crop/{id}/boxes`) -> `{scores:[{box, sharp (0-1), raw, in_focus, label}], ref, in_focus_at}`. Measure: the cull's face sharpness (Laplacian variance at 96 px) on the FULL-resolution region, normalised by the shoot's p90 face reference (floor 30); `in_focus` = sharp >= 0.25 (the Faces row's "soft" cut). Cached by image + box.
- UI: **S** (button "Subject (S)", View menu "Subject mode", `shotsieveMenu('subject')`) in the loupe: crosshair, drag draws a box, corners resize, drag inside moves, X / Delete / the x button removes the selected one, at most 6, Esc (preventDefault) or S leaves the mode. The boxes are crop mode's user boxes (no second storage), positioned by `loupeView.frame()` so they follow zoom and pan; each shows "Subject n: sharp 0.82 ✓ in focus" / "soft 0.12 ✗ soft". **Removed:** the loupe face overlay, the F key, the "Faces" toolbar button and the "Faces in the loupe" menu item (faces stay in the people strip, the Faces row, People and Key faces). `static/subject.js`.

## T23i additions (multi-select + Move to Trash; details in TASKS T23i)
- `POST /api/override {ids: [..]|image_id, decision?, rating?}`: `ids` 1-5000 (unknown id 404, nothing changed); batch response `{ok, n, results[]}`; one batched auto-XMP request. `POST /api/label {image_ids}` (1-5000) as before.
- `POST /api/trash/preview {ids, include_pairs?, include_videos?}` -> `{n, names[6], n_files, n_pairs, n_video_frames, n_unsafe}`.
- `POST /api/trash {ids, include_pairs=true, include_videos=false}` -> `{token, job_id, trashed, failed, skipped, files, n_files}`. Moves to the macOS Trash (NSFileManager; recoverable), never deletes; strictly inside the shoot folder, no symlinks, never `.shotsieve/`; 409 while a cull runs; the files leave the shoot through the incremental cull (`job_id`).
- `POST /api/trash/undo {token}` -> `{restored, failed, ids, job_id}`: moves the files back (never over an existing file), same incremental cull.
- UI: `static/select.js` (+ `select.css`): Cmd-click toggle, Shift-click range, Cmd+A, Esc, Shift+arrows; Ctrl-click = right-click menu; "N photos selected" panel; Move to Trash sheet (Cmd+Delete, File menu, right-click), Undo toast / File > Undo Move to Trash / Cmd+Z. Menu ids `trash`, `undotrash`.

## Release app (T24)
`GET /api/onboarding` also returns `bundled: true` when the models are the app's read-only bundle: the wizard then
skips the download step. `POST /api/models/download` is a no-op that finishes at once in a bundle (nothing is
fetched). Data and logs live in `~/Library/Application Support/ShotSieve` and `~/Library/Logs/ShotSieve`
(`SHOTSIEVE_DATA_DIR` / `SHOTSIEVE_LOG_DIR` override).

- T25a: the toolbar Cull button is gone. Choosing a folder (Choose Folder, Open Recent, drop, Enter in the folder box) starts the cull by itself, and the server re-decides only new/changed frames. A manual re-run is File > "Re-cull this folder" (menu id `recull`; Cmd+Shift+R in the app, mirrored in the browser hamburger menu). "A cull is running" is `S.culling` in app.js; a second start while one runs is refused with a toast.

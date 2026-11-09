# Testing & Acceptance

Run: `uv run pytest -q` (all) ; `uv run pytest -q -m "not models"` (fast, no model files) ; `uv run ruff check src tests`.

## Fixtures (`scripts/make_fixtures.py` → `tests/fixtures/`, idempotent, cached)
- Download a few **CC0 / public-domain** face photos (e.g. from opencv samples: `https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg`,
  `messi5.jpg`; opencv_zoo YuNet/SFace example images; any other freely licensed portraits incl. at least one group photo).
  Verify the download; if a URL fails, try alternatives; don't silently skip.
- Synthesize a **shoot folder** with EXIF (via piexif): 3 bursts × 5 frames of the same face image with small shifts/crops,
  timestamps 0.3 s apart inside a burst and 60 s between bursts, `Flash` tags; variants: gaussian-blurred (σ=6), heavily
  under-exposed (−3 EV), one flash misfire (fired flag but −2 EV darker), one landscape/no-face image, one black accidental frame.
- One small real RAW with embedded preview (CC0 from raw.pixls.us, e.g. a DNG) for `preview.py`; mark test `@pytest.mark.network` if fetched lazily.
- A synthetic `.lrcat` SQLite with the minimal Lightroom schema (`AgLibraryRootFolder`, `AgLibraryFolder`, `AgLibraryFile`, `Adobe_images` with `pick`, `rating`, `colorLabels`, `rootFile`).

## Required tests
| Module | Test |
|---|---|
| exif | reads timestamp+subsec, serial, flash; garbage file → defaults, no exception |
| preview | JPEG downscale + orientation; RAW embedded thumb path; size ≤ max_edge |
| xmp | write → read roundtrip; merge preserves foreign fields; valid XML |
| lrcat | parses synthetic catalog → correct labels and absolute paths |
| copier | copy+backup+verify; re-run skips; corrupted dest detected |
| face_detect (models) | finds ≥1 face on lena/messi; group photo ≥ 3 faces; none on landscape |
| face_landmarks (models) | eyes-open face → eyes_closed < 0.5 |
| face_embed (models) | same face shifted → cos ≥ 0.6; different people → lower |
| image_embed (models) | burst frames cos-dist < 0.12; landscape vs face > 0.3; batch == single result |
| sharpness / exposure | blurred < sharp; dark image ev < −2 |
| flash | misfire frame flagged, normal flash frames not flagged, < 2 neighbours → no decision |
| bursts / duplicates | synthetic shoot → 3 bursts; groups match; O(n·w) (no full n² matrix) |
| quality / selection / reasons | blurred & dark frames rejected with right reason codes; exactly one best per group; every reject has reason text |
| personal | fit on synthetic labels → AUC > 0.8; IPW weights applied; schema mismatch refuses load; < 40 rows → prior |
| audit / calibrate | deterministic sample; IPW estimate math; threshold lowered when estimate > 1% |
| runner (integration, models) | end-to-end on fixture shoot: every image decided, cache re-run ≥ 10× faster, XMP written |
| evaluate | missed-keeper rate computed correctly on hand-made truth |
| api | health, cull job lifecycle, shoot listing, thumb, override, xmp write (TestClient) |
| cli | `--help` for all subcommands; `cull --json` on fixtures |

## Acceptance (all must hold; verified by the reviewer)
1. `uv run pytest -q` → 0 failures, 0 errors (model tests included, models downloaded).
2. `ruff check` clean.
3. `uv run python scripts/benchmark.py --n 300` on synthetic images: ≥ 10 img/s full pipeline, peak RSS ≤ 3 GB; report written to `data/benchmark.md`.
4. `shotsieve serve` starts, `/api/health` ok, UI loads and keyboard handlers exist for every key in 04_API_UI.md.
5. Every MVP feature in 00_OVERVIEW.md maps to code + test (checked in `docs/TASKS.md`).

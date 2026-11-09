# Architecture

## Directory layout (authoritative — do not invent other top-level dirs)
```
shotsieve/
  README.md                 quickstart
  pyproject.toml            uv project, deps, pytest config, console script `shotsieve`
  docs/                     these specs (00..06, TASKS.md)
  models/                   downloaded weights (gitignored), manifest in src/shotsieve/models/registry.py
  data/                     runtime data: cache DBs, logs (gitignored)
  scripts/
    download_models.py      fetch + sha256-verify all models
    make_fixtures.py        build test fixtures (downloads CC0 images, synthesizes bursts)
    benchmark.py            throughput + peak RSS report
  src/shotsieve/
    __init__.py
    config.py               dataclass Settings (thresholds, paths, batch sizes); load/save TOML
    cli.py                  argparse entrypoint (see 04_API_UI.md)
    log.py                  logging setup
    io/
      preview.py            load_preview(path) -> RGB uint8 ndarray (embedded JPEG for RAW, downscaled)
      exif.py               read_meta(path) -> ImageMeta (timestamp, subsec, serial, model, flash_fired, iso, shutter, aperture, orientation, burst seq)
      xmp.py                read_sidecar / write_sidecar (rating, label, shotsieve:decision/reasons)
      lrcat.py              read Lightroom catalog (strictly read-only: `mode=ro` URI, else a checked private copy) -> [CatalogEntry(path, rating, pick, label, touch_ts, capture, id_local, ...)]
      formats.py            supported extensions (RAW + JPEG/HEIC/PNG/TIFF)
    ingest/
      copier.py             copy card -> primary + backup, blake2b verify, skip existing, report
    models/
      registry.py           MODEL_SPECS (name, url, sha256, size, license), ensure_models(), model_path()
      runtime.py            ORT session factory: CoreML EP -> CPU fallback; thread settings
      face_detect.py        YuNet via cv2.FaceDetectorYN -> [Face(bbox, landmarks5, score)]
      face_embed.py         SFace via cv2.FaceRecognizerSF -> 128-d L2-normed vectors
      face_landmarks.py     MediaPipe FaceLandmarker (blendshapes) on face crops -> eye_open L/R, smile, etc.
      image_embed.py        DINOv2-small fp32 ONNX (CPU EP; see 02_MODELS.md) -> 384-d L2-normed vector, batched
    features/
      sharpness.py          global + face-ROI sharpness (variance of Laplacian on normalized luma, + tenengrad)
      exposure.py           mean luma, clipped hi/lo fractions, EV estimate
      eyes.py               eyes-closed score per face from blendshapes
      expression.py         smile / neutral score
      flash.py              flash-misfire detector (EXIF flash + neighbor luminance)
      subject.py            subject presence & importance (face size, centrality)
      vector.py             FeatureVector dataclass + to_array() fixed order (used by personal model)
    grouping/
      bursts.py             split by time gap (per camera serial)
      duplicates.py         within/near bursts: agglomerative clustering on image embeddings (cosine)
    scoring/
      quality.py            rule-based quality score in [0,1] + per-criterion sub-scores
      selection.py          best-pick per group, decide keep/reject/review
      reasons.py            reason codes -> plain-language strings
    personal/
      dataset.py            build (features, label, weight) from catalog/XMP history + UI overrides
      model.py              PersonalModel: HistGradientBoosting + logistic calibration, shrink to prior
      audit.py              propensity sampling of reject bucket into review
      calibrate.py          per-shoot threshold recalibration from audit results
    pipeline/
      runner.py             run_cull(folder, settings) -> CullResult ; staged, batched, cached, progress callback
      cache.py              SQLite feature cache keyed by (path, size, mtime)
      evaluate.py           missed-keeper rate & other metrics vs ground truth
    server/
      app.py                FastAPI app (see 04_API_UI.md); background job thread
      static/index.html, app.js, style.css   keyboard-first review UI (vanilla JS, no build step)
  tests/
    conftest.py, fixtures/ (generated), unit tests per module, test_integration.py, test_api.py
```

## Data flow
```
folder ─► scan (formats.py) ─► exif.read_meta ─► preview.load_preview (≤1280px long edge)
   ─► Stage A (cheap, all images): exposure, global sharpness, flash candidates
   ─► Stage B (faces): YuNet detect → for top-K faces (K=6 by size): SFace embed, MediaPipe blendshapes, face sharpness
   ─► Stage C (embeddings): DINOv2 224px batched (batch=16)
   ─► grouping: bursts (time) → duplicates (embedding cosine)
   ─► scoring: rule quality → personal model (if trained) blended with prior → group best-pick
   ─► decisions + reasons ─► audit sampling ─► persist (cache DB) ─► optional XMP write
```

## Performance rules
- Decode once per image; pass downscaled arrays between stages. Never hold full-res RAW decodes.
- Stream images through a bounded queue (size ≤ 64); preview decode in a ThreadPool (4 workers).
- Models are lazy-loaded singletons; loaded once per process.
- Feature cache: re-running on the same folder must skip all model work (cache hit) and finish < 2s / 1000 imgs.
- ORT: `intra_op_num_threads=6`, CPU provider for DINOv2 (CoreML measured 2-3x slower; opt-in via `Settings.use_coreml`).

## Code rules
- Type hints everywhere, dataclasses for records, no global mutable state except model singletons.
- Every module has unit tests. Network-free tests except those marked `@pytest.mark.models` (require downloaded models; downloaded once by `scripts/download_models.py`).
- Ruff-clean (`ruff check src tests`).

## Lightroom catalog schema assumptions (T20, `io/lrcat.py`)
Tables `Adobe_images` (`id_local, rootFile, pick, rating, colorLabels`, optional `masterImage`, `touchTime`, `captureTime`),
`AgLibraryFile` (`id_local, folder, baseName, extension`, optional `originalFilename`, `sidecarExtensions`), `AgLibraryFolder`
(`id_local, rootFolder, pathFromRoot`), `AgLibraryRootFolder` (`id_local, absolutePath`). Optional columns are found with
`PRAGMA table_info`; `pick`/`rating` may be REAL (cast in Python); `touchTime` is seconds since 2001-01-01 UTC; `colorLabels` holds the
label-set *name* (mapped by `Settings.lr_label_map`). A catalog with a `-wal`, `-shm` or `-journal` next to it (Lightroom running, or not
cleanly closed) is never opened in place: it is copied to `data/tmp/lrcat-*` (APFS clone), the copy is opened normally and checked with
`PRAGMA quick_check` (3 tries, then `CatalogBusyError`), and the temp dir is deleted.

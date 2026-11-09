# Roadmap

**MVP** (this build): personal culling from catalog history + audit sampling + per-shoot recalibration · plain-language reject
reasons · burst/duplicate grouping with best-pick · fast ingest, embedded-preview culling, keyboard UI · XMP round-trip,
cross-platform · flash misfire detection.

**Phase 2** (build order, spec in 07_PHASE2.md): N5 fix · AF-point focus check (Canon/Sony) · e-shutter banding detection ·
multi-camera clock offset with auto-suggested match pair · natural-language + people search (SigLIP2 + sqlite-vec, background
indexer) · adult-sports jersey OCR + roster · video frame-grab culling.

**Phase 3** (build order, spec in 08_PHASE3.md): benchmark HEVC budget 350 MB · local personal editing profile (crs: slider regression, RAW sidecars + JPEG/DNG embedded copies) · reference-frame colour match · shot-list coverage alerts. Deferred: album sequencing, HSL/tone-curve prediction.

**Cut**: school roster workflow + compliance · approximate LR render preview · skin-chroma anchoring · head swap · LLM captions · conformal "guarantees" · SHAP in UI.

Legal notes: face embeddings are biometric data (BIPA/GDPR) — stored only locally in the shoot's `.shotsieve/` folder;
add explicit consent toggle before Phase 2 people-search ships.

# ShotSieve — Overview

Local-first AI photo culling for photographers (wedding, events, portraits, sports).
Everything runs on the user's machine. Originals are never modified.

## Hard constraints
- Target machine: Apple Silicon Mac with **16 GB RAM**. Peak resident memory of the pipeline **≤ 3 GB**.
- The shipped app targets macOS on Apple Silicon. The pipeline itself has CPU fallbacks and no macOS-only code paths (the native window, Trash, Lightroom launcher and installer are macOS only).
- Throughput target: **≥ 10 images/s** on M1-class for the full cull (JPEG previews), measured by `scripts/benchmark.py`.
- UI must stay responsive: inference runs in a background worker; the API never blocks on model work.
- Python 3.11, managed with `uv`. All code, venv, models, caches live inside the repository folder.
- Only permissively licensed models (MIT / Apache-2.0). See `02_MODELS.md`.

## MVP scope (all required)
1. **Personal culling** bootstrapped from catalog history (Lightroom `.lrcat` + XMP sidecars), with
   audit-sampled reject buckets (inverse-propensity weights) and per-shoot recalibration.
2. **Plain-language reason on every reject** (blur, missed focus on face, eyes closed, poor exposure,
   flash misfire, duplicate-not-best, no subject / accidental shot).
3. **Burst & duplicate grouping with best-pick** per group.
4. **Fast ingest** (card copy + backup copy + hash verify), **embedded-preview culling**, keyboard-first review UI.
5. **XMP sidecar writeback** (rating, color label, reasons) readable by Lightroom / Capture One; cross-platform.
6. **Flash misfire detection**.

Out of MVP (see `06_ROADMAP.md`): AF-point check, NL search, multi-camera sync, OCR, banding, editing.

## Decisions
Each image gets exactly one decision:
- `keep`   — confident select
- `reject` — confident reject (always has ≥1 reason)
- `review` — uncertain band OR audit sample from the reject bucket (flag `audit=true`)

## Success metric
**Missed-keeper rate** = (# images the user kept/delivered that ShotSieve rejected) / (# images user kept).
Target ≤ 1%. Computed by `shotsieve eval` against a catalog's ground truth.
Secondary: review-bucket fraction (target ≤ 20%), images/s, peak RSS.

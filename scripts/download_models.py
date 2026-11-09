"""Download and verify all ShotSieve models into ./models."""

from __future__ import annotations

import sys

from shotsieve.models.registry import MODEL_SPECS, ensure_models, sha256_file


def main() -> int:
    try:
        paths = ensure_models()
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        size = path.stat().st_size
        ok = size == MODEL_SPECS[name].size
        print(
            f"{'OK ' if ok else 'BAD'} {name:16s} {size:>10d} B  sha256={sha256_file(path)[:16]}…"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

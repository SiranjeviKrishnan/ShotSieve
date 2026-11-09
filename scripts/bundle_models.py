# ruff: noqa: E501
"""Assemble the read-only model folder of the release app: `bundle_models.py SRC DEST`.

* Every file is verified against the sha256 pinned in the registry before it is copied.
* Only the variants the code loads are shipped: `siglip2_vision_int8` (the rejected int8
  alternative, setting `siglip_vision = "int8"`) is dropped, and `nafnet_sidd` ships as the
  CoreML-repaired graph (the only NAFNet file the runtime ever opens) instead of the raw fp16
  file plus a runtime-made copy.
* Writes `bundle.json` (name -> shipped filename, size, sha256) which marks the folder as a
  complete, read-only bundle: the app never downloads into it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from shotsieve.models.onnx_conv_fix import fix_conv_file
from shotsieve.models.registry import BUNDLE_MANIFEST, MODEL_SPECS, sha256_file

DROPPED = {"siglip2_vision_int8"}
DERIVED = {"nafnet_sidd": "nafnet_sidd_width32_coreml.onnx"}


def main(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    saved = 0
    for name, spec in sorted(MODEL_SPECS.items()):
        f = src / spec.filename
        if name in DROPPED:
            saved += spec.size
            print(f"  drop  {name:22s} {spec.size / 1e6:7.1f} MB (not loaded by default)")
            continue
        if not f.is_file() or f.stat().st_size != spec.size or sha256_file(f) != spec.sha256:
            print(f"FAIL: {name}: {f} missing or not the pinned file", file=sys.stderr)
            return 1
        if name in DERIVED:
            out = dest / DERIVED[name]
            fix_conv_file(f, out)
            saved += spec.size - out.stat().st_size
            ent = {"filename": out.name, "size": out.stat().st_size, "sha256": sha256_file(out),
                   "derived_from_sha256": spec.sha256}
            print(f"  ship  {name:22s} {ent['size'] / 1e6:7.1f} MB (CoreML-repaired {spec.filename})")
        else:
            out = dest / spec.filename
            shutil.copyfile(f, out)
            ent = {"filename": spec.filename, "size": spec.size, "sha256": spec.sha256}
            print(f"  ship  {name:22s} {spec.size / 1e6:7.1f} MB")
        ent["license"] = spec.license
        manifest[name] = ent
    (dest / BUNDLE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = sum(e["size"] for e in manifest.values())
    print(f"bundle: {len(manifest)} models, {total / 1e6:.0f} MB (not shipped: {saved / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: bundle_models.py SRC_MODELS_DIR DEST_DIR")
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))

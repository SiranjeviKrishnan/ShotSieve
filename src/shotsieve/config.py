"""Settings dataclass with TOML load/save."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import tomli_w


def project_root() -> Path:
    """Project root (the folder holding pyproject.toml), or CWD if installed elsewhere."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "shotsieve").exists():
            return parent
    return Path.cwd()


def bundle_resources() -> Path | None:
    """`ShotSieve.app/Contents/Resources` when this code runs from inside a built app bundle
    (scripts/build_release.sh), else None (repo checkout / dev install)."""
    for parent in Path(__file__).resolve().parents:
        if parent.name == "Resources" and parent.parent.name == "Contents" \
                and parent.parent.parent.suffix == ".app":
            return parent
    return None


def in_bundle() -> bool:
    return bundle_resources() is not None


def data_dir() -> Path:
    """Runtime data directory. SHOTSIEVE_DATA_DIR overrides; a built app uses
    ~/Library/Application Support/ShotSieve; a repo checkout uses <repo>/data."""
    env = os.environ.get("SHOTSIEVE_DATA_DIR")
    if env:
        path = Path(env)
    elif in_bundle():
        path = Path.home() / "Library" / "Application Support" / "ShotSieve"
    else:
        path = project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    """Log folder: SHOTSIEVE_LOG_DIR, else ~/Library/Logs/ShotSieve for a built app (unless
    SHOTSIEVE_DATA_DIR is set), else <data dir>/logs."""
    env = os.environ.get("SHOTSIEVE_LOG_DIR")
    if env:
        path = Path(env)
    elif in_bundle() and not os.environ.get("SHOTSIEVE_DATA_DIR"):
        path = Path.home() / "Library" / "Logs" / "ShotSieve"
    else:
        path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Settings:
    """All tunables. Defaults follow docs/03_PIPELINE.md."""

    # preview / batching
    preview_max_edge: int = 1280
    preview_workers: int = 4
    queue_size: int = 64
    embed_batch: int = 8
    max_faces: int = 6
    # runtime
    ort_threads: int = 6
    use_coreml: bool = False
    # grouping
    burst_gap_s: float = 2.0
    dup_window_s: float = 30.0
    dup_cos_dist: float = 0.12
    face_same_person_cos: float = 0.36
    # selection
    keep_per_group: int = 1
    tie_margin: float = 0.03
    t_keep: float = 0.60
    t_reject: float = 0.30
    personal_prior_n: int = 200
    t_soft: float = 0.55  # soft-failed frames with p >= this go to review, not reject
    # eyes
    eyes_closed_thresh: float = 0.55
    eyes_closed_both_thresh: float = 0.45
    # flash
    flash_window_s: float = 10.0
    flash_dark_ev_nofire: float = 1.3
    flash_dark_ev_fired: float = 1.5
    # focus / banding
    af_use_exiftool: bool = False  # fall back to exiftool for AF points the parser cannot read
    # xmp
    rating_keep: int = 3
    rating_review: int = 0
    rating_reject: int = 0
    label_keep: str = "Green"
    label_review: str = "Yellow"
    label_reject: str = "Red"
    # search / indexer (T10)
    search_index_auto: bool = True  # start the background indexer after a cull
    siglip_vision: str = "fp32"  # fp32 | int8 (int8 failed the gate: docs/02_MODELS.md)
    index_threads: int = 4
    index_batch: int = 8
    people_consent: bool = False  # default for shoots that never recorded a choice
    jersey_ocr: bool = False  # T11 opt-in: read jersey numbers in the indexer (adult sports)
    ocr_threads: int = 4
    # video frame grabs (T12, opt-in)
    video: bool = False
    video_fps: float = 2.0
    # subject-aware crop (T22)
    crop_write_xmp: bool = True  # write crs:Crop* to RAW sidecars on accept
    crop_min_long_edge_px: int = 1200
    crop_threads: int = 4
    crop_idle_unload_s: int = 120
    # personal
    min_rating: int = 1
    audit_seed: int = 0
    audit_enabled: bool = True  # sample rejects into review for the missed-keeper spot-check
    personal_model_path: str = ""  # empty -> <data dir>/personal_model.joblib
    # Lightroom sync (T20): colour-label name -> decision (Lightroom stores the label-set *name*)
    lr_label_map: dict[str, str] = field(default_factory=lambda: {"Red": "reject", "Green": "keep"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        """Build from a dict, ignoring unknown keys and coercing to the field types."""
        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        defaults = cls()
        for key, value in data.items():
            if key in known:
                kwargs[key] = type(getattr(defaults, key))(value)
        return cls(**kwargs)


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a TOML file; missing file or None yields defaults."""
    if path is None or not Path(path).exists():
        return Settings()
    with open(path, "rb") as fh:
        return Settings.from_dict(tomllib.load(fh))


def save_settings(settings: Settings, path: str | Path) -> None:
    """Write settings as TOML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        tomli_w.dump(settings.to_dict(), fh)

"""T14: the benchmark's HEVC RSS budget is 350 MB (H.264 stays at 300)."""

import importlib.util
import sys

from tests.conftest import ROOT


def _bench():
    spec = importlib.util.spec_from_file_location("bench_t14", ROOT / "scripts" / "benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_t14"] = mod
    spec.loader.exec_module(mod)
    return mod


def _h(rss: float) -> dict:
    return {
        "seconds": 30.0, "src_frames": 900, "backend": "avfoundation", "realtime_x": 5.9,
        "src_fps": 177.0, "frames": 60, "scored_fps": 20.0, "rss_added_mb": rss,
    }


def test_hevc_budget_constant_and_pass_at_320():
    b = _bench()
    assert b.HEVC_RSS_BUDGET_MB == 350
    text = "\n".join(b.hevc_lines(_h(320)))
    assert "PASS" in text and "target <= 350 MB: PASS" in text


def test_hevc_over_budget_at_360_and_h264_budget_unchanged():
    b = _bench()
    assert "OVER" in "\n".join(b.hevc_lines(_h(360)))
    t12 = {"seconds": 30.0, "src_frames": 900, "backend": "ffmpeg", "realtime_x": 6.0,
           "src_fps": 180.0, "frames": 60, "scored_fps": 2.0, "picks": 3, "rss_added_mb": 320.0}
    assert "FAIL" in "\n".join(b.t12_lines(t12, 1500.0))  # H.264 still capped at 300 MB

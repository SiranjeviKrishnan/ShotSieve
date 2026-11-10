# ruff: noqa: E501
"""Convert the RawNIND UtNet2 Bayer denoiser checkpoint to a static ONNX file (T26).

    uv run --no-project --with torch --with onnx --with onnxruntime --with "numpy<2" \
        python scripts/export_rawnind.py --download --out models/rawnind_utnet2_bayer_512.onnx

This is a clean-room implementation: the network below was written from the paper (Brummer and
De Vleeschouwer, arXiv:2501.08924, a plain 4-level U-Net) and from the names and shapes in the
checkpoint's state_dict. No code of the authors' repository (GPL-3.0) or of darktable is used or
copied. This script is Apache-2.0 like the rest of ShotSieve. Only the WEIGHTS are used, under their CC-BY-4.0 option (the authors publish them under
GPL-3.0 and CC-BY-4.0 both).

    Weights (c) Benoit Brummer, Christophe De Vleeschouwer, CC-BY-4.0
    Trained on RawNIND (CC-BY-SA-4.0, doi:10.14428/DVN/DEQCIM)
    Change made: converted from PyTorch to ONNX (static 1x4x512x512 -> 1x3x1024x1024), fp32.

Input: a Bayer mosaic packed as 4 planes (R, G1, G2, B; i.e. RGGB sites) of a black-subtracted,
white-normalised raw (0..1, camera RGB, no white balance). Output: demosaiced, denoised linear camera
RGB at twice the resolution. The network is kept in fp32: its activations reach about 6.5e4, which
overflows fp16.

`--verify-against REF.onnx` compares our file with another ONNX of the same checkpoint (a numerical
reference only; that file is never shipped) on random tiles and, with `--verify-raw RAW`, on real tiles.
Without `--download`, pass `--checkpoint iter_4350000.pt`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_NAME = "DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_mgout_notrans_valeither_-4/saved_models/iter_4350000.pt"
# the author's Google Drive folder (linked from the rawnind_jddc README): rawnind_denoise/<CHECKPOINT_NAME>
CHECKPOINT_URL = "https://drive.usercontent.google.com/download?id=1dFTLeljWi9wwojcZUsam8bE31JdYy3oM&export=download&confirm=t"
CHECKPOINT_SHA256 = "c1f7de3de24cb12f4803ee65e11cfb467e9a5b80c3cb7126746ef026106c15e0"
TILE = 512
LICENSE_NOTE = (
    "RawNIND UtNet2 Bayer denoiser weights (c) Benoit Brummer, Christophe De Vleeschouwer; "
    "CC-BY-4.0 (the authors dual-license the weights GPL-3.0 / CC-BY-4.0; the CC-BY-4.0 option is used). "
    "Trained on RawNIND (CC-BY-SA-4.0, doi:10.14428/DVN/DEQCIM). "
    "Source: https://github.com/trougnouf/rawnind_jddc, paper arXiv:2501.08924. "
    "Changes made: converted from PyTorch to ONNX (static 512 input, fp32)."
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_network():
    """UtNet2 for 4-plane Bayer input: four double-conv encoder levels (32..256 channels) with 2x2
    max-pooling, a 512-channel bottom, four decoder levels (2x2 transposed conv, concat with the
    skip, double conv) and a 1x1 conv to 12 channels that pixel-shuffles to 3 channels at 2x size.
    Every conv is 3x3 (padding 1) + LeakyReLU(0.2) except the last 1x1."""
    import torch
    from torch import nn

    def double(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.2),
        )  # fmt: skip

    class UtNet2(nn.Module):
        def __init__(self, funit: int = 32) -> None:
            super().__init__()
            f = funit
            self.convs1, self.convs2 = double(4, f), double(f, 2 * f)
            self.convs3, self.convs4 = double(2 * f, 4 * f), double(4 * f, 8 * f)
            self.bottom = double(8 * f, 16 * f)
            self.maxpool = nn.MaxPool2d(2)
            self.up1 = nn.ConvTranspose2d(16 * f, 8 * f, 2, stride=2)
            self.up2 = nn.ConvTranspose2d(8 * f, 4 * f, 2, stride=2)
            self.up3 = nn.ConvTranspose2d(4 * f, 2 * f, 2, stride=2)
            self.up4 = nn.ConvTranspose2d(2 * f, f, 2, stride=2)
            self.tconvs1, self.tconvs2 = double(16 * f, 8 * f), double(8 * f, 4 * f)
            self.tconvs3, self.tconvs4 = double(4 * f, 2 * f), double(2 * f, f)
            self.output_module = nn.Sequential(nn.Conv2d(f, 12, 1), nn.PixelShuffle(2))

        def forward(self, x):
            s1 = self.convs1(x)
            s2 = self.convs2(self.maxpool(s1))
            s3 = self.convs3(self.maxpool(s2))
            s4 = self.convs4(self.maxpool(s3))
            y = self.bottom(self.maxpool(s4))
            y = self.tconvs1(torch.cat([self.up1(y), s4], 1))
            y = self.tconvs2(torch.cat([self.up2(y), s3], 1))
            y = self.tconvs3(torch.cat([self.up3(y), s2], 1))
            y = self.tconvs4(torch.cat([self.up4(y), s1], 1))
            return self.output_module(y)

    return UtNet2()


def fetch_checkpoint(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or sha256_of(dest) != CHECKPOINT_SHA256:
        print(f"downloading the checkpoint (31 MB) from the authors' Google Drive folder -> {dest}")
        req = urllib.request.Request(CHECKPOINT_URL, headers={"User-Agent": "shotsieve-export"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    got = sha256_of(dest)
    if got != CHECKPOINT_SHA256:
        dest.unlink(missing_ok=True)
        sys.exit(f"checkpoint sha256 {got} != pinned {CHECKPOINT_SHA256}")
    return dest


def export(checkpoint: Path, out: Path) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    net = build_network()
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(
        sd, strict=True
    )  # same names and shapes as the checkpoint, nothing left over
    net.eval()
    x = torch.rand(1, 4, TILE, TILE)
    with torch.no_grad():
        ref = net(x).numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    kw = {"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}
    torch.onnx.export(
        net, x, str(out), opset_version=17, input_names=["input"], output_names=["output"],
        do_constant_folding=True, **kw,
    )  # fmt: skip
    m = onnx.load(str(out), load_external_data=True)
    for k, v in (
        ("license", LICENSE_NOTE),
        ("source_checkpoint_sha256", CHECKPOINT_SHA256),
        ("source_checkpoint", CHECKPOINT_NAME),
        (
            "input",
            "1x4x512x512 float32: Bayer RGGB planes (R, G1, G2, B), black-subtracted, 0..1, camera RGB",
        ),
        ("output", "1x3x1024x1024 float32: linear camera RGB, denoised and demosaiced"),
    ):
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(out), save_as_external_data=False)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    scale = float(np.abs(ref).max())  # random noise in drives the outputs to ~1e5: compare relative
    print(
        f"torch vs ONNX (CPU), random tile: max abs diff {float(np.abs(got - ref).max()):.3e} (output range {scale:.3e})"
    )
    print(f"wrote {out}  {out.stat().st_size} bytes  sha256 {sha256_of(out)}")


def verify(ours: Path, reference: Path, raw: Path | None) -> None:
    """Max abs difference to another ONNX of the same checkpoint (never shipped) on random tiles and,
    when `raw` is given, on real tiles of that Bayer RAW (needs rawpy)."""
    import numpy as np
    import onnxruntime as ort

    a = ort.InferenceSession(str(ours), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    tiles = [rng.random((1, 4, TILE, TILE), dtype=np.float32) * s for s in (1.0, 0.1, 0.01)]
    if raw is not None:
        import rawpy

        with rawpy.imread(str(raw)) as r:
            img = r.raw_image_visible.astype(np.float32)
            bl, wl = np.array(r.black_level_per_channel, np.float32), float(r.white_level)
            cols = r.raw_colors_visible
            norm = np.clip((img - bl[cols]) / (wl - bl[cols]), 0, 1)
        oy, ox = [(0, 0), (0, 1), (1, 0), (1, 1)][
            [
                (cols[0, 0], cols[0, 1], cols[1, 0], cols[1, 1]) == q
                for q in ((0, 1, 3, 2), (1, 0, 2, 3), (3, 2, 0, 1), (2, 3, 1, 0))
            ].index(True)
        ]  # offset that makes the mosaic start with R
        hh, ww = (norm.shape[0] - oy) // 2 * 2, (norm.shape[1] - ox) // 2 * 2
        norm = norm[oy : oy + hh, ox : ox + ww]
        pk = np.stack([norm[dy::2, dx::2] for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))])
        h, w = pk.shape[1:]
        for y0, x0 in ((h // 4, w // 4), (h // 2, w // 2), (h // 3, w // 5)):
            tiles.append(np.ascontiguousarray(pk[None, :, y0 : y0 + TILE, x0 : x0 + TILE]))
    worst = 0.0
    for i, t in enumerate(tiles):
        d = float(np.abs(a.run(None, {"input": t})[0] - b.run(None, {"input": t})[0]).max())
        worst = max(worst, d)
        print(f"  tile {i}: max abs diff {d:.3e}")
    print(f"max abs diff over {len(tiles)} tiles: {worst:.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--verify-only", type=Path, help="skip the export: compare this ONNX with --verify-against"
    )
    ap.add_argument("--checkpoint", type=Path, help="iter_4350000.pt of the Bayer denoiser")
    ap.add_argument(
        "--download", action="store_true", help="fetch it from the authors' Drive (sha256 pinned)"
    )
    ap.add_argument("--out", type=Path, default=Path("models/rawnind_utnet2_bayer_512.onnx"))
    ap.add_argument(
        "--verify-against",
        type=Path,
        help="another ONNX of the same checkpoint (numerical reference only)",
    )
    ap.add_argument(
        "--verify-raw", type=Path, help="a Bayer RAW file whose tiles are used for --verify-against"
    )
    args = ap.parse_args()
    if args.verify_only:
        if not args.verify_against:
            ap.error("--verify-only needs --verify-against")
        verify(args.verify_only, args.verify_against, args.verify_raw)
        return 0
    if args.checkpoint is None and not args.download:
        ap.error("give --checkpoint or --download")
    ck = args.checkpoint or fetch_checkpoint(
        args.out.parent / "rawnind_checkpoint" / "iter_4350000.pt"
    )
    if args.checkpoint is not None and sha256_of(ck) != CHECKPOINT_SHA256:
        sys.exit("this is not the pinned checkpoint (sha256 differs)")
    export(ck, args.out)
    if args.verify_against:
        verify(args.out, args.verify_against, args.verify_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

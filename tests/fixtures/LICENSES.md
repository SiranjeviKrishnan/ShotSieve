# Test fixture provenance

Every photo in this folder is CC0 / public domain (or a derivative of such a photo), or synthetic.
No fixture comes from a private photo library. Per-folder details, with source URLs, authors and
sha256:

| Folder | What | Provenance file |
|---|---|---|
| `real/` | 3 football JPEGs; first 256 KB of a Canon 5D III and a Sony A7 III RAW | `real/cc0_football.LICENSE.txt` (JPEGs), `real/raw_heads.LICENSE.txt` (RAW heads: raw.pixls.us URLs, sha256, CC0) |
| `sony_rx100m5a_cc0/` | first 512 KB of a Sony RX100 V ARW | `sony_rx100m5a_cc0/rx100m5a.LICENSE.txt` |
| `closed_eyes_cc0/` | one portrait (eyes closed) | `closed_eyes_cc0/SOURCE.txt` |
| `crop/`, `crop2/` | 44 + 25 Wikimedia Commons photos, CC0 / public domain (crop test sets) | `crop/LICENSES.md`, `crop2/LICENSES.md` |
| `shotlist_blind/`, `shotlist_blind2/` | CC0 / public domain Wikimedia Commons photos for the shot-list tests | `labels.json` (per-image source URL, author, licence) |
| `edit_styles/` | derived from the synthetic benchmark shoot and the CC0 fixtures above | `scripts/make_edit_styles.py` |
| `../../docs/viewer_quality/` | Pentax K-x loupe crops (docs only) | `docs/viewer_quality/LICENSE.txt` (raw.pixls.us CC0 sample) |
| `generated/` | not committed: built by `scripts/make_fixtures.py` | the script |
| `bin/exiftool` | a fake `exiftool` (Python) for tests | - |

# models/

Model weights live here but are **not committed** (about 1.4 GB). Fetch them, with sha256
verification against the pinned URLs in `src/shotsieve/models/registry.py`:

```bash
uv run python scripts/fetch_models.py          # everything the release app ships
uv run python scripts/fetch_models.py --core   # only the 131 MB core group
```

The release `.dmg` bundles every model inside the app; nothing is downloaded after install.
Licences and citations: `THIRD_PARTY_NOTICES.md`.

# Contributing to ShotSieve

Thanks for helping. ShotSieve is a local-first photo culling app; the ground rules are in
[docs/00_OVERVIEW.md](docs/00_OVERVIEW.md): everything runs on the user's machine, originals are
never modified, and nothing may need the network after install.

## Development setup (macOS, Apple Silicon)

```bash
git clone https://github.com/SiranjeviKrishnan/shotsieve && cd shotsieve
./install.sh --dev              # uv + Python 3.11 + locked deps (incl. dev tools) + core models,
                                # builds the dev ShotSieve.app that runs from this folder
uv run python scripts/fetch_models.py   # optional: every model (about 1.4 GB, sha256-verified)
```

Or by hand: `uv sync` (needs [uv](https://docs.astral.sh/uv/)), then
`uv run python scripts/fetch_models.py --core`.

Useful commands:

```bash
uv run shotsieve --help
uv run shotsieve serve <folder>        # the UI in a browser at 127.0.0.1
uv run shotsieve app                   # the native window
uv run shotsieve doctor                # install health check
```

## Tests, lint, privacy check

```bash
uv run ruff check .                                   # must be clean
uv run pytest -q -m "not models and not network and not ui and not install and not release and not large" --basetemp=.pytest_tmp_p
uv run pytest -q --basetemp=.pytest_tmp_p             # everything that is available locally
bash scripts/privacy_check.sh                         # nothing personal in files that will be published
uv run python scripts/gen_third_party.py --check      # THIRD_PARTY_NOTICES.md is current
```

Markers: `models` (needs downloaded models), `network`, `ui` (Playwright + Chromium:
`PLAYWRIGHT_BROWSERS_PATH=.playwright uv run playwright install chromium`), `large` (10k-frame
synthetic shoots), `install` (sandbox installs) and `release` (the built `dist/ShotSieve.app`,
`scripts/build_release.sh` first). Use a separate `--basetemp` per pytest process and never run two
pytest processes against the same one. Do not run the benchmark while the suite runs.

## Making a change

- **Every fix gets a regression test** that fails on the old code. Every feature needs tests.
- Keep to the budgets in `docs/` (memory <= 3 GB, cull >= 10 img/s on M1-class); `scripts/benchmark.py`
  measures them.
- No network calls after install, no telemetry. New models and dependencies must be
  permissively licensed (MIT / BSD / Apache-2.0) and pinned (commit + sha256 for models, `uv.lock`
  for packages); add them to `scripts/gen_third_party.py` and re-run it.
- **Test data must be CC0 / public domain** with its source recorded next to it (see
  `tests/fixtures/LICENSES.md`). Never add your own photos, catalogs or personal paths.
- Do not tune on held-out evaluation sets ("burned sets" in `docs/RESUME.md`); add a new one.
- Code style: `ruff` (line length 100). Prefer small, focused pull requests.

## Privacy rules for contributions

`scripts/privacy_check.sh` fails on absolute home paths (`/Users/<name>`), local scratch paths,
e-mail addresses (except noreply / example), `*.local` host names, key shapes and a few other
patterns, in tracked and untracked-but-not-ignored files. Use placeholders such as
`/Users/you/...` in docs and tests.

### Local privacy patterns

The committed guard knows only generic patterns. Put your own strings (your account name, mail
domain, host name, private folder names) one regex per line into the git-ignored
`.privacy_patterns.local` at the repository root; `scripts/privacy_check.sh` reads it when present
(CI runs without it, generic patterns only). Never write personal strings into a committed file.

## Releasing

```bash
scripts/build_release.sh              # dist/ShotSieve.app, dist/ShotSieve-<version>-arm64.dmg, .sha256
scripts/make_gpl_sources.sh           # dist/ShotSieve-<version>-gpl-sources.tar.gz (GPL/LGPL source)
uv run pytest -q -m release           # copies the app away from the repo and runs it offline
```

The flow for a published release: bump `version` (pyproject.toml and `src/shotsieve/__init__.py`),
push to `main`, wait for CI, then `git tag v<version> && git push origin v<version>`. **The tag
workflow `.github/workflows/release.yml` builds the `.dmg`, its `.sha256` and the gpl-sources
archive on a clean macOS runner and publishes all of them as the GitHub Release of that tag. Do not
also upload a locally built `.dmg` of the same name** (the two would race, and the local build
differs byte for byte). The local commands above are for testing a build before tagging. The runner
needs about 10 GB of free disk; the workflow frees space first and fails early below that.

Before the very first push, `scripts/prepare_repo.sh <github-username> <github-numeric-id>` sets the
repo-local git identity (GitHub noreply address), replaces the `SiranjeviKrishnan` placeholders and
runs the privacy check.

## Reporting bugs and ideas

Use the issue templates. For security problems see [SECURITY.md](SECURITY.md). By contributing you
agree that your contribution is licensed under the Apache License 2.0.

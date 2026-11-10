#!/bin/bash
# ShotSieve installer (macOS). Idempotent: running it again repairs / updates the install.
# Writes only inside this folder: .tools/ .uv-cache/ .uv-python/ .venv/ models/ data/ ShotSieve.app
#   ./install.sh [--open] [--dev]
# Offline / test overrides: SHOTSIEVE_UV_TARBALL=/path  SHOTSIEVE_MODELS_SRC=/dir  UV_OFFLINE=1
set -euo pipefail

UV_VERSION="0.11.4"
UV_SHA256_ARM64="9b9cb6c6f58c3246dbf3351ed4e97c500bc3266f5f237d2fd620b66e1c31dc56"
UV_SHA256_X86_64="c326edaf3fd492f53d1c58777f3459c0d87bf9dae8d89e80aec4b0da6622dcf3"
MIN_FREE_KB="${SHOTSIEVE_MIN_FREE_KB:-3145728}"   # 3 GB

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
OPEN_APP=0
SYNC_DEV="--no-dev"
for arg in "$@"; do
  case "$arg" in
    --open) OPEN_APP=1 ;;
    --dev) SYNC_DEV="" ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$ROOT/data/logs"
LOG="$ROOT/data/logs/install.log"
say() { echo "$*"; echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }
die() {  # die CODE message
  local code="$1"; shift
  echo "$*" >&2; echo "[$(date '+%H:%M:%S')] FAILED: $*" >> "$LOG"
  echo "Details: $LOG" >&2
  exit "$code"
}
run() {  # run a command quietly, log its output; fail with a plain message
  local what="$1"; shift
  echo "+ $*" >> "$LOG"
  if ! "$@" >> "$LOG" 2>&1; then
    tail -n 8 "$LOG" >&2
    die 1 "Failed: $what. Check your internet connection and run the install again."
  fi
}
echo "=== install $(date) ===" >> "$LOG"

# ---- 1. platform --------------------------------------------------------------------------
say "Checking this Mac..."
[ "${SHOTSIEVE_TEST_UNAME_S:-$(uname -s)}" = "Darwin" ] || die 2 "ShotSieve is macOS only for now."
ARCH="$(uname -m)"
case "$ARCH" in
  arm64) UV_ARCH="aarch64"; UV_SHA256="$UV_SHA256_ARM64" ;;
  x86_64) UV_ARCH="x86_64"; UV_SHA256="$UV_SHA256_X86_64"
          say "Note: this is an Intel Mac. ShotSieve will run on the CPU only (slower)." ;;
  *) die 2 "Unsupported CPU: $ARCH." ;;
esac
OSV="${SHOTSIEVE_TEST_OS_VERSION:-$(sw_vers -productVersion)}"
[ "${OSV%%.*}" -ge 14 ] || die 2 "ShotSieve needs macOS 14 or newer (this is $OSV)."
FREE_KB="$(df -k "$ROOT" | awk 'NR==2 {print $4}')"
if [ "$FREE_KB" -lt "$MIN_FREE_KB" ]; then
  die 3 "Not enough disk space: $((FREE_KB / 1024)) MB free, $((MIN_FREE_KB / 1024)) MB needed."
fi

# ---- 2. uv, pinned, inside this folder ----------------------------------------------------
UV_BIN="$ROOT/.tools/uv/uv"
have_uv() { [ -x "$UV_BIN" ] && [ "$("$UV_BIN" --version 2>/dev/null | awk '{print $2}')" = "$UV_VERSION" ]; }
if have_uv; then
  say "Installer tool already present."
else
  say "Fetching the installer tool (uv $UV_VERSION)..."
  TMP="$(mktemp -d "$ROOT/.tools.XXXXXX" 2>/dev/null)" || die 1 "Cannot create a temporary folder inside $ROOT (is it writable?)."
  trap 'rm -rf "$TMP"' EXIT
  TARBALL="$TMP/uv.tar.gz"
  if [ -n "${SHOTSIEVE_UV_TARBALL:-}" ]; then
    [ -f "$SHOTSIEVE_UV_TARBALL" ] || die 1 "SHOTSIEVE_UV_TARBALL not found: $SHOTSIEVE_UV_TARBALL"
    cp "$SHOTSIEVE_UV_TARBALL" "$TARBALL"
  else
    URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$UV_ARCH-apple-darwin.tar.gz"
    run "download uv" /usr/bin/curl -fL --retry 3 -o "$TARBALL" "$URL"
  fi
  GOT="$(/usr/bin/shasum -a 256 "$TARBALL" | awk '{print $1}')"
  if [ "$GOT" != "$UV_SHA256" ]; then
    die 4 "The installer tool download is damaged (checksum mismatch). Run the install again, or set SHOTSIEVE_UV_TARBALL to a good copy."
  fi
  mkdir -p "$TMP/x"
  /usr/bin/tar -xzf "$TARBALL" -C "$TMP/x" --strip-components=1
  mkdir -p "$ROOT/.tools"
  rm -rf "$ROOT/.tools/uv"
  mv "$TMP/x" "$ROOT/.tools/uv"
  have_uv || die 1 "The installer tool did not start. Run the install again."
fi

export UV_CACHE_DIR="$ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export UV_NO_MODIFY_PATH=1
export UV_NO_CONFIG=1  # a user-level uv.toml (index URL, etc.) must not change the install
export UV_TOOL_DIR="$ROOT/.tools/uv-tools"
export UV_PYTHON_BIN_DIR="$ROOT/.tools/bin"
export PIP_REQUIRE_VIRTUALENV=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
cd "$ROOT"

# ---- 3. Python + packages -----------------------------------------------------------------
say "Setting up Python 3.11..."
run "install Python" "$UV_BIN" python install 3.11 --no-bin
say "Installing ShotSieve's libraries (a few minutes on the first run)..."
# shellcheck disable=SC2086
run "install libraries" "$UV_BIN" sync --frozen $SYNC_DEV --inexact
SHOTSIEVE="$ROOT/.venv/bin/shotsieve"
[ -x "$SHOTSIEVE" ] || die 1 "Libraries did not install. Run the install again."

# ---- 4. core models -----------------------------------------------------------------------
say "Getting the core models (162 MB)..."
if ! "$SHOTSIEVE" models download >> "$LOG" 2>&1; then
  tail -n 6 "$LOG" >&2
  die 5 "The models could not be fetched or verified. Check your internet connection and run the install again."
fi

# ---- 5. app + doctor ----------------------------------------------------------------------
say "Building ShotSieve.app..."
run "build the app" /bin/bash "$ROOT/scripts/build_app.sh"
say "Warming up (first start would otherwise be slow)..."
run "warm up" "$ROOT/.venv/bin/python" -c "import shotsieve.desktop, webview, shotsieve.server.app, shotsieve.pipeline.runner, cv2, rawpy, sklearn; from shotsieve.models.runtime import coreml_available; coreml_available()"
say "Checking the install..."
if ! "$SHOTSIEVE" doctor | tee -a "$LOG"; then
  die 6 "The install check found a problem (see above)."
fi

# ---- 6. done ------------------------------------------------------------------------------
say "Done. Open ShotSieve.app: it is a Mac app window (drag it to your Dock to keep it)."
if [ "$OPEN_APP" = 1 ]; then
  open "$ROOT/ShotSieve.app" || true
fi

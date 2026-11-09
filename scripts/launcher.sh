#!/bin/bash
# ShotSieve.app launcher: checks the install, then EXECs the app process (`shotsieve app`).
# exec matters: the process LaunchServices started must itself be the window process, otherwise
# macOS tears the app's children down when this script exits (server died ~15 s after Finder open).
# build_app.sh replaces the ROOT line with the absolute install folder.
# Test knobs: SHOTSIEVE_DIALOG=echo (print instead of osascript)
ROOT='@ROOT@'
case "$ROOT" in @*) ROOT="$(cd "$(dirname "$0")/../../.." && pwd -P)" ;; esac

BIN="$ROOT/.venv/bin/shotsieve"
DATA="${SHOTSIEVE_DATA_DIR:-$ROOT/data}"
LOGDIR="$DATA/logs"
LOG="$LOGDIR/serve.log"

dialog() {
  if [ "${SHOTSIEVE_DIALOG:-}" = "echo" ]; then echo "$1"
  else /usr/bin/osascript -e 'on run argv' -e 'display dialog (item 1 of argv) buttons {"OK"} default button "OK" with title "ShotSieve" with icon caution' -e 'end run' "$1" >/dev/null 2>&1
  fi
}

if [ ! -x "$BIN" ]; then
  dialog "ShotSieve's folder moved or is incomplete. Run Install ShotSieve.command again."
  exit 1
fi
mkdir -p "$LOGDIR" || { dialog "ShotSieve cannot write to $DATA."; exit 1; }
if ! "$ROOT/.venv/bin/python" -c "import webview" >/dev/null 2>&1; then
  dialog "ShotSieve's window component is missing. Run Install ShotSieve.command again."
  exit 1
fi

# rotate the log at 5 MB (keep one old copy)
if [ -f "$LOG" ] && [ "$(/usr/bin/stat -f %z "$LOG")" -gt 5242880 ]; then mv -f "$LOG" "$LOG.1"; fi

cd "$ROOT" || exit 1
export SHOTSIEVE_DATA_DIR="$DATA"
ICON="$(cd "$(dirname "$0")/.." && pwd -P)/Resources/ShotSieve.icns"
[ -f "$ICON" ] && export SHOTSIEVE_APP_ICON="$ICON"
exec "$BIN" app >> "$LOG" 2>&1

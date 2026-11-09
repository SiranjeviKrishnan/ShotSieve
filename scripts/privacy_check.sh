#!/bin/bash
# Fail (exit 1) when personal data or forbidden references appear in the files that would be
# published: tracked files plus untracked files that are not ignored (or every file under DIR).
#
#   scripts/privacy_check.sh          # the repository (git, or .gitignore rules without a .git)
#   scripts/privacy_check.sh DIR      # every file under DIR (used by the tests)
#
# Committed patterns are generic only: absolute home paths (/Users/<name>), local scratch paths,
# e-mail addresses (except noreply / example), machine names (*.local), the product name of another
# company (no comparisons in public files) and API-key / token / private-key shapes.
# Personal patterns (account name and variants, mail domain, host name, private folders and names)
# live in the git-ignored file `.privacy_patterns.local` at the repository root, one regex per
# line, so they are never published; PRIVACY_PATTERNS_FILE overrides the path (tests). Without that
# file (CI) only the generic patterns run.
# Lines starting with `~` in that file are not applied to the third-party texts under licenses/.
# Generic placeholders (/Users/you, /Users/x, /Users/me, /Users/Shared, /Users/runner) are fine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
TARGET="${1:-}"
LIST="$(mktemp)"; trap 'rm -f "$LIST"; [ -z "${GD:-}" ] || rm -rf "$GD"' EXIT

if [ -n "$TARGET" ]; then
  ( cd "$TARGET" && find . -type f -not -path './.git/*' -print0 ) > "$LIST"
  BASE="$TARGET"
else
  BASE="$ROOT"
  cd "$ROOT"
  if [ -d .git ] && command -v git >/dev/null; then
    git ls-files --cached --others --exclude-standard -z | sort -zu > "$LIST"
  else  # no repository yet: apply the .gitignore rules with a throw-away git dir
    GD="$(mktemp -d)"; git init -q --bare "$GD"
    git --git-dir="$GD" --work-tree="$ROOT" ls-files -o --exclude-standard -z > "$LIST"
  fi
fi

python3 - "$BASE" "$LIST" "${PRIVACY_PATTERNS_FILE:-$ROOT/.privacy_patterns.local}" <<'PY'
import re, sys
from pathlib import Path

base, listing, local = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
PLACEHOLDER = r"(?:you|x|me|Shared|runner|name|user|username)(?![A-Za-z0-9._-])|[$<{%*]"
MAIL_OK = r"(?:users\.noreply\.github\.com|noreply\.github\.com|example\.(?:com|org|net)|localhost)(?![A-Za-z0-9-])"
PATTERNS = [
    ("absolute home path", rb"/Users/(?!" + PLACEHOLDER.encode() + rb")[A-Za-z0-9._-]+"),
    ("absolute home path (url-encoded)", rb"%2FUsers%2F(?!" + PLACEHOLDER.encode() + rb")[A-Za-z0-9._-]+"),
    ("local scratch path", rb"(?i)/private/tmp/clau" rb"de-|%2Fprivate%2Ftmp%2Fclau" rb"de-"),
    ("e-mail address", rb"(?i)[A-Za-z0-9._%+-]+@(?!" + MAIL_OK.encode() + rb")[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
    ("machine name (.local host)", rb"(?i)\b[A-Za-z0-9][A-Za-z0-9-]*\.local\b(?![A-Za-z0-9_/-])"),
    ("another company's product", rb"(?i)after[ _-]?shoot"),
    ("API key / token", rb"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|\bAKIA[0-9A-Z]{16}\b|\bsk-[A-Za-z0-9_-]{24,}|\bhf_[A-Za-z0-9]{30,}|\bxox[baprs]-[A-Za-z0-9-]{10,}|\bAIza[0-9A-Za-z_-]{35}"),
    ("private key", rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
]
local_note = ""
if local.is_file():
    for n, line in enumerate(local.read_text().splitlines(), 1):
        line = line.strip()
        if line and not line.startswith("#"):
            soft = line.startswith("~")  # "~regex": first names etc.; not applied to third-party licence texts
            try:
                PATTERNS.append((f"personal {'name ' if soft else ''}pattern (line {n} of {local.name})", line.lstrip("~").encode()))
            except Exception:
                sys.exit(f"privacy_check: bad pattern in {local}:{n}")
    local_note = f" (+ personal patterns from {local.name})"
else:
    local_note = " (generic patterns only; no local pattern file)"
compiled = [(n, re.compile(p)) for n, p in PATTERNS]
bad = 0
for rel in Path(listing).read_bytes().split(b"\0"):
    if not rel:
        continue
    name = rel.decode("utf-8", "replace").removeprefix("./")
    if name == local.name or ".dist-info/" in name:
        continue
    path = base / name
    if not path.is_file():
        continue
    data = path.read_bytes()  # binary files too: sqlite catalogs, PNG/JPEG metadata
    is_text = b"\0" not in data[:4096]
    for label, rx in compiled:
        third_party = name.startswith("licenses/")  # upstream authors' names and addresses are expected there
        if third_party and (label.startswith("e-mail") or "name pattern" in label):
            continue
        if not is_text and (label.startswith("personal name") or label.startswith("e-mail") or label.startswith("machine")):
            continue  # random byte runs in images/databases are not names or addresses
        for m in rx.finditer(data):
            line = data.count(b"\n", 0, m.start()) + 1
            shown = m.group(0)[:60].decode("utf-8", "replace")
            print(f"{name}:{line}: {label}: {shown!r}")
            bad += 1
            break  # one hit per pattern and file is enough
if bad:
    print(f"privacy_check: {bad} problem(s)", file=sys.stderr)
    sys.exit(1)
print("privacy_check: clean" + local_note)
PY

#!/bin/bash
# One-time helper before the first public commit.
#
#   scripts/prepare_repo.sh <github-username> <github-numeric-id>
#
# 1. refuses to run when a .git already exists, and when the privacy check fails
# 2. git init (branch main) with a REPO-LOCAL identity:
#       user.name  = <github-username>
#       user.email = <github-numeric-id>+<github-username>@users.noreply.github.com
#    (your global git identity is never read or changed, so it cannot leak into the history)
# 3. replaces every `<github-user>` placeholder (README incl. the URL-encoded %3Cgithub-user%3E badges,
#    CONTRIBUTING, pyproject.toml, THIRD_PARTY_NOTICES.md + its generator, build_release.sh, the
#    workflows) with the username; the bundle id of the release build becomes
#    io.github.<username>.shotsieve
# 4. runs the privacy check again and prints the next commands. It does NOT commit, add a remote,
#    push or tag.
#
# Your numeric GitHub id: https://api.github.com/users/<github-username>  ("id": ...), or
# GitHub > Settings > Emails shows the full noreply address "<id>+<username>@users.noreply.github.com".
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT"

usage() { echo "usage: scripts/prepare_repo.sh <github-username> <github-numeric-id>" >&2; exit 2; }
[ $# -eq 2 ] || usage
USER_NAME="$1"; USER_ID="$2"
[[ "$USER_NAME" =~ ^[A-Za-z0-9]([A-Za-z0-9-]{0,37}[A-Za-z0-9])?$ ]] || { echo "not a GitHub username: $USER_NAME" >&2; usage; }
[[ "$USER_ID" =~ ^[0-9]+$ ]] || { echo "the GitHub id must be numeric (see the comment at the top of this script)" >&2; usage; }

[ ! -e .git ] || { echo "refusing to run: .git already exists (this helper is for the very first commit only)." >&2; exit 1; }
command -v git >/dev/null || { echo "git not found" >&2; exit 2; }

echo "== privacy check (before anything is created)"
bash scripts/privacy_check.sh || { echo "refusing to proceed: the privacy check failed (see above)." >&2; exit 1; }

echo "== git init with a repo-local identity"
git init -q -b main
git config --local user.name "$USER_NAME"
git config --local user.email "${USER_ID}+${USER_NAME}@users.noreply.github.com"
echo "   user.name  = $(git config --local user.name)"
echo "   user.email = $(git config --local user.email)"

echo "== replacing the <github-user> placeholder with '$USER_NAME'"
LOWER="$(printf '%s' "$USER_NAME" | tr '[:upper:]' '[:lower:]')"
LIST="$(mktemp)"; trap 'rm -f "$LIST"' EXIT
git ls-files --cached --others --exclude-standard -z > "$LIST"
python3 - "$LIST" "$USER_NAME" <<'PY'
import sys
from pathlib import Path

listing, user = sys.argv[1], sys.argv[2]
# not rewritten: this helper itself and the historical review / task logs that describe the placeholder
SKIP = {"scripts/prepare_repo.sh", "docs/REVIEW.md", "docs/TASKS.md"}
changed = 0
for rel in Path(listing).read_bytes().split(b"\0"):
    name = rel.decode()
    if not name or name in SKIP:
        continue
    p = Path(name)
    if not p.is_file():
        continue
    data = p.read_bytes()
    if b"\0" in data[:4096]:
        continue
    new = data.replace(b"%3Cgithub-user%3E", user.encode()).replace(b"<github-user>", user.encode())
    if new != data:
        p.write_bytes(new)
        changed += 1
        print(f"   {name}")
print(f"   {changed} file(s) updated")
PY
LEFT="$(git ls-files --cached --others --exclude-standard | grep -vE '^(scripts/prepare_repo.sh|docs/REVIEW.md|docs/TASKS.md)$' | xargs grep -IlE '<github-user>|%3Cgithub-user%3E' 2>/dev/null || true)"
[ -z "$LEFT" ] || { echo "placeholder still present in: $LEFT" >&2; exit 1; }

echo "== privacy check (after)"
bash scripts/privacy_check.sh || { echo "the privacy check failed after the replacement (see above); fix it before committing." >&2; exit 1; }

cat <<EOF

== Done. Nothing has been committed. Next:

  cd "$ROOT"
  git status --short | head                     # review what will be published
  git add -A
  git commit -m "ShotSieve 0.1.0"
  git log --format='%an <%ae>'                  # must show: $USER_NAME <${USER_ID}+${USER_NAME}@users.noreply.github.com>
  # create an EMPTY repository https://github.com/${USER_NAME}/shotsieve on GitHub (no README / licence), then:
  git remote add origin https://github.com/${USER_NAME}/shotsieve.git
  git push -u origin main
  # wait for CI to turn green on GitHub, enable "Private vulnerability reporting" in Settings > Security, then:
  git tag v0.1.0
  git push origin v0.1.0                        # the Release workflow builds the .dmg + the gpl-sources archive and publishes both
  # bundle id of the build: io.github.${LOWER}.shotsieve
EOF

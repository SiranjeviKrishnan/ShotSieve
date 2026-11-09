"""scripts/privacy_check.sh: nothing personal may be published (T24).

The committed script only knows generic patterns (home paths, scratch paths, e-mail addresses, host
names, another company's product name, key shapes). Personal strings live in the git-ignored
`.privacy_patterns.local`; the tests below use made-up names and a temporary pattern file, so this
file contains nothing personal.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "privacy_check.sh"


def run(*args: str, patterns: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # never read the maintainer's real local file in the unit tests: point at a (maybe missing) file
    env["PRIVACY_PATTERNS_FILE"] = str(patterns or ROOT / "no-such-patterns-file")
    cmd = ["/bin/bash", str(SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def test_repository_is_clean():
    # with the maintainer's local pattern file when present (default path), generic-only in CI
    env = {k: v for k, v in os.environ.items() if k != "PRIVACY_PATTERNS_FILE"}
    r = subprocess.run(["/bin/bash", str(SCRIPT)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_guard_has_no_skip_list():
    # H1: the guard files used to be skipped and held the personal strings; they are scanned now
    assert "SKIP" not in SCRIPT.read_text()


@pytest.mark.parametrize("text", [
    "path = '/Users/" + "jdoe" + "/Pictures'",
    "see /Users/" + "jdoe" + "/Documents/x",
    "folder=%2FUsers%2F" + "jdoe" + "%2Fshoot",
    "/private/tmp/" + "claude-501/scratch",
    "contact someone.else@" + "uni.example.ac.uk",
    "mail me at someone@" + "gmail" + ".com",
    "host " + "my-laptop" + ".local is mine",
    "a" + "ftershoot" + " comparison",
    "After" + "Shoot Ultimate",
    "token " + "ghp_" + "a" * 36,
    "key " + "AKIA" + "ABCDEFGHIJKLMNOP",
    "-----BEGIN " + "RSA PRIVATE KEY-----",
])
def test_generic_patterns_are_caught(tmp_path, text):
    (tmp_path / "notes.md").write_text(f"hello\n{text}\n")
    r = run(str(tmp_path))
    assert r.returncode == 1, (text, r.stdout)
    assert "notes.md:2" in r.stdout


@pytest.mark.parametrize("text", [
    "placeholder /Users/you/Pictures/Wedding",
    "fake dialog /Users/x/Wedding and /Users/me/Pics and /Users/Shared",
    "CI wheel path /Users/runner/work/x and /Users/$(id -un)",
    "a code identifier sid and a subject id",
    "noreply 12345+someone@users.noreply.github.com and a@example.com",
    "server at localhost and 127.0.0.1",
])
def test_placeholders_pass(tmp_path, text):
    (tmp_path / "notes.md").write_text(text + "\n")
    r = run(str(tmp_path))
    assert r.returncode == 0, r.stdout


def test_local_patterns_are_loaded(tmp_path):
    pat = tmp_path / "patterns.txt"
    pat.write_text("# comment\n(?i)\\bfoo" + "bar\\b\n~\\bJane\\b\n")
    d = tmp_path / "repo"
    d.mkdir()
    (d / "a.md").write_text("a note about Foo" + "Bar here\n")
    (d / "b.md").write_text("photo of Jane Roe\n")
    (d / "c.md").write_text("nothing here\n")
    r = run(str(d), patterns=pat)
    assert r.returncode == 1
    assert "a.md:1" in r.stdout and "b.md:1" in r.stdout and "c.md" not in r.stdout
    # without the file the same tree is clean (CI: only the generic patterns run)
    assert run(str(d)).returncode == 0


def test_name_patterns_skip_third_party_licence_texts(tmp_path):
    pat = tmp_path / "patterns.txt"
    pat.write_text("~\\bJane\\b\n")
    d = tmp_path / "repo"
    (d / "licenses").mkdir(parents=True)
    mail = "jane@" + "vendor.test-domain" + ".org"
    (d / "licenses" / "x.txt").write_text(f"Copyright Jane Roe <{mail}>\n")
    assert run(str(d), patterns=pat).returncode == 0


def test_binary_files_are_scanned(tmp_path):
    (tmp_path / "x.db").write_bytes(b"\0\1\2" + b"/Users/" + b"jdoe" + b"/p\0")
    assert run(str(tmp_path)).returncode == 1

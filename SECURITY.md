# Security policy

## Model

ShotSieve is a local desktop app. After install it makes no network connections (the release build
bundles every model and is tested with outbound network denied). Its only listening socket is the
review UI server on `127.0.0.1` (random free port), which the app window talks to:

- bound to loopback only; the `Host` header must be a loopback name (DNS-rebinding protection);
- every state-changing request must carry a loopback `Origin` (cross-site requests are refused);
- file access goes through checked paths inside the folders you opened (no `..`, no NUL, symlink
  escapes resolved);
- your Lightroom catalog is opened read-only (a private copy if it uses WAL) and never written;
- photos are never modified; "Move to Trash" uses the system Trash only, strictly inside the shoot.

The `.dmg` is ad-hoc signed and **not notarised**: verify the download against the `.sha256` file on
the release page, and prefer building from source if you need to audit what you run.

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please report privately, not in a public issue:

1. Use GitHub's **Report a vulnerability** button on the repository's Security tab (private
   advisory), or
2. if that is unavailable, open a public issue that only says you have a security report and wait
   for a maintainer to reply with a private channel. Do not post details or exploit code.

Please include the version, macOS version, what you did and what happened. We aim to acknowledge a
report within 7 days and to publish a fix and an advisory once a patch is available. Bugs in the
bundled third-party libraries (see `THIRD_PARTY_NOTICES.md`) are best reported upstream as well.

## Out of scope

Anything that needs an attacker who can already run code as you or edit files in your home folder
(ShotSieve trusts its own data folder), and photos or videos that crash a third-party decoder
without a security impact (please file those as ordinary bugs).

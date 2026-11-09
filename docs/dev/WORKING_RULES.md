# Project conventions (development process)

These are the conventions the project was built under. They are kept here, out of the public
docs, for anyone who wants to reproduce the process; contributors do not have to follow the
review workflow, only the engineering rules (see `CONTRIBUTING.md`).

- Everything the project needs (tools, virtualenv, models, caches, data) lives inside the
  repository folder; nothing is installed system-wide.
- Work was split into builders (implement) and reviewers (advise and review independently).
- Keep status updates short. Automate compile, test, debug. Every feature ships with code and
  passing tests; every bug fix gets a regression test that fails on the old code first.
- Models must run smoothly on a 16 GB Apple Silicon Mac. Only permissive licences for models and
  new dependencies.
- When a task is finished, tick it in `docs/TASKS.md` and refresh its STATUS section.
- Never run two pytest processes against the same `--basetemp` at once; use a separate
  `--basetemp` per parallel lane. Never run a benchmark while the test suite runs.
- Held-out evaluation sets are scored once and never used for tuning ("burned sets" in
  `docs/RESUME.md`).

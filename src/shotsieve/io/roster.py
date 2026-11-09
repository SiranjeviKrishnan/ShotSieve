"""Roster CSV (`number,name[,team]`) parsing. Tolerates a BOM, whitespace, `#23`, quotes, a header
row, and comma / semicolon / tab delimiters."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RosterEntry:
    number: str
    name: str
    team: str = ""


class RosterError(ValueError):
    """The roster file has no usable rows."""


def normalize_number(raw: str) -> str:
    """'#07' -> '7'; '00' stays '00' (a distinct number in some sports)."""
    s = raw.strip().lstrip("#").strip()
    if not s.isdigit():
        return ""
    t = s.lstrip("0")
    return t or s


def parse_roster(source: str | bytes | Path) -> list[RosterEntry]:
    if isinstance(source, Path):
        source = source.read_bytes()
    text = source.decode("utf-8-sig", "replace") if isinstance(source, bytes) else source
    text = text.lstrip("﻿")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise RosterError("roster is empty")
    head = lines[0]
    delim = max(",;\t", key=head.count) if any(d in head for d in ",;\t") else ","
    seen: dict[tuple[str, str], RosterEntry] = {}  # (team, number): numbers repeat across teams
    for row in csv.reader(io.StringIO("\n".join(lines)), delimiter=delim, skipinitialspace=True):
        cells = [c.strip().strip('"').strip() for c in row]
        if len(cells) < 2:
            continue
        number = normalize_number(cells[0])
        if not number or not cells[1]:
            continue  # header row, blank name, or a non-numeric number
        team = cells[2] if len(cells) > 2 else ""
        seen[(team.lower(), number)] = RosterEntry(number, cells[1], team)
    if not seen:
        raise RosterError("no `number,name` rows found")
    return list(seen.values())

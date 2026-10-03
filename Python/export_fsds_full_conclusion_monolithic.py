#!/usr/bin/env python3
"""Inline \\input files into fsds_full_conclusion_po_en_monolithic.tex."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "docs" / "latex"
MAIN = LATEX / "fsds_full_conclusion_po_en.tex"
OUT = LATEX / "fsds_full_conclusion_po_en_monolithic.tex"


def main() -> int:
    text = MAIN.read_text(encoding="utf-8")

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if not name.endswith(".tex"):
            name = name + ".tex"
        path = LATEX / name
        if not path.is_file():
            return match.group(0)
        body = path.read_text(encoding="utf-8")
        return f"% --- inlined {name} ---\n{body}\n% --- end {name} ---\n"

    text = re.sub(r"\\input\{([^}]+)\}", repl, text)
    header = "% Monolithic full conclusion PO — export_fsds_full_conclusion_monolithic.py\n"
    OUT.write_text(header + text)
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

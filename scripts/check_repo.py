#!/usr/bin/env python3
"""Static checks that do not require datasets or model weights."""
from pathlib import Path
import ast
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []
for path in ROOT.rglob("*.py"):
    if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
        continue
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        errors.append(f"syntax: {path.relative_to(ROOT)}: {exc}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.resolve() != Path(__file__).resolve() and '/gly/' in text:
        errors.append(f"absolute shared path: {path.relative_to(ROOT)}")
for path in ROOT.rglob("*"):
    if path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
        errors.append(f"large file: {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
subprocess.run([sys.executable, "-m", "compileall", "-q", str(ROOT)], check=True)
print(f"OK: {sum(1 for _ in ROOT.rglob('*.py'))} Python files checked")

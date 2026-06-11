#!/usr/bin/env python3
"""Helper to run dr.py with API keys from .bashrc.

Loads every *_API_KEY= export from .bashrc into the subprocess env so
dr.py can read them via os.environ.get(). The early-return in .bashrc
(line 6-9) prevents these keys from being available in non-interactive
shells, hence this wrapper.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

ENV_VARS = ["OPENROUTER_API_KEY", "TAVILY_API_KEY"]


def _load_bashrc_value(var: str) -> str | None:
    bashrc = Path.home() / ".bashrc"
    if not bashrc.exists():
        return None
    for line in bashrc.read_text().splitlines():
        if line.startswith("export ") and "=" in line:
            kv = line[len("export "):].split("=", 1)
            if len(kv) == 2 and kv[0] == var:
                return kv[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    env = os.environ.copy()
    for var in ENV_VARS:
        val = _load_bashrc_value(var)
        if val:
            env[var] = val

    # Make the venv's `treval` (and any other venv binaries) findable
    # by the subprocess — sys.executable points at the venv's python.
    venv_bin = os.path.dirname(sys.executable)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    args = [sys.executable, "dr.py"] + sys.argv[1:]
    proc = subprocess.run(args, env=env, cwd=os.path.dirname(__file__) or ".")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

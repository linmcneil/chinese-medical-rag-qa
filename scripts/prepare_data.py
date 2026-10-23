"""Compatibility launcher: use python -m ragqa split --help for v2 arguments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragqa.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["split", *sys.argv[1:]]))

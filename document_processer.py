"""Compatibility launcher: use python -m ragqa build --help for v2 arguments."""
import sys

from ragqa.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build", *sys.argv[1:]]))

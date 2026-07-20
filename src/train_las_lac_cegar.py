"""Compatibility launcher for ``ldp train --agent ldp``."""

import sys

from ldp.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", "--agent", "ldp", *sys.argv[1:]]))

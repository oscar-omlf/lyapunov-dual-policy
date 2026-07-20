"""Compatibility launcher for ``ldp train --agent lac``."""

import sys

from ldp.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", "--agent", "lac", *sys.argv[1:]]))

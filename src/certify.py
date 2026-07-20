"""Compatibility launcher for ``ldp certify --agent lac``."""

import sys

from ldp.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["certify", "--agent", "lac", *sys.argv[1:]]))

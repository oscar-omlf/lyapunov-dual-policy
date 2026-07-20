"""Compatibility launcher for ``ldp certify --agent ldp``."""

import sys

from ldp.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["certify", "--agent", "ldp", *sys.argv[1:]]))

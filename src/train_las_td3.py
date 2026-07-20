"""Compatibility launcher for ``ldp train --agent las-td3``."""

import sys

from ldp.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", "--agent", "las-td3", *sys.argv[1:]]))

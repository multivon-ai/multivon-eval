"""Create a label-free TAT-QA input projection for the financial study."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tatqa_financial import UPSTREAM_REVISION, prepare_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream {UPSTREAM_REVISION}, got {revision}")
    manifest = prepare_inputs(args.upstream.resolve(), args.out.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

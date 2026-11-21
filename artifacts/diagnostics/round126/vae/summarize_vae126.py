#!/usr/bin/env python3
"""Aggregate completed R126 VAE receipts without importing Torch or writing data."""
import argparse
import json
from pathlib import Path
from contracts126 import aggregate_rank_receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="An already-completed remote/local receipt-only directory")
    args = parser.parse_args(argv)
    root = Path(args.run).resolve(strict=True)
    rows = [json.loads((root / ("rank%d.json" % rank)).read_text()) for rank in range(8)]
    print(json.dumps(aggregate_rank_receipts(rows), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

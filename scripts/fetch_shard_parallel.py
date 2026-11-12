#!/usr/bin/env python3
"""Parallel range downloader for the last few model shards.

The HF mirror throttles per connection (~2 MB/s), so a single stream would need
40 minutes for a 4.9 GB shard. This splits the file into N byte ranges, fetches
them concurrently with resume support, and concatenates them in order.

    python fetch_shard_parallel.py URL DEST [--conns 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path


def head_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, path: Path) -> None:
    """Download [start, end] into path, resuming an existing partial file."""
    have = path.stat().st_size if path.exists() else 0
    expected = end - start + 1
    if have >= expected:
        return
    headers = {"Range": f"bytes={start + have}-{end}"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, path.open("ab") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("dest")
    parser.add_argument("--conns", type=int, default=8)
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    total = head_size(args.url)
    print(f"total={total} bytes, conns={args.conns}", flush=True)

    if dest.exists() and dest.stat().st_size == total:
        print("already complete")
        return 0

    part_dir = dest.with_suffix(dest.suffix + ".parts")
    part_dir.mkdir(exist_ok=True)
    span = total // args.conns + 1

    ranges = []
    for index in range(args.conns):
        start = index * span
        end = min(start + span - 1, total - 1)
        if start <= end:
            ranges.append((index, start, end))

    errors: list[BaseException] = []

    def worker(index: int, start: int, end: int) -> None:
        try:
            fetch_range(args.url, start, end, part_dir / f"{index:02d}.part")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=r, daemon=True) for r in ranges]
    started = time.time()
    for thread in threads:
        thread.start()
    while any(t.is_alive() for t in threads):
        done = sum((part_dir / f"{i:02d}.part").stat().st_size
                   for i, _, _ in ranges if (part_dir / f"{i:02d}.part").exists())
        rate = done / max(time.time() - started, 1e-6) / 1e6
        print(f"\r{done/1e9:.2f}/{total/1e9:.2f} GB  {rate:.1f} MB/s", end="", flush=True)
        time.sleep(2)
    print()

    if errors:
        print(f"errors: {errors[:3]}", file=sys.stderr)
        return 1

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("wb") as out:
        for index, _, _ in ranges:
            out.write((part_dir / f"{index:02d}.part").read_bytes())
    size = tmp.stat().st_size
    if size != total:
        print(f"size mismatch: {size} != {total}", file=sys.stderr)
        return 1
    os.replace(tmp, dest)
    for index, _, _ in ranges:
        (part_dir / f"{index:02d}.part").unlink(missing_ok=True)
    part_dir.rmdir()
    print(f"done in {time.time()-started:.0f}s -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
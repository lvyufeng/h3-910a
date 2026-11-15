#!/usr/bin/env python3
"""Minimal read-only MP4 track metadata parser (no decoder dependency)."""
import json
from pathlib import Path
import struct
import sys


def boxes(data, start=0, end=None):
    end = len(data) if end is None else end
    while start + 8 <= end:
        size, typ = struct.unpack_from(">I4s", data, start)
        header = 8
        if size == 1:
            if start + 16 > end:
                raise ValueError("truncated extended box")
            size = struct.unpack_from(">Q", data, start + 8)[0]
            header = 16
        elif size == 0:
            size = end - start
        if size < header or start + size > end:
            raise ValueError(f"bad box {typ!r}")
        yield typ.decode("ascii", "replace"), data[start + header:start + size]
        start += size


def parse(path):
    data = Path(path).read_bytes()
    moov = next(payload for typ, payload in boxes(data) if typ == "moov")
    tracks = []
    for typ, track in boxes(moov):
        if typ != "trak":
            continue
        fields = dict(boxes(track))
        tkhd = fields["tkhd"]
        width, height = struct.unpack_from(">II", tkhd, len(tkhd) - 8)
        mdia = dict(boxes(fields["mdia"]))
        handler = mdia["hdlr"][8:12].decode("ascii")
        mdhd = mdia["mdhd"]
        offset = 20 if mdhd[0] == 1 else 12
        timescale = struct.unpack_from(">I", mdhd, offset)[0]
        duration = struct.unpack_from(">Q" if mdhd[0] == 1 else ">I", mdhd, offset + 4)[0]
        out = {"handler": handler, "width": width / 65536, "height": height / 65536,
               "timescale": timescale, "duration_seconds": duration / timescale}
        minf = dict(boxes(mdia["minf"]))
        if "stbl" in minf:
            stbl = dict(boxes(minf["stbl"]))
            if "stts" in stbl:
                stts = stbl["stts"]
                entries = struct.unpack_from(">I", stts, 4)[0]
                pairs = [struct.unpack_from(">II", stts, 8 + 8 * i) for i in range(entries)]
                out["sample_count"] = sum(n for n, delta in pairs)
                if handler == "vide" and duration:
                    out["frames_per_second"] = out["sample_count"] / out["duration_seconds"]
        tracks.append(out)
    return {"file": str(path), "bytes": len(data), "tracks": tracks}


if __name__ == "__main__":
    print(json.dumps(parse(sys.argv[1]), indent=2))

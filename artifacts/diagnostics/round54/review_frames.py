#!/usr/bin/env python3
"""Decode representative original output frames for honest visual sanity review."""
import argparse
import hashlib
import json
from pathlib import Path
import av

p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
root=Path(a.run_dir).resolve(strict=True)
assert root.name.startswith('h3_round54_e2e') and root.is_dir()
r=json.loads((root/'result.json').read_text());assert r['response']['status']=='completed'
video=root/'video.mp4';assert video.stat().st_size==r['video_bytes']
OUT=root/'frame_summary.json'
if OUT.exists():raise RuntimeError('Refuse to overwrite review evidence')
count=0;frames=[]
with av.open(str(video)) as container:
    for i,frame in enumerate(container.decode(video=0)):
        count+=1
        if i not in (0,62,123):continue
        path=root/f'frame_{i:03d}.png'
        if path.exists():raise RuntimeError('Review frame already exists')
        frame.to_image().save(path)
        arr=frame.to_ndarray(format='rgb24')
        frames.append({'index':i,'width':frame.width,'height':frame.height,'image_file':path.name,'pixel_std':float(arr.std())})
assert count==124
out={'decoded_frames':count,'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'frames':frames,
     'scope':'Selected visual review and full decoding/checksum; not official quality equivalence'}
OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

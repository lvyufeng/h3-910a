#!/usr/bin/env python3
"""Decode all native-baseline frames; preserve representative originals and metadata."""
import hashlib
import json
from pathlib import Path
import av

root=Path('/tmp/h3_round53_native49_e2e03')
result=json.loads((root/'result.json').read_text())
assert result.get('response',{}).get('status')=='completed'
video=root/'video.mp4'
out=root/'frame_summary.json'
if out.exists(): raise RuntimeError('Refuse to overwrite frame-review evidence')
frames=[]; count=0
with av.open(str(video)) as container:
    for i,frame in enumerate(container.decode(video=0)):
        count+=1
        if i not in (0,62,123): continue
        arr=frame.to_ndarray(format='rgb24')
        p=root/f'frame_{i:03d}.png'
        if p.exists(): raise RuntimeError('Frame exists')
        frame.to_image().save(p)
        frames.append({'index':i,'width':frame.width,'height':frame.height,'mean_rgb':arr.mean(axis=(0,1)).tolist(),
                       'pixel_std':float(arr.std()),'image_file':p.name})
assert count==124
r={'decoded_video_frames':count,'samples':frames,'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
   'scope':'Decoded content sanity and review frames only, not a numerical GPU/CPU reference-quality proof'}
out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))

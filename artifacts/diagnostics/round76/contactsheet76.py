#!/usr/bin/env python3
"""Reusable frame-contact-sheet deliverable from actually decoded frames,not mock data."""
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
INDICES=(0,12,24,36,48,62,74,86,98,110,123)


def build(root=ROOT,labels=('optimized','HF_reference_conditioning'),decoded=False):
    tilew,tileh=448,256;labelh=30;cols=4;rows=3
    for name in labels:
        sheet=Image.new('RGB',(cols*tilew,rows*(tileh+labelh)),(242,244,247));draw=ImageDraw.Draw(sheet)
        for position,index in enumerate(INDICES):
            path=root/(name+('_decoded_frame'if decoded else'_frame')+f'{index:03d}.png');image=Image.open(path).convert('RGB')
            if image.size!=(1344,768):raise ValueError('Unexpectedgeneratedframegeometry')
            x=(position%cols)*tilew;y=(position//cols)*(tileh+labelh)
            # Contact sheet is an explicit multi-frame comparison deliverable,
            # not a hidden resized substitute for inspecting original PNGs.
            sheet.paste(image.resize((tilew,tileh),Image.Resampling.LANCZOS),(x,y+labelh))
            draw.text((x+8,y+8),f'{name}: frame {index:03d} / {index/24:.2f}s',fill=(25,30,40))
        destination=root/(name+'_contactsheet.png')
        if destination.exists():raise RuntimeError('Refuse to overwrite earlier quality artifact')
        sheet.save(destination)
    return True


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--label',action='append',choices=['optimized','HF_reference_conditioning']);parser.add_argument('--decoded',action='store_true');args=parser.parse_args()
    build(labels=args.label or('optimized','HF_reference_conditioning'),decoded=args.decoded)

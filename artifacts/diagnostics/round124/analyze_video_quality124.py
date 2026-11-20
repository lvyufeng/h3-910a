#!/usr/bin/env python3
"""Inspect every generated frame and native audio without changing delivered media.

Red-ball connected-component tracking, time/finiteness/codec checks and a contact
sheet are diagnostic aids, not an automatic perceptual-quality score or a proof
that all possible logos/ghosts are absent. No pixel retouching, logo removal,
frame interpolation, audio gain normalization or model inference is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--reference',default=None)
    args=parser.parse_args()
    import av,numpy as np
    from scipy.ndimage import label
    from PIL import Image,ImageDraw
    video_path=Path(args.video);out=Path(args.output_root);out.mkdir(exist_ok=True)
    receipt=out/'all_frame_quality_inspection124.json';assert not receipt.exists()
    rows=[];selected_images={};previous=None;selected={0,12,24,36,48,62,74,86,98,110,123}
    with av.open(str(video_path)) as container:
        video=container.streams.video[0];audio=container.streams.audio[0]
        metadata={'width':video.width,'height':video.height,'fps':str(video.average_rate),'video_codec':video.codec_context.name,
                  'audio_codec':audio.codec_context.name,'sample_rate':audio.codec_context.sample_rate,'audio_channels':audio.codec_context.channels}
        for i,frame in enumerate(container.decode(video=0)):
            rgb=frame.to_ndarray(format='rgb24');r,g,b=[rgb[...,j].astype(np.int16) for j in range(3)]
            mask=(r>120)&(r-g>45)&(r-b>40)
            regions,count=label(mask)
            counts=np.bincount(regions.flatten())
            counts[0]=0;largest=int(counts.argmax()) if len(counts)>1 else 0
            yy,xx=np.where(regions==largest) if largest else (np.empty(0),np.empty(0))
            row={'frame_index':i,'pts_seconds':float(frame.time) if frame.time is not None else i/24,
                 'red_pixels':int(mask.sum()),'red_components_over100px':int((counts>=100).sum()),
                 'largest_red_component_area':int(len(xx)),
                 'largest_red_centroid_x':float(xx.mean()) if len(xx) else None,
                 'largest_red_centroid_y':float(yy.mean()) if len(yy) else None,
                 'largest_red_bbox':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None,
                 'luma_mean':float((0.2126*r+0.7152*g+0.0722*b).mean())}
            if previous is not None:row['RGB_RMSE_vs_previous_frame']=float(np.sqrt(np.mean((rgb.astype(np.float32)-previous.astype(np.float32))**2)))
            previous=rgb
            if i in selected:
                image=Image.fromarray(rgb);image.save(out/f'inspect_frame{i:03d}.png');selected_images[i]=image.resize((336,192))
            rows.append(row)
    with av.open(str(video_path)) as container:
        samples=[]
        for frame in container.decode(audio=0):
            array=frame.to_ndarray()
            if np.issubdtype(array.dtype,np.integer):array=array.astype(np.float64)/np.iinfo(array.dtype).max
            samples.append(array.astype(np.float64).flatten())
    wave=np.concatenate(samples) if samples else np.empty(0)
    report={'scope':__doc__,'completed':True,'media_sha256':hashlib.sha256(video_path.read_bytes()).hexdigest(),'metadata':metadata,
            'decoded_video_frames':len(rows),'all_frames_examined_by_diagnostic_tracker':True,
            'red_ball_detected_frames':sum(row['largest_red_component_area']>100 for row in rows),
            'frames_with_multiple_large_red_components':[row['frame_index'] for row in rows if row['red_components_over100px']>1],
            'first_ball_centroid':[rows[0]['largest_red_centroid_x'],rows[0]['largest_red_centroid_y']] if rows else None,
            'last_ball_centroid':[rows[-1]['largest_red_centroid_x'],rows[-1]['largest_red_centroid_y']] if rows else None,
            'native_encoded_audio_finite':bool(np.isfinite(wave).all()),'audio_sample_values':len(wave),
            'encoded_audio_peak':float(np.abs(wave).max()) if len(wave) else None,'encoded_audio_RMS':float(np.sqrt(np.mean(wave**2))) if len(wave) else None,
            'automatic_perceptual_quality_pass_claimed':False,'tracker_does_not_prove_logo_or_ghost_absence':True,'pixel_and_audio_modifications':False,'frames':rows}
    sheet=Image.new('RGB',(4*336,3*216),(245,245,245));draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(sorted(selected_images)):
        x=(j%4)*336;y=(j//4)*216;sheet.paste(selected_images[i],(x,y));draw.text((x+5,y+196),f'frame{i:03d}  {i/24:.2f}s',fill=(20,20,20))
    sheet.save(out/'all_frame_inspection_contact_sheet.png')
    receipt.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)


if __name__=='__main__':main()

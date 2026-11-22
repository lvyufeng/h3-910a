"""Pure media extraction/export helpers; no model inference or audio gain changes."""
from fractions import Fraction
from pathlib import Path


def find_frames(value):
    import av,numpy as np,torch
    from PIL import Image
    if isinstance(value,(bytes,bytearray)) and bytes(value)[4:8]==b'ftyp':
        import io
        with av.open(io.BytesIO(bytes(value))) as container:return np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
    if isinstance(value,(torch.Tensor,np.ndarray)):
        array=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value
        if array.ndim==5 and array.shape[0]==1:array=array[0]
        if array.ndim==4:
            if array.shape[-1]==3:return array
            if array.shape[0]==3:return array.transpose(1,2,3,0)
            if array.shape[1]==3:return array.transpose(0,2,3,1)
    if isinstance(value,(list,tuple)) and value and isinstance(value[0],Image.Image):return np.stack([np.asarray(x.convert('RGB')) for x in value])
    if isinstance(value,dict):
        for item in value.values():
            frames=find_frames(item)
            if frames is not None:return frames
    if isinstance(value,(list,tuple)):
        for item in value:
            frames=find_frames(item)
            if frames is not None:return frames
    return None


def find_audio(value):
    import numpy as np,torch
    if isinstance(value,dict):
        for name in ['audio','waveform','audio_output']:
            item=value.get(name)
            if isinstance(item,(np.ndarray,torch.Tensor)):
                item=item.detach().cpu().numpy() if isinstance(item,torch.Tensor) else item
                if item.ndim==3 and item.shape[0]==1:item=item[0]
                if item.ndim==2 and item.shape[0]==2:return item.astype(np.float32,copy=False)
        for item in value.values():
            audio=find_audio(item)
            if audio is not None:return audio
    return None


def export_av(frames,audio,path,fps=24,sample_rate=32000):
    import av,numpy as np
    with av.open(str(path),mode='w') as container:
        video=container.add_stream('libx264',rate=fps);video.width=frames.shape[2];video.height=frames.shape[1];video.pix_fmt='yuv420p'
        video.options={'crf':'18','preset':'medium'}
        sound=container.add_stream('aac',rate=sample_rate);sound.layout='stereo';sound.bit_rate=192000
        for i,array in enumerate(frames):
            frame=av.VideoFrame.from_ndarray(np.ascontiguousarray(array),format='rgb24');frame.pts=i;frame.time_base=Fraction(1,fps)
            for packet in video.encode(frame):container.mux(packet)
        for packet in video.encode():container.mux(packet)
        for i in range(0,audio.shape[1],1024):
            frame=av.AudioFrame.from_ndarray(np.ascontiguousarray(audio[:,i:i+1024],dtype=np.float32),format='fltp',layout='stereo')
            frame.sample_rate=sample_rate;frame.pts=i;frame.time_base=Fraction(1,sample_rate)
            for packet in sound.encode(frame):container.mux(packet)
        for packet in sound.encode():container.mux(packet)


def validate_av(path):
    import av
    with av.open(str(path)) as container:
        assert len(container.streams.video)==len(container.streams.audio)==1
        v=container.streams.video[0];a=container.streams.audio[0]
        result={'width':v.codec_context.width,'height':v.codec_context.height,'fps':str(v.average_rate),
                'video_codec':v.codec_context.name,'audio_codec':a.codec_context.name,'sample_rate':a.codec_context.sample_rate,'audio_channels':a.codec_context.channels}
        result['decoded_video_frames']=sum(1 for frame in container.decode(video=0))
    with av.open(str(path)) as container:result['decoded_audio_samples']=sum(frame.samples for frame in container.decode(audio=0))
    assert result['decoded_video_frames']==124 and result['decoded_audio_samples']>160000
    return result

"""Extract native returned arrays without casts, codec decoding, gains or pixel edits."""


def find_native_frames(value,torch,np):
    from PIL import Image
    if isinstance(value,(torch.Tensor,np.ndarray)):
        array=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value
        if array.ndim==5 and array.shape[0]==1:array=array[0]
        if array.ndim==4:
            if array.shape[-1]==3:return array
            if array.shape[0]==3:return array.transpose(1,2,3,0)
            if array.shape[1]==3:return array.transpose(0,2,3,1)
    if isinstance(value,(list,tuple)) and value and isinstance(value[0],Image.Image):
        assert all(image.mode=='RGB' for image in value),'Do not silently convert returned image modes'
        return np.stack([np.asarray(image) for image in value])
    if isinstance(value,dict):
        for item in value.values():
            result=find_native_frames(item,torch,np)
            if result is not None:return result
    if isinstance(value,(list,tuple)):
        for item in value:
            result=find_native_frames(item,torch,np)
            if result is not None:return result
    return None  # Encoded bytes are not a native-pixel metric reference.


def find_native_audio(value,torch,np):
    if isinstance(value,dict):
        for name in ('audio','waveform','audio_output'):
            item=value.get(name)
            if isinstance(item,(np.ndarray,torch.Tensor)):
                item=item.detach().cpu().numpy() if isinstance(item,torch.Tensor) else item
                if item.ndim==3 and item.shape[0]==1:item=item[0]
                if item.ndim==2 and item.shape[0]==2:return item  # No float32 upcast hiding the original dtype.
        for item in value.values():
            result=find_native_audio(item,torch,np)
            if result is not None:return result
    return None

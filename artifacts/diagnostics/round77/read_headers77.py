#!/usr/bin/env python3
"""Read only bounded safetensors headers to distinguish FP16 storage from BF16/F32 files."""
import collections
import hashlib
import json
from pathlib import Path
import struct
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parent
TARGETS={
 'official_dit_first':('MiniMaxAI/MiniMax-H3','0919fb684102ef115126612f3fbb84d1027e7317','FL2VA/transformer/model-00001-of-00013.safetensors'),
 'official_text_first':('MiniMaxAI/MiniMax-H3','0919fb684102ef115126612f3fbb84d1027e7317','FL2VA/text_encoder/model-00001-of-00014.safetensors'),
 'comfy_unpruned_dit':('Comfy-Org/MiniMax-H3','4b53a35191437de05370da548e59aeabe2a5a09a','diffusion_models/minimax_h3_fl2va_bf16.safetensors'),
 'comfy_unpruned_text':('Comfy-Org/MiniMax-H3','4b53a35191437de05370da548e59aeabe2a5a09a','text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors'),
 'comfy_video_vae_fp16':('Comfy-Org/MiniMax-H3','4b53a35191437de05370da548e59aeabe2a5a09a','vae/minimax_h3_video_vae_fp16.safetensors'),
}


def range_read(url,begin,end):
    request=urllib.request.Request(url+f'?h3_header_range={begin}-{end}',headers={'User-Agent':'H3-FP16-baseline-audit/1.0','Range':f'bytes={begin}-{end}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request,timeout=60)as response:
        content_range=response.headers.get('Content-Range');expected_prefix=f'bytes {begin}-{end}/'
        if response.status!=206 or content_range is None or not content_range.startswith(expected_prefix):
            raise ValueError('Server did not honor exact bounded byte range;model payload will not be read')
        data=response.read(end-begin+2);assert len(data)==end-begin+1
        return data,{'status':response.status,'content_range':content_range,'etag':response.headers.get('ETag')}


def inspect_header(name,target):
    model,revision,filename=target;url=f'https://huggingface.co/{model}/resolve/{revision}/{filename}'
    result={'url':url,'model':model,'revision':revision,'filename':filename,'weight_payload_downloaded':False}
    try:
        first,first_response=range_read(url,0,7);length=struct.unpack('<Q',first)[0]
        assert 0<length<=262144,'Refuse excessive safetensors header'
        data,header_response=range_read(url,8,7+length);header=json.loads(data)
        tensors={k:v for k,v in header.items()if k!='__metadata__'}
        result.update({'responses':[first_response,header_response],'header_bytes':length,'header_sha256':hashlib.sha256(data).hexdigest(),
                       'dtype_counts':dict(collections.Counter(v['dtype']for v in tensors.values())),'tensor_count':len(tensors),
                       'sample_tensors':{k:tensors[k]for k in list(tensors)[:5]},'metadata':header.get('__metadata__',{})})
        path=ROOT/'headers'/(name+'.json')
        with path.open('x')as f:f.write(json.dumps(header,indent=2)+'\n')
        result['header_path']=str(path.relative_to(ROOT));result['saved_header_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    except BaseException as error:result['failure']={'type':type(error).__name__,'message':str(error)}
    return result


if __name__=='__main__':
    (ROOT/'headers').mkdir(exist_ok=False)
    result={name:inspect_header(name,target)for name,target in TARGETS.items()}
    path=ROOT/'header_receipts77.json';assert not path.exists();path.write_text(json.dumps({'scope':__doc__,'bounded_no_tensor_payloads':True,'targets':result},indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in value.items()if k in('dtype_counts','tensor_count','failure')}for name,value in result.items()},indent=2))

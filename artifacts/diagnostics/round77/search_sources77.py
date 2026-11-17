#!/usr/bin/env python3
"""Bounded public GitHub/HF baseline discovery;never download model tensors or execute fetched code."""
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parent
LIMIT=1024*1024
URLS={
    'hf_name_search':'https://huggingface.co/api/models?search=MiniMax-H3&limit=100',
    'hf_fp16_name_search':'https://huggingface.co/api/models?search=MiniMax-H3-fp16&limit=20',
    'hf_official':'https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3',
    'hf_comfy':'https://huggingface.co/api/models/Comfy-Org/MiniMax-H3',
    'hf_deepbeep':'https://huggingface.co/api/models/DeepBeepMeep/MiniMax-H3',
    'hf_kijai_experimental':'https://huggingface.co/api/models/Kijai/MiniMax-H3-experimental',
    'hf_kijai_comfy':'https://huggingface.co/api/models/Kijai/MiniMax-H3_comfy',
    'hf_unsloth':'https://huggingface.co/api/models/unsloth/MiniMax-H3-GGUF',
    'hf_official_base_index':'https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/0919fb684102ef115126612f3fbb84d1027e7317/FL2VA/model_index.json',
    'gh_official_README':'https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/README.md',
    'hf_diffusers_documentation':'https://raw.githubusercontent.com/huggingface/diffusers/f53d552036a0d1bd5570782a39cd40cfabf112bc/docs/source/en/api/pipelines/minimax_h3.md',
    'gh_official_fp16_issues':'https://api.github.com/search/issues?q=repo%3AMiniMax-AI%2FMiniMax-H3+fp16&per_page=10',
    'gh_repository_fp16_search':'https://api.github.com/search/repositories?q=%22MiniMax-H3%22+fp16&per_page=10',
    'comfy_supported_models':'https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy/supported_models.py',
    'comfy_minimax_encoder':'https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy/text_encoders/minimax.py',
}


def fetch(item):
    name,url=item;request=urllib.request.Request(url,headers={'User-Agent':'H3-FP16-baseline-audit/1.0'})
    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        try:response=urllib.request.urlopen(request,timeout=45)
        except urllib.error.HTTPError as error:response=error
        with response:
            data=response.read(LIMIT+1);assert len(data)<=LIMIT,'Refuse unbounded search or large model data'
            data.decode('utf-8');status=response.status;resolved=response.geturl();etag=response.headers.get('ETag');content_type=response.headers.get('Content-Type')
        destination=ROOT/'search_sources'/(name+'.txt')
        with destination.open('xb')as f:f.write(data)
        result={'requested_url':url,'resolved_url':resolved,'status':status,'etag':etag,'content_type':content_type,'timestamp_utc':timestamp,
                'path':str(destination.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        if name.startswith('hf_')and name not in('hf_official_base_index','hf_diffusers_documentation')and status==200:
            parsed=json.loads(data)
            if isinstance(parsed,list):result['model_ids']=[row.get('id',row.get('modelId'))for row in parsed];result['count']=len(parsed)
            else:result['model_id']=parsed.get('id');result['revision']=parsed.get('sha');result['siblings']=[row['rfilename']for row in parsed.get('siblings',[])]
        if name.startswith('gh_')and status==200 and 'api.github.com/search/'in url:
            parsed=json.loads(data);result['total_count']=parsed.get('total_count');result['incomplete_results']=parsed.get('incomplete_results');result['items']=[{'url':row['html_url'],'title':row.get('title',row.get('name'))}for row in parsed.get('items',[])]
        return name,result
    except BaseException as error:return name,{'requested_url':url,'timestamp_utc':timestamp,'failure':{'type':type(error).__name__,'message':str(error)}}


if __name__=='__main__':
    (ROOT/'search_sources').mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=4)as workers:results=dict(workers.map(fetch,URLS.items()))
    payload={'scope':__doc__,'search_is_bounded_not_exhaustive':True,'searched_name_limit':100,'searched_fp16_name_limit':20,
             'no_model_weights_downloaded':True,'no_cpu_or_npu_inference_launched':True,'queries':results}
    output=ROOT/'search_receipts77.json';assert not output.exists();output.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({name:{k:v for k,v in r.items()if k in('status','count','model_id','revision','failure','total_count')}for name,r in results.items()},indent=2))

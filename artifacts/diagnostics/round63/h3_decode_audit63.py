"""Default-off actual H3 VAE latent/tile captures and bounded enqueue/event trace."""
import collections
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType
import torch

_STATES={}
_PROFILE_TYPES={'TransformerBlock','Attention','FeedForward','ViT3DDecoder'}


def _tensor_meta(x):
    return {'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride()),'device':str(x.device)}


class DecodeAudit:
    def __init__(self,vae,latent,root,rank):
        self.root=root/('rank%d'%rank);self.root.mkdir(parents=True,exist_ok=True)
        self.rank=rank;self.tile_index=-1;self.active=False;self.calls=0;self.events=[];self.stacks=collections.defaultdict(list)
        self.captured_bytes=0;self.handles=[];self.capture_skipped=[];self.tiles=[]
        self.meta={'rank':rank,'latent':_tensor_meta(latent),'torch_inference_mode':torch.is_inference_mode_enabled(),
                   'autocast_enabled':torch.is_autocast_enabled('npu'),'autocast_dtype':str(torch.get_autocast_dtype('npu')),
                   'decoder_class':type(vae.model.decoder).__name__,'decoder_config':dict(vae.model.decoder.config),
                   'scope':'Instrumented first real decode: first tile captures; only first two tile forwards traced. Events are inclusive stream/enqueue latencies, not an attribution/roofline or clean E2E proof.'}
        if rank==0:
            self.save('video_latent',latent,cap=64*1024*1024)
        decoder=vae.model.decoder
        dtype_counts=collections.Counter();linear_dtype=collections.Counter();bytes_=0;inference=collections.Counter();grad=collections.Counter()
        for p in decoder.parameters():
            dtype_counts[str(p.dtype)]+=1;bytes_+=p.numel()*p.element_size()
            inference[str(p.is_inference())]+=1;grad[str(p.requires_grad)]+=1
        for m in decoder.modules():
            if isinstance(m,torch.nn.Linear):linear_dtype[str(m.weight.dtype)]+=1
        self.meta.update(decoder_parameter_bytes=bytes_,decoder_parameter_dtypes=dict(dtype_counts),linear_weight_dtypes=dict(linear_dtype),
                         parameter_inference_tensor_counts=dict(inference),parameter_requires_grad_counts=dict(grad),
                         optimized_operator_installed=bool(getattr(decoder,'_omni_h3_vae_optimizations_installed',False)))
        self.handles.append(decoder.register_forward_pre_hook(self.root_pre))
        self.handles.append(decoder.register_forward_hook(self.root_post))
        for name,module in decoder.named_modules():
            if not name:continue
            if isinstance(module,(torch.nn.Linear,torch.nn.RMSNorm,torch.nn.LayerNorm,torch.nn.SiLU)) or type(module).__name__ in _PROFILE_TYPES:
                self.handles.append(module.register_forward_pre_hook(self.pre(name)))
                self.handles.append(module.register_forward_hook(self.post(name)))
            if type(module).__name__=='Attention' and hasattr(module,'_perform_attention'):
                original=module._perform_attention
                def attention(self_module,q,k,v,pack_info,original=original,name=name):
                    tag=name+'.perform_attention'
                    if self.tile_index==0 and name=='transformer_blocks.0.attn':
                        self.save_payload('attention0_inputs',{'q':q,'k':k,'v':v})
                        self.meta['attention0_pack_info']={key:str(type(value).__name__) for key,value in pack_info.items()}
                        self.meta['attention0_input_meta']={key:_tensor_meta(value) for key,value in {'q':q,'k':k,'v':v}.items()}
                    event=self.start(tag,q) if self.active else None
                    out=original(q,k,v,pack_info)
                    if event is not None:self.end(tag,event,out)
                    if self.tile_index==0 and name=='transformer_blocks.0.attn':self.save('attention0_output',out)
                    return out
                module._perform_attention=MethodType(attention,module)
        self.root.joinpath('instance.json').write_text(json.dumps(self.meta,indent=2,default=str)+'\n')

    def save(self,name,tensor,cap=128*1024*1024):
        path=self.root/(name+'.pt')
        if path.exists():return
        count=tensor.numel()*tensor.element_size()
        if count>cap or self.captured_bytes+count>512*1024*1024:
            self.capture_skipped.append({'name':name,'bytes':count,'reason':'bounded_capture'});return
        x=tensor.detach().cpu().contiguous().clone()
        with path.open('xb') as f:torch.save(x,f)
        self.captured_bytes+=count
        path.with_suffix('.json').write_text(json.dumps({'tensor':_tensor_meta(tensor),'bytes':count,
            'sha256':hashlib.sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest()},indent=2)+'\n')

    def save_payload(self,name,data):
        path=self.root/(name+'.pt')
        if path.exists():return
        count=sum(x.numel()*x.element_size() for x in data.values())
        if count>128*1024*1024 or self.captured_bytes+count>512*1024*1024:
            self.capture_skipped.append({'name':name,'bytes':count,'reason':'bounded_capture'});return
        cpu={key:value.detach().cpu().contiguous().clone() for key,value in data.items()}
        with path.open('xb') as f:torch.save(cpu,f)
        self.captured_bytes+=count

    def start(self,name,x):
        e=torch.npu.Event(enable_timing=True);e.record()
        return {'name':name,'tile':self.tile_index,'input':_tensor_meta(x),'start':e,'host_start':time.perf_counter()}

    def end(self,name,event,out):
        e=torch.npu.Event(enable_timing=True);e.record()
        event['end']=e;event['host_enqueue_ms']=(time.perf_counter()-event.pop('host_start'))*1000
        event['output']=_tensor_meta(out) if isinstance(out,torch.Tensor) else str(type(out).__name__)
        self.events.append(event)

    def pre(self,name):
        def hook(module,args):
            if self.tile_index==0:
                if name in ('transformer_blocks.0.norm1','transformer_blocks.17.norm1','transformer_blocks.35.norm1'):
                    self.save(name.replace('.','_')+'_input',args[0])
            if self.active and args and isinstance(args[0],torch.Tensor):self.stacks[name].append(self.start(name,args[0]))
            return None
        return hook

    def post(self,name):
        def hook(module,args,out):
            if self.active and self.stacks[name]:self.end(name,self.stacks[name].pop(),out)
            if self.tile_index==0 and name=='transformer_blocks.0.ff.w1':self.save('ff0_packed',out)
            return None
        return hook

    def root_pre(self,module,args):
        self.tile_index=self.calls;self.calls+=1;self.active=self.tile_index<2
        if self.active:
            self.tiles.append({'index':self.tile_index,'input':_tensor_meta(args[0])})
            if self.tile_index==0:self.save('decoder_tile_input',args[0])
            self.stacks['decoder'].append(self.start('decoder',args[0]))
        return None

    def root_post(self,module,args,out):
        if self.active:self.end('decoder',self.stacks['decoder'].pop(),out)
        if self.tile_index==0:self.save('decoder_tile_output',out)
        self.active=False
        return None

    def finish(self,decoded):
        torch.npu.synchronize()
        records=[]
        for event in self.events:
            start,end=event.pop('start'),event.pop('end')
            event['inclusive_stream_ms']=start.elapsed_time(end);records.append(event)
        self.meta.update(total_tile_decoder_calls=self.calls,profiled_tiles=self.tiles,captured_bytes=self.captured_bytes,
                         capture_skipped=self.capture_skipped,output=_tensor_meta(decoded),events=records)
        if self.rank==0:
            # Complete real reference frame output, bounded independently from
            # tile/operator captures. This is a one-step diagnostic reference.
            n=decoded.numel()*decoded.element_size();self.meta['full_output_reference_bytes']=n
            if n<=512*1024*1024:
                p=self.root/'decoded_video_reference.pt'
                if not p.exists():
                    x=decoded.detach().cpu().contiguous()
                    with p.open('xb') as f:torch.save(x,f)
                    self.meta['full_output_sha256']=hashlib.sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest()
        self.root.joinpath('decode_profile.json').write_text(json.dumps(self.meta,indent=2,default=str)+'\n')
        for handle in self.handles:handle.remove()


def begin(vae,latent):
    path=os.environ.get('VLLM_OMNI_910A_VAE_AUDIT_DIR')
    if not path or not torch.distributed.is_initialized():return None
    if id(vae) in _STATES:return None
    rank=torch.distributed.get_rank()
    root=Path(path)
    if (root/('rank%d'%rank)/'decode_profile.json').exists():return None
    state=DecodeAudit(vae,latent,root,rank);_STATES[id(vae)]=state
    return state


def finish(state,out):
    if state is not None:state.finish(out)
    return out

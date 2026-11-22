"""Audio-tail fingerprints and paired local-dispatch snapshots, original returns only.

Full projection input/rows are never subsetted for model computation. Selected
rows are for measurements only. D2H snapshots synchronize and are nonclean.
"""
from collections import defaultdict
import hashlib
import inspect
from pathlib import Path
from fingerprints129 import fingerprint,normalized_fingerprint,digest
from paired_words132 import compare_fp32


def callable_info(function):
    row={'module':getattr(function,'__module__',None),'qualified_name':getattr(function,'__qualname__',type(function).__qualname__),
        'type':type(function).__module__+'.'+type(function).__qualname__}
    try:path=inspect.getsourcefile(function)
    except TypeError:path=None
    row['source_path']=path
    if path and Path(path).is_file():row['source_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    else:row['implementation_source_unknown']=True
    return row


class AudioTailTrace132:
    def __init__(self,torch,model,audio_positions,seq_len):
        self.handles=[];self.saved=[];self.references=None;self.frozen=None;self.post_gather_reference=None
        try:self._initialize(torch,model,audio_positions,seq_len)
        except BaseException as primary:
            try:self.close()
            except BaseException as rollback:raise primary from rollback
            raise

    def _initialize(self,torch,model,audio_positions,seq_len):
        self.torch=torch;self.model=model;self.positions=audio_positions;self.positions_cpu=audio_positions.cpu();self.seq_len=seq_len
        assert seq_len==41792 and self.positions.numel()==414 and len(set(self.positions_cpu.tolist()))==414
        self.rows=[];self.ordinal=defaultdict(int);self.handles=[];self.saved=[];self.references=None;self.paired=[];self.frozen=None
        self.post_gather_reference=None;self.paired_post_gather=[]
        self.audio_layer=model.final_layer.audio_out;self.video_layer=model.final_layer.video_out
        expected_bytes=seq_len*5376*4
        available=None
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):available=int(line.split()[1])*1024
        free,total=torch.npu.mem_get_info()
        assert available is not None and available>=3*expected_bytes*8+2*1024**3,'Insufficient/unknown CPU budget for full-input pair'
        assert free>=2*expected_bytes+64*1024**2,'Insufficient device budget for the frozen full input'
        self.memory_budget={'full_input_bytes':expected_bytes,'host_available_bytes_before_capture':available,
            'device_free_bytes_before_capture':int(free),'device_total_bytes':int(total),'retained_full_CPU_inputs_max':1,'retained_full_NPU_inputs_max':1}
        for name,module in model.named_modules():
            if name not in {'sp_gather','time_embedder','final_layer','final_layer.norm','final_layer.adaln_proj','final_layer.video_out','final_layer.audio_out'}:continue
            def pre(owner,args,kwargs,name=name):
                ordinal=self.ordinal[name];self.ordinal[name]+=1
                self.rows.append({'module':name,'ordinal':ordinal,'boundary':'input','leaves':self.tree({'args':args,'kwargs':kwargs})})
                return None
            def post(owner,args,kwargs,output,name=name):
                self.rows.append({'module':name,'ordinal':self.ordinal[name]-1,'boundary':'output','leaves':self.tree(output)})
                if name=='final_layer.audio_out':
                    tensor=output[0] if isinstance(output,tuple) else output
                    assert tensor.shape==(41792,32) and tensor.dtype==torch.float32
                    cpu=tensor.detach().cpu().contiguous().clone()
                    if self.post_gather_reference is None:self.post_gather_reference=cpu
                    self.paired_post_gather.append({'full_post_TP_output':compare_fp32(cpu,self.post_gather_reference,torch),
                        'audio_rows_post_TP_output':compare_fp32(cpu.index_select(0,self.positions_cpu),self.post_gather_reference.index_select(0,self.positions_cpu),torch)})
                return None  # Preserve original module return.
            self.handles.append(module.register_forward_pre_hook(pre,with_kwargs=True))
            self.handles.append(module.register_forward_hook(post,with_kwargs=True))
        # One wrapper per method instance; route by the actual layer identity.
        methods={id(layer.quant_method):layer.quant_method for layer in (self.video_layer,self.audio_layer)}
        self.dispatch_info={name:{'scope':'Installed Python apply/dispatch objects, not an identified device kernel',
            'apply':callable_info(layer.quant_method.apply),
            'gemm_impl':callable_info(getattr(layer.quant_method,'_gemm_impl',layer.quant_method.apply)),
            'weight':fingerprint(layer.weight,torch,include_pointers=True),'bias':fingerprint(layer.bias,torch,include_pointers=True)}
            for name,layer in (('video_out',self.video_layer),('audio_out',self.audio_layer))}
        self.original_audio_apply=self.audio_layer.quant_method.apply
        for method in methods.values():
            old=method.apply;attributes=getattr(method,'__dict__',{})
            had_override='apply' in attributes;previous_override=attributes.get('apply')
            def apply(layer,x,bias=None,*args,old=old,**kwargs):
                if layer is not self.audio_layer and layer is not self.video_layer:return old(layer,x,bias,*args,**kwargs)
                name='final_layer.audio_out.local_apply' if layer is self.audio_layer else 'final_layer.video_out.local_apply'
                ordinal=self.ordinal[name];self.ordinal[name]+=1
                self.rows.append({'module':name,'ordinal':ordinal,'boundary':'input','leaves':self.tree({'x':x,'weight':layer.weight,'bias':bias})})
                current=None
                if layer is self.audio_layer:
                    assert x.shape==(41792,5376) and x.dtype==layer.weight.dtype==torch.float32
                    assert bias is None or bias.dtype==torch.float32
                    assert not args and not kwargs,'Refuse an unknown changed original apply contract'
                    current={'x':x.detach().cpu().contiguous().clone(),'weight':layer.weight.detach().cpu().contiguous().clone(),
                        'bias':None if bias is None else bias.detach().cpu().contiguous().clone()}
                    if self.frozen is None:
                        self.original_input_layout={'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride()),
                            'storage_offset':x.storage_offset(),'device':str(x.device),'physical_NPU_format':'not_measured'}
                        self.frozen=(x.detach().clone(),None if bias is None else bias.detach().clone())
                result=old(layer,x,bias,*args,**kwargs)
                self.rows.append({'module':name,'ordinal':ordinal,'boundary':'output','leaves':self.tree(result)})
                if current is not None:
                    assert result.shape==(41792,16) and result.dtype==torch.float32
                    current['output']=result.detach().cpu().contiguous().clone()  # Before native TP gather.
                    input_after=normalized_fingerprint(fingerprint(x,torch));input_before=normalized_fingerprint(fingerprint(current['x'],torch))
                    input_after.pop('tensor_version',None);input_before.pop('tensor_version',None)
                    # Original CPU snapshot dtype/value/shape is checked; device/pointers are not a logical-input difference.
                    assert input_after['sha256_logical_words']==input_before['sha256_logical_words'] and input_after['shape']==input_before['shape']
                    if self.references is None:self.references=current
                    reference=self.references
                    pair={'full_input':compare_fp32(current['x'],reference['x'],torch),
                        'audio_row_input':compare_fp32(current['x'].index_select(0,self.positions_cpu),reference['x'].index_select(0,self.positions_cpu),torch),
                        'weight':compare_fp32(current['weight'],reference['weight'],torch),
                        'full_local_output_before_TP_gather':compare_fp32(current['output'],reference['output'],torch),
                        'audio_row_local_output_before_TP_gather':compare_fp32(current['output'].index_select(0,self.positions_cpu),reference['output'].index_select(0,self.positions_cpu),torch),
                        'actual_full_matrix_shape_not_reduced':True,'snapshots_immediate_before_mutation_or_gather':True}
                    if current['bias'] is not None:pair['bias']=compare_fp32(current['bias'],reference['bias'],torch)
                    else:pair['bias']={'both_none':reference['bias'] is None}
                    self.paired.append(pair)
                return result  # The observer never substitutes a snapshot or computes a replacement output.
            method.apply=apply;self.saved.append((method,old,apply,had_override,previous_override))

    def tree(self,value):
        torch=self.torch
        if isinstance(value,torch.Tensor):
            result={'full':normalized_fingerprint(fingerprint(value,torch,include_pointers=False)),'hash_only_not_word_count':True}
            if value.ndim and value.shape[0]==self.seq_len:result['audio_selected_rows']=normalized_fingerprint(fingerprint(value.index_select(0,self.positions),torch))
            return result
        if isinstance(value,dict):return {str(k):self.tree(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [self.tree(v) for v in value]
        return fingerprint(value,torch)

    def reset(self):self.rows=[];self.ordinal.clear()

    def close_hooks(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()
        for method,old,wrapper,had_override,previous_override in self.saved:
            assert method.apply is wrapper
            if had_override:method.apply=previous_override
            else:delattr(method,'apply')
        self.saved.clear()

    def close(self):
        self.close_hooks();self.references=None;self.frozen=None;self.post_gather_reference=None

    @staticmethod
    def compare(first,second):
        assert [(r['module'],r['ordinal'],r['boundary']) for r in first]==[(r['module'],r['ordinal'],r['boundary']) for r in second]
        rows=[{'module':a['module'],'ordinal':a['ordinal'],'boundary':a['boundary'],'all_leaf_SHA_equal':digest(a['leaves'])==digest(b['leaves'])} for a,b in zip(first,second)]
        return {'first_different_boundary':next((r for r in rows if not r['all_leaf_SHA_equal']),None),
            'comparisons':rows,'SHA_boundaries_not_all_full_word_proofs':True,'D2H_observers_change_schedule':True}

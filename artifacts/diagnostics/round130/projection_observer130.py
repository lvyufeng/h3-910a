"""Scoped process-private observations of original torch.mm and Tensor.add_ calls.

Never replaces model results, operands or arithmetic. Copies immediately at
mutation boundaries so a later inplace add cannot falsify a pre-add snapshot.
Instrumentation synchronization is nonclean and may mask a scheduling hazard.
"""
from __future__ import annotations
from contextlib import contextmanager
import os
import threading


class ProjectionObserver130:
    def __init__(self,torch,*,retain_operands=False):
        self.torch=torch;self.retain_operands=retain_operands
        self.pid=os.getpid();self.thread=threading.get_native_id()
        self.rows=[];self.saved_operands=[];self._active=False

    def cpu(self,value):return value.detach().cpu().contiguous().clone()

    @contextmanager
    def observe(self):
        assert not self._active;self._active=True
        torch=self.torch;mm=torch.mm;add=torch.Tensor.add_
        self.original_mm=mm;self.original_add=add;self.rows=[];self.saved_operands=[]
        def require():
            assert os.getpid()==self.pid and threading.get_native_id()==self.thread,'Observer is only valid on the original caller thread'
        def matrix(a,b,*args,**kwargs):
            require();left=self.cpu(a);right=self.cpu(b)
            result=mm(a,b,*args,**kwargs)
            output=self.cpu(result)  # Snapshot before caller can mutate GEMM0 storage.
            self.rows.append({'op':'mm','index':sum(r['op']=='mm' for r in self.rows),
                'input_a':left,'input_b':right,'output':output,'a_stride':tuple(a.stride()),'b_stride':tuple(b.stride()),
                'kwargs':{k:str(v) for k,v in kwargs.items()},'result_is_original':True})
            if self.retain_operands:self.saved_operands.append(('mm',a.detach().clone(),b.detach().clone(),dict(kwargs)))
            return result
        def addition(destination,source,*args,**kwargs):
            require();before=self.cpu(destination);operand=self.cpu(source)
            result=add(destination,source,*args,**kwargs)
            assert result is destination,'Original inplace Tensor.add_ should return the same tensor'
            self.rows.append({'op':'add_','index':sum(r['op']=='add_' for r in self.rows),'input_a':before,'input_b':operand,
                'output':self.cpu(result),'kwargs':dict(kwargs),'result_is_original_destination':True})
            if self.retain_operands:self.saved_operands.append(('add_',before.to(destination.device),operand.to(source.device),dict(kwargs)))
            return result
        try:
            torch.mm=matrix;torch.Tensor.add_=addition
            yield self
        finally:
            assert torch.mm is matrix and torch.Tensor.add_ is addition,'Do not overwrite third-party function changes'
            torch.mm=mm;torch.Tensor.add_=add;self._active=False

    def report(self,reference=None):
        torch=self.torch;reported=[]
        def compare(a,b):
            assert a.dtype==b.dtype and a.shape==b.shape
            raw=a.reshape(-1).view(torch.uint8);ref=b.reshape(-1).view(torch.uint8)
            different_bytes=int((raw!=ref).sum())
            diff=a.double()-b.double();den=float(b.double().square().sum().sqrt().clamp_min(1e-30))
            return {'shape':list(a.shape),'dtype':str(a.dtype),'different_bytes':different_bytes,'all_raw_words_equal':different_bytes==0,
                'max_abs_difference':float(diff.abs().max()),'relative_RMS_error':float(diff.square().sum().sqrt())/den}
        if reference is not None:assert [(r['op'],r['index']) for r in reference.rows]==[(r['op'],r['index']) for r in self.rows]
        for pos,row in enumerate(self.rows):
            other=reference.rows[pos] if reference is not None else row
            reported.append({'op':row['op'],'index':row['index'],'kwargs':row['kwargs'],
                'a':compare(row['input_a'],other['input_a']),'b':compare(row['input_b'],other['input_b']),
                'output':compare(row['output'],other['output']),
                'original_returns_preserved':True,'mutation_boundary_snapshots_immediate':True})
        return reported

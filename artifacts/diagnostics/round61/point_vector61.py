"""Direct FP16 indexed affine/gate TIK kernels with UB-cached tiny tables."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from tbe import tik
import tbe.common.platform as platform

ROOT=Path('/tmp/h3_round61_adaln01')
KERNELS=ROOT/'kernels'
_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def build_point(kind,rows,width,table_rows,cores,a_stride=None,b_stride=None):
    assert kind in ('mod','gate') and width%128==0 and table_rows<=6
    a_stride=width if a_stride is None else a_stride
    b_stride=width if b_stride is None else b_stride
    assert a_stride>=width and b_stride>=width
    platform.set_current_compile_soc_info('Ascend910A')
    name=f'point61_{kind}_r{rows}_d{width}_m{table_rows}_c{cores}_sa{a_stride}_sb{b_stride}_{_HASH}'
    obj=KERNELS/name/(name+'.o')
    if obj.exists():
        meta=json.loads(obj.with_suffix('.json').read_text());assert meta['blockDim']==cores
        return str(obj),name+'__kernel0',cores
    t=tik.Tik(disable_debug=True)
    xgm=t.Tensor('float16',(rows,width),name='point_xgm',scope=tik.scope_gm)
    agm=t.Tensor('float16',((table_rows-1)*a_stride+width,),name='point_agm',scope=tik.scope_gm)
    bgm=t.Tensor('float16',(rows,width) if kind=='gate' else ((table_rows-1)*b_stride+width,),name='point_bgm',scope=tik.scope_gm)
    igm=t.Tensor('int32',(((rows+7)//8)*8,),name='point_igm',scope=tik.scope_gm)
    ygm=t.Tensor('float16',(rows,width),name='point_ygm',scope=tik.scope_gm)
    with t.for_range(0,cores,block_num=cores) as core:
        a=t.Tensor('float16',(table_rows,width),name='point_a',scope=tik.scope_ubuf)
        b=t.Tensor('float16',(width,) if kind=='gate' else (table_rows,width),name='point_b',scope=tik.scope_ubuf)
        x=t.Tensor('float16',(width,),name='point_x',scope=tik.scope_ubuf)
        ids=t.Tensor('int32',(8,),name='point_ids',scope=tik.scope_ubuf)
        for table_row in range(table_rows):
            t.data_move(a[table_row,0],agm[table_row*a_stride],0,1,width//16,0,0)
        if kind=='mod':
            for table_row in range(table_rows):
                t.data_move(b[table_row,0],bgm[table_row*b_stride],0,1,width//16,0,0)
            # Cache exactly-rounded FP16 (1+scale), not a fused FP32 factor.
            total=table_rows*width//128
            for start in range(0,total,255):
                count=min(255,total-start)
                t.vadds(128,b[start*128],b[start*128],1.0,count,1,1,8,8)
        with t.for_range(0,(rows+cores-1)//cores) as step:
            row=core+step*cores
            with t.if_scope(row<rows):
                t.data_move(ids,igm[(row//8)*8],0,1,1,0,0)
                index=t.Scalar('int32',name='point_index');index.set_as(ids[row%8])
                t.data_move(x,xgm[row,0],0,1,width//16,0,0)
                if kind=='gate':
                    t.data_move(b,bgm[row,0],0,1,width//16,0,0)
                    t.vmul(128,b,b,a[index,0],width//128,1,1,1,8,8,8)
                    t.vadd(128,x,x,b,width//128,1,1,1,8,8,8)
                else:
                    t.vmul(128,x,x,b[index,0],width//128,1,1,1,8,8,8)
                    t.vadd(128,x,x,a[index,0],width//128,1,1,1,8,8,8)
                t.data_move(ygm[row,0],x,0,1,width//16,0,0)
    work=os.path.realpath(tempfile.mkdtemp(prefix='point61_build_',dir=ROOT))
    if Path(work).parent!=ROOT.resolve() or not Path(work).name.startswith('point61_build_'):
        raise RuntimeError('Unexpected point build directory')
    old=os.getcwd()
    try:
        os.chdir(work)
        t.BuildCCE(kernel_name=name,inputs=[xgm,agm,bgm,igm],outputs=[ygm])
        src=Path(work)/'kernel_meta'/(name+'.o');assert src.exists() and src.resolve()==src.absolute()
        obj.parent.mkdir(exist_ok=False)
        shutil.copyfile(src.with_suffix('.json'),obj.with_suffix('.json'));shutil.copyfile(src,obj)
    finally:
        os.chdir(old)
        if Path(work).resolve()==Path(work).absolute() and Path(work).parent==ROOT.resolve():shutil.rmtree(work)
    meta=json.loads(obj.with_suffix('.json').read_text());assert meta['blockDim']==cores
    return str(obj),name+'__kernel0',cores

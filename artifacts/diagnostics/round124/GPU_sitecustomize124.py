"""GPU worker bootstrap for trajectory capture through the previously successful pipeline."""
import os
if os.environ.get('H3_GPU_PIPELINE_TRAJECTORY124')=='1':
    os.environ.setdefault('H3_GPU_RESULT_ROOT',os.environ['H3_GPU_PIPELINE_CAPTURE_ROOT124'])
    os.environ.pop('H3_GPU_PIPELINE_TRAJECTORY124')
    try:
        from GPU_pipeline_hooks124 import install
        install()
    finally:
        os.environ['H3_GPU_PIPELINE_TRAJECTORY124']='1'

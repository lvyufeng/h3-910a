#!/usr/bin/env python3
"""Analytical same-input RF/Euler increment error from completed five-sigma checks.

Uses only already captured activations/predictions for CPU statistics, not any
CPU model inference or performance baseline. Under eta0, the exact-arithmetic
increment is (sigma_current-sigma_next)*velocity. This diagnostic reports absolute
increment error and reference/state energy; it is not an actual NPU scheduler
kernel test, a free-running trajectory metric, a49-step error forecast, or a
perceptual-quality proof. No tensor clipping, BF16 conversion, or weight loading.
"""
import json
from pathlib import Path

ROOT=Path('/tmp/h3_align12401')


def main():
    import torch
    torch.set_num_threads(2)
    receipt=ROOT/'all_sigma_update_error_diagnostics124.json';assert not receipt.exists()
    report=json.loads((ROOT/'NPU_sigma_checks124c/sigma_checks_rank0.json').read_text())
    assert report['completed'] and report['all_selected_sigma_gates_accepted']
    bundle=torch.load('/tmp/h3_align12301/same_initial_sampling123.pt',map_location='cpu',weights_only=True)
    target=torch.nonzero(bundle['packed']['update_mask'],as_tuple=False).flatten().long()
    results=[]
    for step in [0,10,25,40,48]:
        g=torch.load(ROOT/f'GPU_corrected_pipeline_snapshot_step{step:02d}.pt',map_location='cpu',weights_only=True)
        n=torch.load(ROOT/f'NPU_sigma_checks124c/NPU_prediction_step{step:02d}.pt',map_location='cpu',weights_only=True)
        assert g['step_index']==n['step_index']==step
        row={'zero_based_step':step}
        for label in ['video','audio']:
            x=g[label+'_rows'];gv=g[label+'_velocity'];nv=n[label+'_velocity']
            if label=='video':x=x.index_select(0,target);gv=gv.index_select(0,target);nv=nv.index_select(0,target)
            x=x.double();gv=gv.double();nv=nv.double();sigma=g['sigma_'+label].double()
            delta=float(sigma[0]-sigma[1]);d=nv-gv
            assert bool(torch.isfinite(x).all() and torch.isfinite(gv).all() and torch.isfinite(nv).all())
            state_rms=float(x.square().mean().sqrt());reference_rms=float(gv.square().mean().sqrt())
            rmse=float(d.square().mean().sqrt());inc_error=abs(delta)*rmse
            row[label]={'shape':list(x.shape),'sigma_current':float(sigma[0]),'sigma_next':float(sigma[1]),
                        'effective_Euler_delta_sigma':delta,'GPU_current_state_RMS':state_rms,'GPU_velocity_RMS':reference_rms,
                        'velocity_absolute_RMSE':rmse,'velocity_relative_RMS':rmse/max(reference_rms,1e-30),
                        'same_input_increment_error_RMSE':inc_error,'same_input_increment_max_abs_error':abs(delta)*float(d.abs().max()),
                        'increment_error_over_current_state_RMS':inc_error/max(state_rms,1e-30),'reference_velocity_not_near_zero_for_this_sample':reference_rms>1e-4}
        results.append(row)
    result={'scope':__doc__,'completed':True,'no_model_weights_loaded':True,'CPU_model_inference':False,'actual_NPU_scheduler_kernel_measurement':False,
            'same_input_only':True,'all5_prediction_gates_already_accepted':True,'rows':results}
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

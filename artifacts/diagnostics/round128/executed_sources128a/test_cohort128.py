#!/usr/bin/env python3
"""Eight-owned-rank TCP status and injected phase failure, no H3 model or weights."""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT=Path('/tmp/h3_overlap12801')


def main():
    import torch
    import torch_npu
    from cohort128 import Cohort128,CohortPhaseError
    rank=int(os.environ['RANK']);torch.npu.set_device(rank)
    torch.distributed.init_process_group('hccl',init_method='env://')
    report={'scope':__doc__,'rank':rank,'completed':False,'model_weights_loaded':False}
    try:
        guard=Cohort128(torch,fatal_destination=None,prefix='R128cohortunit',timeout_s=60)
        value=guard.run('all_success',lambda:rank);assert value==rank
        tensor=torch.tensor([1.],device=f'npu:{rank}')
        torch.distributed.all_reduce(tensor);assert float(tensor.cpu())==8
        def fail_one_rank():
            if rank==3:raise RuntimeError('Injected setup failure before any model collective')
            return True
        try:guard.run('one_rank_failure',fail_one_rank)
        except CohortPhaseError as exc:
            report['one_rank_failure_refused_by_all8_before_next_collective']=True
            report['failure_detail']=str(exc)
        else:raise AssertionError('Injected cohort failure was not refused')
        guard.run('all_restored_ready',lambda:True)
        tensor=torch.tensor([1.],device=f'npu:{rank}');torch.distributed.all_reduce(tensor)
        assert float(tensor.cpu())==8
        report['HCCL_after_controlled_TCP_phase_failure_still_usable']=True
        report['fatal_runtime_faults_not_claimed_recoverable']=True
        report['completed']=True
        path=ROOT/f'cohort_unit_rank{rank}128.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:torch.distributed.destroy_process_group()


if __name__=='__main__':main()

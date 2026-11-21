#!/usr/bin/env python3
"""Event-driven milestone updates to the already-claimed GitHub Issue3 only.

Waits for existing request markers using inotify. No model launch, data transfer,
resource claim, scheduled task, busy polling or quality acceptance.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    script=r'''
import json,sys
from pathlib import Path
root=Path('/tmp/h3_runtime12701');sys.path.insert(0,str(root))
from audit_completion127 import wait_complete
session=root/'engine_lifecycle127c'
for index,label in [(1,'r01_cold'),(2,'r02_warm'),(3,'r03_native_reload')]:
    ready=wait_complete(session/label,index,timeout_s=2400)
    rows=[json.loads(line) for p in session.glob('worker_runtime_pid*127.jsonl') for line in p.read_text().splitlines() if json.loads(line)['request_index']==index]
    ends=[r for r in rows if r['event']=='native_request_end'];steps=[r for r in rows if r['event']=='dit_step_end']
    assert len(ends)==8 and len(steps)==392 and sum(r['actual_Half_matrix_modules_this_step'] for r in steps)==60760
    result={'index':index,'label':label,'worker_pids':ready['pids'],'native_worker_rank_max_s':max(r['host_elapsed_s'] for r in ends),'strict_Half_checks':60760,'actual_steps_each_worker':49,'native_models_complete_not_driver_media_export':True}
    print(json.dumps(result),flush=True)
'''
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=60','-o','ServerAliveInterval=30',
        '-o','ServerAliveCountMax=20','-o','ControlMaster=no','-o','ControlPath=none','ts-217','/usr/bin/python3 -']
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    process.stdin.write(script);process.stdin.close()
    sys.path.insert(0,str(ROOT.parent/'github-execution'))
    from create_issues import comment_once
    for line in process.stdout:
        row=json.loads(line)
        text=f'''## R127c原生请求{row['index']}完成（模型边界，媒体导出另计）

{row['label']}：同8worker每rank49个原50块forward，60760 strict Half checks通过。
原生worker请求rank-max **{row['native_worker_rank_max_s']:.3f}s**。
PID列表：{row['worker_pids']}。数值仅是这个worker request边界；不是纯encoder／VAE／DiT或driver生成wall。
本评论由已有completion marker事件驱动，不busy-poll、不创建新模型任务。

仍需完整3request身份／source／media／加载状态验证，随后才关闭Issue或晋升任何优化。
共享TP2＋USP4／encoderTP8与原精度不改；同liveencoder重复误差不作为候选容忍阈值。
'''
        posted=comment_once(3,f"R127c-request-{row['index']}-native-complete",text)
        print(json.dumps({**row,'github_comment':posted['html_url']},ensure_ascii=False),flush=True)
    error=process.stderr.read();rc=process.wait()
    if rc:raise RuntimeError(error)


if __name__=='__main__':main()

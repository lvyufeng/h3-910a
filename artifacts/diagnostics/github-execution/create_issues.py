#!/usr/bin/env python3
"""Create the explicitly authorized H3 roadmap issues; reconcile before every write.

No compute, GitHub credential printing, blind retries, scheduling, or automatic
performance/precision promotion. Project setup is intentionally a separate step.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / 'roadmap_spec.json').read_text())
REPO = SPEC['repo']
BASE = f'repos/{REPO}'
RECEIPT = ROOT / 'issue_receipts.json'


def gh_json(args: list[str], payload: dict | None = None):
    command = ['gh', *args]
    raw = None
    if payload is not None:
        command += ['--input', '-']
        raw = json.dumps(payload, ensure_ascii=False)
    result = subprocess.run(command, input=raw, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'gh failed ({result.returncode}) for {args[:4]}: {result.stderr.strip()}\n'
                           'Re-read GitHub state before any retry; do not replay uncertain writes.')
    return json.loads(result.stdout) if result.stdout.strip() else None


def api(path: str, method='GET', payload=None):
    return gh_json(['api', '--method', method, '-H', 'Accept: application/vnd.github+json',
                    '-H', 'X-GitHub-Api-Version: 2022-11-28', path], payload)


def pages(path: str):
    result = gh_json(['api', '--method', 'GET', '--paginate', '--slurp', path])
    return [entry for page in result for entry in page]


def marker(key: str) -> str:
    return f"<!-- h3-roadmap:{SPEC['roadmap_key']}:{key} -->"


def save(state):
    # Read before updating, preserving human-added metadata outside managed keys.
    old = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    assert not old or old['repo'] == REPO
    old.update(state)
    old['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    RECEIPT.write_text(json.dumps(old, ensure_ascii=False, indent=2) + '\n')


def source_link(path: str) -> str:
    if path.startswith('https://'):
        return f'- {path}'
    ref = 'master' if path.startswith('artifacts/diagnostics/github-execution/') else SPEC['baseline_commit']
    return f'- [{Path(path).name}](https://github.com/{REPO}/blob/{ref}/{path})'


CONTRACT = '''## 所有实验共享的硬边界

- **TP持续开启**：当前DiT TP2＋USP4，live FP16 encoder TP8，8卡；新TP映射先核验真实shards。
- 原50主块×49实际forward、FP16矩阵与原FP32 patch/time/final/VAE、FP32残差/norm/wide输出；
  155strict Half checks／步，HF32关闭；不做BF16回退、clip、teacher hidden替换或student/LoRA。
- 同输入可测候选偏差先Restore／停用，写出数字并问用户；native noise band不自动许可。
- owned live-claim资源，权重复用主机缓存；新下载去用户指定4090，不本地下载／中转。
- private instance hooks，禁止共享SDK／驱动重置；清理private进程／IPC并release lease。
- 低干扰clean→candidate→restored测量；每个指标注明cold/warm、rank-max与实际scope；不相加嵌套spans。
- source/input/config/weight SHA、实际kernel/copy/collective命中、失败回执和关联commit/PR回写本Issue。

## 执行状态合同

`Backlog → Todo → In Progress → Done`；依赖未满足用`Blocked`，数值决定用`Awaiting approval`。
开始前读主Issue、本Issue、依赖与最新评论；评论实验合同后claim。完成可为Keep或Reject；
Reject不合并，Awaiting approval不自动推进。结束必须写结果／恢复状态／下一步，不只记在聊天里。
'''


def body_for(item, entries):
    if item['key'] == 'roadmap':
        return (marker('roadmap') + '\n\n' + item['objective'] + '\n\n' +
                (ROOT / 'PLAN.md').read_text() + '\n\n' +
                '<!-- H3-roadmap-tasklist:start -->\n子Issue创建后填入原生关系与清单。\n'
                '<!-- H3-roadmap-tasklist:end -->\n')
    parent = entries['roadmap']['url']
    dependencies = '\n'.join(f"- #{entries[key]['number']} — {entries[key]['url']}"
                             for key in item['dependencies']) or '- 无前置实验依赖。'
    work = '\n'.join(f'- [ ] {text}' for text in item['work'])
    acceptance = '\n'.join(f'- [ ] {text}' for text in item['acceptance'])
    sources = '\n'.join(source_link(path) for path in item['sources'])
    return f'''{marker(item['key'])}

主路线图：{parent}

- 优先级：**{item['priority']}**；组件：**{item['component']}**；阶段：**{item['phase']}**。
- 精度策略：**{item['precision']}**；初始执行状态：**{item['status']}**。
- 初始状态是建单时快照；后续以Project当前字段＋最新执行评论为准，不从本段推断正在运行。

## 目标与理由

{item['objective']}

## 依赖（同时登记GitHub原生blocked-by关系）

{dependencies}

## 工作清单

{work}

## 验收／退出标准

{acceptance}

{CONTRACT}

## 基线证据与官方资料

{sources}
'''


def comment_once(number, tag, text):
    key = f"<!-- {SPEC['roadmap_key']}:comment:{tag} -->"
    comments = pages(f'{BASE}/issues/{number}/comments?per_page=100')
    matches = [c for c in comments if key in c.get('body', '')]
    assert len(matches) <= 1
    if matches:
        return matches[0]
    return api(f'{BASE}/issues/{number}/comments', 'POST', {'body': key + '\n\n' + text})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Perform user-authorized GitHub writes')
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({'repo': REPO, 'issues': [{k: v[k] for k in ('key', 'title', 'dependencies')}
                                                 for v in SPEC['issues']], 'writes': False},
                         ensure_ascii=False, indent=2))
        return
    info = api(BASE)
    assert info['full_name'] == REPO and info['has_issues'] and info['permissions']['push']
    labels = {v['name'] for v in pages(f'{BASE}/labels?per_page=100')}
    for label in SPEC['labels']:
        if label['name'] not in labels:
            api(f'{BASE}/labels', 'POST', label)
    existing = pages(f'{BASE}/issues?state=all&labels=h3-roadmap&per_page=100')
    entries = {}
    for item in SPEC['issues']:
        matches = [v for v in existing if marker(item['key']) in v.get('body', '')]
        assert len(matches) <= 1, ('Duplicate marker, reconcile manually', item['key'])
        if matches:
            issue = api(f"{BASE}/issues/{matches[0]['number']}")
        else:
            issue = api(f'{BASE}/issues', 'POST', {
                'title': item['title'], 'body': body_for(item, entries), 'labels': item['labels']})
        entries[item['key']] = {'id': issue['id'], 'node_id': issue['node_id'],
                                'number': issue['number'], 'url': issue['html_url'],
                                'title': issue['title'], 'initial_status': item['status']}
        save({'schema': 1, 'repo': REPO, 'roadmap_key': SPEC['roadmap_key'], 'issues': entries,
              'all_issues_created': False, 'native_dependencies_verified': False})
        print(json.dumps({'key': item['key'], 'issue': issue['number'], 'url': issue['html_url']},
                         ensure_ascii=False), flush=True)
    root = entries['roadmap']['number']
    children = pages(f'{BASE}/issues/{root}/sub_issues?per_page=100')
    current_children = {v['id'] for v in children}
    for item in SPEC['issues'][1:]:
        row = entries[item['key']]
        if row['id'] not in current_children:
            api(f'{BASE}/issues/{root}/sub_issues', 'POST', {'sub_issue_id': row['id']})
        blocked = pages(f"{BASE}/issues/{row['number']}/dependencies/blocked_by?per_page=100")
        current = {v['id'] for v in blocked}
        for key in item['dependencies']:
            if entries[key]['id'] not in current:
                api(f"{BASE}/issues/{row['number']}/dependencies/blocked_by", 'POST',
                    {'issue_id': entries[key]['id']})
        verified = pages(f"{BASE}/issues/{row['number']}/dependencies/blocked_by?per_page=100")
        assert {entries[k]['id'] for k in item['dependencies']}.issubset({v['id'] for v in verified})
    baseline = entries['baseline']['number']
    receipt = comment_once(baseline, 'baseline-complete',
        '已完成，不重新跑模型：commit [4384e89](https://github.com/' + REPO +
        '/commit/4384e89) 保存真实TP／kernel、8×49／60760checks、完整124帧媒体、source SHA、'
        '失败回执和cleanup。R126 verifier --require-final --require-combined通过。'
        '\n\n此Issue关闭的是证据封存，不是性能／精度目标。新候选仍仅private trial；'
        '共享R124＋FA未改变；19.372s whole-step、142.143s cold VAE都未达长期目标。')
    current = api(f'{BASE}/issues/{baseline}')
    if current['state'] != 'closed':
        api(f'{BASE}/issues/{baseline}', 'PATCH', {'state': 'closed', 'state_reason': 'completed'})
    task_lines = ['## 子Issue与当前执行顺序', '']
    for item in SPEC['issues'][1:]:
        row = entries[item['key']]
        checkbox = 'x' if item['key'] == 'baseline' else ' '
        deps = ', '.join('#' + str(entries[k]['number']) for k in item['dependencies']) or '无'
        task_lines.append(f"- [{checkbox}] #{row['number']} — {item['priority']} · {item['component']} · "
                          f"初始{item['status']} · 前置：{deps}")
    task_lines += ['', '第一项执行：#' + str(entries['runtime-facts']['number']) +
                   '。后续实际状态以Project字段和Issue最新评论为准；此清单不是第二套状态数据库。',
                   '可测数值变化在对应候选Issue停用等待用户，不把可选近似分支作为无损主线依赖。']
    issue = api(f'{BASE}/issues/{root}')
    start = '<!-- H3-roadmap-tasklist:start -->'
    end = '<!-- H3-roadmap-tasklist:end -->'
    before, sep, rest = issue['body'].partition(start)
    middle, sep2, after = rest.partition(end)
    assert sep and sep2
    new_body = before + start + '\n' + '\n'.join(task_lines) + '\n' + end + after
    if new_body != issue['body']:
        api(f'{BASE}/issues/{root}', 'PATCH', {'body': new_body})
    children = pages(f'{BASE}/issues/{root}/sub_issues?per_page=100')
    assert {v['id'] for k, v in entries.items() if k != 'roadmap'}.issubset({v['id'] for v in children})
    save({'issues': entries, 'all_issues_created': True, 'native_dependencies_verified': True,
          'native_subissues_verified': True, 'baseline_completed_comment': receipt['html_url']})
    print(json.dumps({'completed': True, 'root_issue': entries['roadmap']['url'],
                      'issue_count': len(entries), 'native_subissues': len(children)},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)

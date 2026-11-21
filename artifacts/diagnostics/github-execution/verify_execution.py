#!/usr/bin/env python3
"""Read-only verification of the actual GitHub roadmap, dependency DAG and Project.

This checks persistent execution tracking, not performance or precision acceptance.
It never mutates GitHub, initializes board state, runs models or reads tokens.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

from create_issues import api, pages, marker
from create_project import read_project

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / 'roadmap_spec.json').read_text())
REPO = SPEC['repo']
BASE = f'repos/{REPO}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-project', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    receipt = json.loads((ROOT / 'issue_receipts.json').read_text())
    assert receipt['all_issues_created'] and receipt['native_dependencies_verified']
    assert receipt['repo'] == REPO and len(receipt['issues']) == 13
    entries = receipt['issues']
    issues = {}
    edges = []
    for item in SPEC['issues']:
        row = entries[item['key']]
        live = api(f"{BASE}/issues/{row['number']}")
        assert live['id'] == row['id'] and live['node_id'] == row['node_id']
        assert marker(item['key']) in live['body'] and live['html_url'] == row['url']
        blocked = pages(f"{BASE}/issues/{row['number']}/dependencies/blocked_by?per_page=100")
        expected = {entries[k]['id'] for k in item['dependencies']}
        assert expected.issubset({v['id'] for v in blocked}), ('Missing native dependency', row['number'])
        edges += [(row['number'], entries[k]['number']) for k in item['dependencies']]
        issues[item['key']] = {'number': row['number'], 'url': live['html_url'],
                               'title': live['title'], 'state': live['state'],
                               'assignees': [v['login'] for v in live['assignees']],
                               'comments': live['comments'], 'updated_at': live['updated_at'],
                               'blocked_by': [v['number'] for v in blocked]}
    children = pages(f"{BASE}/issues/{entries['roadmap']['number']}/sub_issues?per_page=100")
    assert {v['id'] for k, v in entries.items() if k != 'roadmap'}.issubset({v['id'] for v in children})
    assert issues['baseline']['state'] == 'closed'
    assert entries['optional-numeric']['number'] not in issues['integration']['blocked_by']
    result = {'scope': 'Actual GitHub execution-tracking verification; not model/precision/latency acceptance',
              'verified_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'repo': REPO,
              'roadmap': entries['roadmap']['url'], 'issues_verified': len(issues),
              'native_subissues_verified': len(children), 'native_dependency_edges_verified': len(edges),
              'native_dependency_edges': edges, 'issues': issues,
              'spec_sha256': hashlib.sha256((ROOT / 'roadmap_spec.json').read_bytes()).hexdigest(),
              'model_run_started': False, 'new_precision_policy_accepted': False}
    path = ROOT / 'project_receipt.json'
    if args.require_project or path.exists():
        stored = json.loads(path.read_text())
        assert stored['fully_configured_and_verified']
        project = read_project(stored['project_id'])
        assert project['number'] == 10 and REPO in {v['nameWithOwner'] for v in project['repositories']['nodes']}
        by_issue = {v['content']['id']: v for v in project['items']['nodes'] if v['content']}
        fields = {v['name']: v for v in project['fields']['nodes'] if v}
        required_fields = ('Status','Priority','Component','Phase','Precision gate','Depends on','Execution order')
        assert set(required_fields).issubset(fields)
        allowed_statuses = {'Backlog','Todo','In Progress','Blocked','Awaiting approval','Done'}
        assert allowed_statuses.issubset({v['name'] for v in fields['Status']['options']})
        ready = []
        for key, row in entries.items():
            item = by_issue[row['node_id']]
            assert item['id'] == stored['items'][key]['item_id']
            values = {v['field']['name']: v for v in item['fieldValues']['nodes'] if v and v.get('field')}
            assert set(required_fields).issubset(values)
            state = values['Status']['name']
            assert state in allowed_statuses
            issues[key]['project_status'] = state
            issues[key]['priority'] = values['Priority']['name']
            if issues[key]['state'] == 'open' and state == 'Todo':
                dependency_numbers = issues[key]['blocked_by']
                states = {v['number']: v['state'] for v in issues.values()}
                if all(states.get(number) == 'closed' for number in dependency_numbers):
                    ready.append(row['number'])
        assert {'执行看板','执行顺序','等待精度决定'}.issubset({v['name'] for v in project['views']['nodes']})
        result.update({'project': project['url'], 'project_items_verified': len(entries),
                       'project_repository_link_verified': True,
                       'project_public': project['public'], 'ready_issue_numbers': ready,
                       'views': project['views']['nodes'], 'project_fields_verified': list(required_fields)})
    result['execution_tracking_verified'] = True
    if args.output:
        assert not args.output.exists(), 'Use a fresh verification output; preserve prior receipts.'
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('issues','native_dependency_edges')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)

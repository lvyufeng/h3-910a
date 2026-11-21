#!/usr/bin/env python3
"""Create/reconcile the explicitly authorized GitHub Project for existing H3 issues.

Writes only the roadmap's marked Project, never unrelated projects. Initializes
new item fields only; re-running does not reset live execution state to the spec.
No token printing, resource acquisition, model execution or scheduled tasks.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

from create_issues import gh_json, api, pages, comment_once

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / 'roadmap_spec.json').read_text())
REPO = SPEC['repo']
OWNER = SPEC['owner']
RECEIPT = ROOT / 'project_receipt.json'
MARKER = f"<!-- h3-project:{SPEC['roadmap_key']} -->"

FIELDS_QUERY = '''query($id: ID!) {
 node(id:$id) { ... on ProjectV2 {
 id number title url public closed readme viewerCanUpdate
 repositories(first:100) { nodes { nameWithOwner } }
 fields(first:100) { nodes {
 ... on ProjectV2Field { id name dataType }
 ... on ProjectV2SingleSelectField { id name dataType options { id name color description } }
 } }
 items(first:100) { nodes {
 id content { ... on Issue { id number url state repository { nameWithOwner } } }
 fieldValues(first:100) { nodes {
 ... on ProjectV2ItemFieldSingleSelectValue { name optionId field { ... on ProjectV2SingleSelectField { id name } } }
 ... on ProjectV2ItemFieldTextValue { text field { ... on ProjectV2Field { id name } } }
 ... on ProjectV2ItemFieldNumberValue { number field { ... on ProjectV2Field { id name } } }
 } }
 } pageInfo { hasNextPage } }
 views(first:30) { nodes { id name layout } }
 } }
}'''


def gql(query, variables):
    value = gh_json(['api', 'graphql'], {'query': query, 'variables': variables})
    if value.get('errors'):
        raise RuntimeError(json.dumps(value['errors'], ensure_ascii=False))
    return value['data']


def read_project(project_id):
    value = gql(FIELDS_QUERY, {'id': project_id})['node']
    assert value and MARKER in (value['readme'] or '') and value['viewerCanUpdate']
    assert not value['items']['pageInfo']['hasNextPage'], 'Project has more than 100 items; paginate before editing.'
    return value


def save(state):
    old = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    assert not old or old['repo'] == REPO
    old.update(state)
    old['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    RECEIPT.write_text(json.dumps(old, ensure_ascii=False, indent=2) + '\n')


def project_readme(issues):
    return (MARKER + '\n\n# 原MiniMax-H3／910A执行看板\n\n主Issue：' +
            issues['roadmap']['url'] + '\n\n## 看板约定\n\n'
            '- Status：Backlog／Todo／In Progress／Blocked／Awaiting approval／Done。\n'
            '- Priority：P0先运行时，P1真实FA／GEMM／VAE与验收，P2能力／拓扑／硬件。\n'
            '- Component与Phase用于筛选；Precision gate必须逐项核验，不能从Done推断精度接受。\n'
            '- Done可以是Keep或Reject实验；Awaiting approval不合并。原生Issue dependencies为前置约束。\n'
            '- 先读Issue／评论／依赖，claim并记录experiment；结束回写证据与commit／PR。\n'
            '- 本看板创建不是性能目标完成，也不会自动占用机器或启动compute。\n\n' +
            (ROOT / 'PLAN.md').read_text())


def option_names(field):
    return {v['name']: v['id'] for v in field.get('options', [])}


def edit_option(project_id, item_id, field, name):
    option_id = option_names(field)[name]
    return gh_json(['project', 'item-edit', '--id', item_id, '--project-id', project_id,
                    '--field-id', field['id'], '--single-select-option-id', option_id,
                    '--format', 'json'])


def edit_text(project_id, item_id, field, text):
    return gh_json(['project', 'item-edit', '--id', item_id, '--project-id', project_id,
                    '--field-id', field['id'], '--text', text, '--format', 'json'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({'writes': False, 'title': SPEC['project_title'],
                          'owner': OWNER, 'issue_count': len(SPEC['issues'])}, ensure_ascii=False))
        return
    issue_receipts = json.loads((ROOT / 'issue_receipts.json').read_text())
    assert issue_receipts['all_issues_created'] and issue_receipts['native_dependencies_verified']
    issues = issue_receipts['issues']
    listed = gh_json(['project', 'list', '--owner', OWNER, '--limit', '100', '--format', 'json'])
    matches = [v for v in listed['projects'] if v['title'] == SPEC['project_title']]
    assert len(matches) <= 1, 'Ambiguous Project title; reconcile rather than duplicate.'
    if matches:
        project = matches[0]
        project_id = project['id']
        project_number = project['number']
        info = read_project(project_id)
    else:
        project = gh_json(['project', 'create', '--owner', OWNER,
                           '--title', SPEC['project_title'], '--format', 'json'])
        project_id = project['id']
        project_number = project['number']
        save({'schema': 1, 'repo': REPO, 'project_id': project_id, 'project_number': project_number,
              'project_url': project['url'], 'roadmap_key': SPEC['roadmap_key'],
              'fully_configured_and_verified': False, 'items': {}})
        gh_json(['project', 'edit', str(project_number), '--owner', OWNER,
                 '--description', 'Original H3 on8×910A: runtime→FA/GEMM→VAE→graph/topology; TP stays ON, numeric changes require user approval.',
                 '--readme', project_readme(issues), '--format', 'json'])
        info = read_project(project_id)
    if REPO not in {v['nameWithOwner'] for v in info['repositories']['nodes']}:
        subprocess.run(['gh', 'project', 'link', str(project_number), '--owner', OWNER,
                        '--repo', REPO], check=True)
    fields = {v['name']: v for v in info['fields']['nodes'] if v}
    status = fields['Status']
    existing_options = status['options']
    planned_statuses = [('Backlog','GRAY'),('Todo','BLUE'),('In Progress','YELLOW'),
                        ('Blocked','RED'),('Awaiting approval','PURPLE'),('Done','GREEN')]
    missing = [(name, color) for name, color in planned_statuses if name not in option_names(status)]
    if missing:
        # Preserve all existing option IDs and values. Append only our missing states.
        options = [{k: v[k] for k in ('id','name','color','description')} for v in existing_options]
        options += [{'name': n, 'color': c, 'description': 'H3 execution contract: see roadmap issue.'}
                    for n, c in missing]
        gql('''mutation($input:UpdateProjectV2FieldInput!) {
          updateProjectV2Field(input:$input) { projectV2Field { ... on ProjectV2SingleSelectField { id } } }
        }''', {'input': {'fieldId': status['id'], 'singleSelectOptions': options}})
    definitions = {
        'Priority': ('SINGLE_SELECT', ['P0','P1','P2']),
        'Component': ('SINGLE_SELECT', list(dict.fromkeys(v['component'] for v in SPEC['issues']))),
        'Phase': ('SINGLE_SELECT', list(dict.fromkeys(v['phase'] for v in SPEC['issues']))),
        'Precision gate': ('SINGLE_SELECT', ['既定基线','逐位门禁','需数值评估','先批准再执行']),
        'Depends on': ('TEXT', []),
        'Execution order': ('NUMBER', []),
    }
    for name, (kind, options) in definitions.items():
        if name in fields:
            assert fields[name]['dataType'] == kind, ('Conflicting field type', name)
            continue
        command = ['project', 'field-create', str(project_number), '--owner', OWNER,
                   '--name', name, '--data-type', kind, '--format', 'json']
        if options:
            command += ['--single-select-options', ','.join(options)]
        gh_json(command)
    info = read_project(project_id)
    fields = {v['name']: v for v in info['fields']['nodes'] if v}
    by_issue = {v['content']['id']: v for v in info['items']['nodes'] if v['content']}
    stored = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    recorded_items = stored.get('items', {})
    for order, item in enumerate(SPEC['issues']):
        row = issues[item['key']]
        existing = by_issue.get(row['node_id'])
        if existing:
            item_id = existing['id']
        else:
            created = gh_json(['project', 'item-add', str(project_number), '--owner', OWNER,
                               '--url', row['url'], '--format', 'json'])
            item_id = created['id']
        current_values = {v['field']['name']: v for v in existing['fieldValues']['nodes']
                          if v and v.get('field')} if existing else {}
        initial_values = {'Status': item['status'], 'Priority': item['priority'],
                          'Component': item['component'], 'Phase': item['phase'],
                          'Precision gate': item['precision']}
        for name, value in initial_values.items():
            # A first add has a default Todo, which must be replaced with true initial status.
            # A previously recorded item is live state: never reset its status on reconciliation.
            if name == 'Status' and item['key'] in recorded_items:
                continue
            if name != 'Status' and name in current_values:
                continue
            edit_option(project_id, item_id, fields[name], value)
        if 'Depends on' not in current_values:
            dependency_text = ', '.join('#' + str(issues[k]['number']) for k in item['dependencies']) or 'None'
            edit_text(project_id, item_id, fields['Depends on'], dependency_text)
        if 'Execution order' not in current_values:
            gh_json(['project', 'item-edit', '--id', item_id, '--project-id', project_id,
                     '--field-id', fields['Execution order']['id'], '--number', str(order),
                     '--format', 'json'])
        recorded_items[item['key']] = {'item_id': item_id, 'issue_number': row['number'], 'url': row['url']}
        save({'repo': REPO, 'project_id': project_id, 'project_number': project_number,
              'project_url': info['url'], 'roadmap_key': SPEC['roadmap_key'],
              'items': recorded_items, 'fully_configured_and_verified': False})
        print(json.dumps({'issue': row['number'], 'project_item': item_id}, ensure_ascii=False), flush=True)
    final = read_project(project_id)
    views = {v['name']: v for v in final['views']['nodes']}
    visible_ids = [fields[name]['id'] for name in
                   ('Status','Priority','Component','Phase','Precision gate','Depends on','Execution order')]
    for view_name, layout, filter_text in [
            ('执行看板', 'BOARD_LAYOUT', None),
            ('执行顺序', 'TABLE_LAYOUT', None),
            ('等待精度决定', 'BOARD_LAYOUT', 'status:"Awaiting approval"')]:
        if view_name in views:
            continue
        created = gql('''mutation($input:CreateProjectV2ViewInput!) {
          createProjectV2View(input:$input) { projectV2View { id name layout } }
        }''', {'input': {'projectId': project_id, 'name': view_name, 'layout': layout,
                         'configuration': {'visibleFieldIds': visible_ids}}})['createProjectV2View']['projectV2View']
        if filter_text:
            gql('''mutation($input:UpdateProjectV2ViewInput!) {
              updateProjectV2View(input:$input) { projectV2View { id name layout } }
            }''', {'input': {'viewId': created['id'], 'filter': filter_text}})
    final = read_project(project_id)
    assert {'执行看板','执行顺序','等待精度决定'}.issubset({v['name'] for v in final['views']['nodes']})
    assert REPO in {v['nameWithOwner'] for v in final['repositories']['nodes']}
    final_by_issue = {v['content']['id']: v for v in final['items']['nodes'] if v['content']}
    expected_ids = {v['node_id'] for v in issues.values()}
    assert expected_ids.issubset(final_by_issue)
    for item in SPEC['issues']:
        row = final_by_issue[issues[item['key']]['node_id']]
        values = {v['field']['name']: v for v in row['fieldValues']['nodes'] if v and v.get('field')}
        for name in ('Status','Priority','Component','Phase','Precision gate','Depends on','Execution order'):
            assert name in values, ('Missing field value', item['key'], name)
    root_number = issues['roadmap']['number']
    receipt = comment_once(root_number, 'project-configured',
        f"执行Project已配置：{final['url']}\n\n13个Issue已加入，12个原生sub-issues、21条原生blocked-by依赖；"
        'Status／Priority／Component／Phase／Precision gate／Depends on／Execution order可筛选。'
        '\n\n已封存基线Issue关闭；下一项为Runtime cold/warm/reload事实Issue。'
        'Project本身不会启动compute或绕过精度审批。后续每次开始／结束必须回写Issue和Project。')
    save({'fully_configured_and_verified': True, 'repository_link_verified': True,
          'project_url': final['url'], 'project_public': final['public'], 'field_definitions': final['fields']['nodes'],
          'views': final['views']['nodes'], 'verified_issue_count': len(expected_ids),
          'root_project_comment': receipt['html_url']})
    print(json.dumps({'completed': True, 'project': final['url'], 'issues': len(expected_ids),
                      'views': final['views']['nodes']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)

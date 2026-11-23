#!/bin/bash
set -euo pipefail
D=/tmp/h3_verify186
rm -rf $D; mkdir -p $D/audit $D/out; chmod 700 $D
cp -a /tmp/h3_budget149/. $D/
# drop prior artefacts and the audit copy that would shadow ours via sys.path[0]
rm -rf $D/budget149a $D/budget149a.log $D/budget149a_process.json $D/run_budget149.py $D/launch_budget149.py
rm -f $D/matrix_safety126.py
cp /tmp/matrix_safety186.py $D/audit/matrix_safety126.py
cp /tmp/fast_audit186.py    $D/audit/fast_audit186.py
cp /tmp/run186.py           $D/run186.py
echo "shadow copy gone: $([ -f $D/matrix_safety126.py ] && echo NO || echo YES)"
echo "audit sha: $(sha256sum $D/audit/matrix_safety126.py | cut -c1-16)"

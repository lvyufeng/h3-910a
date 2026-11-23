#!/bin/bash
# R182: inject the hcc C++ include flags into the device compile through a PRIVATE overlay of
# the asc_op_compile_base python package (shared CANN untouched). The anchor is the end of
# gen_compile_cmd_v200/v220, immediately before `return compile_cmd`, so the flags land in the
# actual bisheng command line regardless of what the rest of the pipeline does.
set -euo pipefail
R=/home/mseco/Ascend/cann-9.2.0
H=$R/toolkit/toolchain/hcc/aarch64-target-linux-gnu
OVL=${OVL:-/tmp/h3_overlay182}
O=${O:-/tmp/h3_kern173/out}

source $R/set_env.sh >/dev/null 2>&1
export ASCEND_CANN_PACKAGE_PATH=$R

echo "=== refresh the private overlay"
rm -rf "$OVL"; mkdir -p "$OVL"; chmod 700 "$OVL"
cp -a "$R/python/site-packages/asc_op_compile_base" "$OVL/"

echo "=== patch every gen_compile_cmd_* to append the include flags"
python3 - "$OVL" "$H" <<'PY'
import sys, pathlib, re
ovl, H = sys.argv[1], sys.argv[2]
flags = ['-isystem', H + '/include',
         '-isystem', H + '/include/c++/14.3.0',
         '-isystem', H + '/include/c++/14.3.0/aarch64-target-linux-gnu']
lit = '[' + ', '.join(repr(f) for f in flags) + ']'
patched = 0
for t in pathlib.Path(ovl).rglob('ascendc_compile_v2*.py'):
    s = t.read_text()
    if 'H3_INCLUDE_INJECTED' in s:
        print('already patched:', t.name); continue
    # append right before the function's return so it is unconditional
    n = s.count('    return compile_cmd\n')
    if n == 0:
        print('NO_ANCHOR:', t.name); continue
    s = s.replace(
        '    return compile_cmd\n',
        '    # H3_INCLUDE_INJECTED: CANN ships these headers but does not pass them to bisheng\n'
        '    compile_cmd += ' + lit + '\n'
        '    return compile_cmd\n')
    t.write_text(s)
    patched += 1
    print('patched', t.name, 'anchors', n)
assert patched > 0, 'nothing patched'
PY

echo "=== rebuild with the overlay first on PYTHONPATH"
cd "$O"
export PYTHONPATH="$OVL:$R/python/site-packages:$PYTHONPATH"
echo "PYTHONPATH head: ${PYTHONPATH%%:*}"
bash build.sh > /tmp/b_overlay182.log 2>&1; echo "build exit=$?"
tail -8 /tmp/b_overlay182.log
echo "=== cstdint occurrences in build log"
grep -c cstdint /tmp/b_overlay182.log || true
echo "=== artifacts"
find build_out -maxdepth 2 -name 'custom_opp*' 2>/dev/null | head -3
find build_out -name '*.run' 2>/dev/null | head -3

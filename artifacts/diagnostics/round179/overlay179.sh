#!/bin/bash
# R179: inject the hcc C++ include flags into the device compile via a PRIVATE overlay of the
# asc_op_compile_base python package. The shared CANN installation is never modified: the
# package is copied to $OVL and put first on PYTHONPATH, so Python imports the patched copy.
#
# Why this layer: R176/R177 showed the flags do not come from the generated dynamic python
# file (there are two of them and the one that matters is under ascendc_kernels/binary/),
# and do not survive in build_out/autogen/custom_compile_options.ini because the build
# rewrites that file. asc_op_compile_base is imported as a normal package by asc_opc, so a
# PYTHONPATH overlay is the one injection point that is both effective and non-invasive.
set -euo pipefail
R=/home/mseco/Ascend/cann-9.2.0
H=$R/toolkit/toolchain/hcc/aarch64-target-linux-gnu
OVL=${OVL:-/tmp/h3_overlay179}
O=${O:-/tmp/h3_kern173/out}

source $R/set_env.sh >/dev/null 2>&1
export ASCEND_CANN_PACKAGE_PATH=$R

echo "=== copy the package to a private overlay"
rm -rf "$OVL"; mkdir -p "$OVL"; chmod 700 "$OVL"
cp -a "$R/python/site-packages/asc_op_compile_base" "$OVL/"

echo "=== patch the place that assembles the bisheng command"
python3 - "$OVL" "$H" <<'PY'
import sys, pathlib
ovl, H = sys.argv[1], sys.argv[2]
flags = ['-isystem', H + '/include',
         '-isystem', H + '/include/c++/14.3.0',
         '-isystem', H + '/include/c++/14.3.0/aarch64-target-linux-gnu']
targets = list(pathlib.Path(ovl).rglob('ascendc_compile_v200.py'))
assert targets, 'ascendc_compile_v200.py not found in the overlay'
for t in targets:
    s = t.read_text()
    if 'H3_HCC_INCLUDE_INJECTED' in s:
        print('already patched:', t); continue
    anchor = 'compile_options = '
    # append the flags unconditionally at the top of the module, then extend the option list
    marker = '# H3_HCC_INCLUDE_INJECTED\n'
    inject = marker + 'H3_EXTRA = ' + repr(flags) + '\n'
    s = inject + s
    # find a line that builds compile_options and extend it right after
    import re
    m = re.search(r'^(\s*)compile_options = (.*)$', s, re.M)
    assert m, 'no compile_options assignment in ' + str(t)
    indent = m.group(1)
    s = s[:m.end()] + '\n' + indent + 'compile_options = list(compile_options) + H3_EXTRA' + s[m.end():]
    t.write_text(s)
    print('patched:', t)
PY

echo "=== rebuild with the overlay first on PYTHONPATH"
cd "$O"
export PYTHONPATH="$OVL:$R/python/site-packages:$PYTHONPATH"
echo "PYTHONPATH head: ${PYTHONPATH%%:*}"
bash build.sh > /tmp/b_overlay179.log 2>&1; echo "build exit=$?"
tail -8 /tmp/b_overlay179.log
echo "=== cstdint occurrences"
grep -c cstdint /tmp/b_overlay179.log || true
echo "=== artifacts"
find build_out -maxdepth 2 -name 'custom_opp*' 2>/dev/null | head -3
find build_out -name '*.run' 2>/dev/null | head -3

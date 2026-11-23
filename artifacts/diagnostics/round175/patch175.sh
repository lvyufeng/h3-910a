#!/bin/bash
set -x
R=/home/mseco/Ascend/cann-9.2.0
H=$R/toolkit/toolchain/hcc/aarch64-target-linux-gnu
source $R/set_env.sh >/dev/null 2>&1
D=/tmp/h3_kern173/out/build_out/op_kernel/H3RangeStats_ascend910
PYF=$D/dynamic/H3RangeStats.py
grep -n "custom_compile_options = " "$PYF"

python3 - "$PYF" "$H" <<'PY'
import re, sys
path, H = sys.argv[1], sys.argv[2]
s = open(path).read()
flags = ['-isystem', H + '/include',
         '-isystem', H + '/include/c++/14.3.0',
         '-isystem', H + '/include/c++/14.3.0/aarch64-target-linux-gnu']
lit = '[' + ', '.join(repr(f) for f in flags) + ']'
new = 'custom_compile_options = {"__ALLSOC__": ' + lit + '},'
s2 = re.sub(r'custom_compile_options = \{\},', new, s, count=1)
assert s2 != s, 'patch did not apply'
open(path, 'w').write(s2)
print('patched OK')
PY

grep -c "__ALLSOC__" "$PYF"
PARAM=$D/bin_param/H3RangeStats_c99b8ccfd021b79ccd971b98c1146467_param.json
OUT=$D/kernel_0
mkdir -p "$OUT"
asc_opc /tmp/h3_kern173/out/op_kernel/h3_range_stats.cpp \
  --main_func=h3_range_stats --input_param="$PARAM" \
  --soc_version=Ascend910A --output="$OUT" \
  --impl_mode=high_performance,optional --op_mode=dynamic --simplified_key_mode=0 2>&1 | tail -15
echo "===ARTIFACTS"
ls "$OUT" 2>/dev/null | head
echo "===PATCH_SURVIVED"
grep -c "__ALLSOC__" "$PYF" || true

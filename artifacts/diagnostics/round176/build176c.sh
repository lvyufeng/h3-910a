#!/bin/bash
R=/home/mseco/Ascend/cann-9.2.0
H=$R/toolkit/toolchain/hcc/aarch64-target-linux-gnu
source $R/set_env.sh >/dev/null 2>&1
export ASCEND_CANN_PACKAGE_PATH=$R
O=/tmp/h3_kern173/out
PYF=$O/build_out/op_kernel/H3RangeStats_ascend910b/dynamic/H3RangeStats.py
cd $O

patch_py() {
  [ -f "$PYF" ] || { echo "NO_PY"; return 1; }
  python3 - "$PYF" "$H" <<'PY'
import sys
path, H = sys.argv[1], sys.argv[2]
s = open(path).read()
needle = 'custom_compile_options = {},'
if needle not in s:
    print('NEEDLE_NOT_FOUND'); raise SystemExit(3)
flags = ['-isystem', H + '/include',
         '-isystem', H + '/include/c++/14.3.0',
         '-isystem', H + '/include/c++/14.3.0/aarch64-target-linux-gnu']
lit = '[' + ', '.join(repr(f) for f in flags) + ']'
s = s.replace(needle, 'custom_compile_options = {"__ALLSOC__": ' + lit + '},', 1)
open(path, 'w').write(s)
print('PATCHED_EXACTLY_ONE')
PY
}

echo "===pass 1"
bash build.sh > /tmp/b1.log 2>&1; echo "pass1 exit=$?"
patch_py
echo "===verify"
grep -n 'custom_compile_options = ' "$PYF" | head -3
echo "===pass 2"
bash build.sh > /tmp/b2.log 2>&1; echo "pass2 exit=$?"
tail -5 /tmp/b2.log
echo "===fatal error count in pass2"
grep -cE "fatal error|cstdint" /tmp/b2.log
echo "===artifacts"
find $O/build_out -maxdepth 2 -name "custom_opp*" 2>/dev/null | head -5
find $O/build_out -name "*.run" 2>/dev/null | head -3

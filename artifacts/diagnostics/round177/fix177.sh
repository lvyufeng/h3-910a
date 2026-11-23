#!/bin/bash
R=/home/mseco/Ascend/cann-9.2.0
H=$R/toolkit/toolchain/hcc/aarch64-target-linux-gnu
source $R/set_env.sh >/dev/null 2>&1
export ASCEND_CANN_PACKAGE_PATH=$R
O=/tmp/h3_kern173/out
cd $O

echo "===opdesc_parser ini format (385-430)"
sed -n '385,430p' $R/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/opdesc_parser.py

# the file actually used by the kernel compile
REAL=build_out/op_kernel/ascendc_kernels/binary/dynamic/h3_range_stats.py
echo "===patch the REAL dynamic py"
if [ -f "$REAL" ]; then
  python3 - "$REAL" "$H" <<'PY'
import sys
path, H = sys.argv[1], sys.argv[2]
s = open(path).read()
needle = 'custom_compile_options = {},'
flags = ['-isystem', H + '/include',
         '-isystem', H + '/include/c++/14.3.0',
         '-isystem', H + '/include/c++/14.3.0/aarch64-target-linux-gnu']
lit = '[' + ', '.join(repr(f) for f in flags) + ']'
if needle in s:
    s = s.replace(needle, 'custom_compile_options = {"__ALLSOC__": ' + lit + '},', 1)
    open(path, 'w').write(s)
    print('PATCHED_REAL_PY')
else:
    print('NEEDLE_ABSENT_IN_REAL_PY')
PY
else
  echo "REAL_PY_NOT_PRESENT"
fi
grep -n 'custom_compile_options = ' "$REAL" 2>/dev/null | head -2

echo "===pass 2 build"
bash build.sh > /tmp/b3.log 2>&1; echo "exit=$?"
tail -6 /tmp/b3.log
echo "===cstdint count"
grep -cE "cstdint" /tmp/b3.log
echo "===artifacts"
find build_out -maxdepth 2 -name "custom_opp*" 2>/dev/null | head -3
find build_out -name "*.run" 2>/dev/null | head -3

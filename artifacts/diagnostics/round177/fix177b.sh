#!/bin/bash
R=/home/mseco/Ascend/cann-9.2.0
H=$R/toolkit/toolchain/hcc/aarch64-target-linux-gnu
source $R/set_env.sh >/dev/null 2>&1
export ASCEND_CANN_PACKAGE_PATH=$R
O=/tmp/h3_kern173/out
cd $O
INI=build_out/autogen/custom_compile_options.ini
echo "===ensure autogen exists (one pass may be needed)"
[ -d build_out/autogen ] || { bash build.sh > /tmp/b4a.log 2>&1; echo "warmup pass exit=$?"; }
ls -l $INI 2>/dev/null || echo "INI_MISSING"

echo "===write the official ini (op_type,soc_ver,opts;)"
python3 - "$INI" "$H" <<'PY'
import sys
path, H = sys.argv[1], sys.argv[2]
opts = [
    '-isystem', H + '/include',
    '-isystem', H + '/include/c++/14.3.0',
    '-isystem', H + '/include/c++/14.3.0/aarch64-target-linux-gnu',
]
line = 'ALL,,' + ';'.join(opts) + '\n'
open(path, 'w').write(line)
print('WROTE:', line.strip()[:140])
PY
cat $INI | head -3

echo "===build with ini"
bash build.sh > /tmp/b4.log 2>&1; echo "exit=$?"
tail -6 /tmp/b4.log
echo "===cstdint count"
grep -c cstdint /tmp/b4.log
echo "===ini survived"
head -1 $INI
echo "===artifacts"
find build_out -maxdepth 2 -name "custom_opp*" 2>/dev/null | head -3
find build_out -name "*.run" 2>/dev/null | head -3

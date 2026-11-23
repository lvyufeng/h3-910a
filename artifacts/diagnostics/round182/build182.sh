#!/bin/bash
set -euo pipefail
R=/home/mseco/Ascend/cann-9.2.0
H=$R/toolkit/toolchain/hcc/aarch64-target-linux-gnu
OVL=/tmp/h3_overlay182
BIN=/tmp/h3_bin182
O=/tmp/h3_kern173/out
source $R/set_env.sh >/dev/null 2>&1
export ASCEND_CANN_PACKAGE_PATH=$R

echo "=== install the private asc_opc wrapper ahead of CANN in PATH"
mkdir -p $BIN
cp /tmp/asc_opc_overlay182 $BIN/asc_opc
chmod +x $BIN/asc_opc
export PATH="$BIN:$PATH"
echo "asc_opc resolves to: $(command -v asc_opc)"
head -3 "$(command -v asc_opc)" | tail -1

echo "=== verify the overlay is the imported package"
PYTHONPATH="$OVL:$R/python/site-packages" ${HI_PYTHON:-python3} -c "
import asc_op_compile_base, pathlib
p = pathlib.Path(asc_op_compile_base.__file__).resolve()
print('imported from:', p)
src = (p.parent / 'asc_op_compiler' / 'ascendc_compile_v200.py').read_text()
print('patch present:', 'H3_INCLUDE_INJECTED' in src)
"

echo "=== rebuild"
cd $O
bash build.sh > /tmp/b182.log 2>&1; echo "build exit=$?"
tail -6 /tmp/b182.log
echo "=== cstdint count"
grep -c cstdint /tmp/b182.log || true
echo "=== artifacts"
find build_out -maxdepth 2 -name 'custom_opp*' 2>/dev/null | head -3
find build_out -name '*.run' 2>/dev/null | head -3
find build_out -path '*h3_range_stats*' -name '*.o' 2>/dev/null | head -3

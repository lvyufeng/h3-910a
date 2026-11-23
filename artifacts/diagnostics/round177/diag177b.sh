#!/bin/bash
R=/home/mseco/Ascend/cann-9.2.0
M=$R/compiler/tikcpp/ascendc_kernel_cmake
echo "===WHERE_COMPILE_OPTIONS_WRITTEN"
grep -rn 'compile_options' $M/fwk_modules/ 2>/dev/null | head -20
echo "===GENERATED_FILES_WITH_COMPILE_OPTIONS"
grep -rln 'compile_options' /tmp/h3_kern173/out/build_out/ 2>/dev/null | head -8
echo "===ONE_INI_HEAD"
F=$(grep -rln 'compile_options' /tmp/h3_kern173/out/build_out/ 2>/dev/null | head -1)
echo "FILE=$F"
head -12 "$F" 2>/dev/null

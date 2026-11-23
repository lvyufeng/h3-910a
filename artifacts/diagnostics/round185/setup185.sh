#!/bin/bash
# R185: measure the end-to-end step time of the stage A candidate.
# Two identical 8-rank runs differ only by H3_FAST_AUDIT, and each run imports a different
# matrix_safety126: the native run gets the untouched original, the fast run gets the patched
# copy. The goal is a measurement, not adoption; the repo default stays the native path.
set -euo pipefail
R=/home/mseco/Ascend/cann-9.2.0
D=/tmp/h3_fast185
rm -rf $D; mkdir -p $D/native $D/fast $D/out; chmod 700 $D
# module set: reuse the R149 experiment tree, it already holds the benchmark and its deps
cp -a /tmp/h3_budget149/. $D/
rm -rf $D/budget149a $D/budget149a.log $D/budget149a_process.json $D/run_budget149.py $D/launch_budget149.py
# native: the untouched audit
cp /tmp/h3_budget149/matrix_safety126.py $D/native/matrix_safety126.py
# fast: the patched audit plus its helper
cp /tmp/matrix_safety185.py  $D/fast/matrix_safety126.py
cp /tmp/fast_audit185.py     $D/fast/fast_audit185.py
[ -f /tmp/h3_lib184/libh3fast184.so ] || { echo "missing libh3fast184.so"; exit 1; }
echo "native audit sha: $(sha256sum $D/native/matrix_safety126.py | cut -c1-16)"
echo "fast   audit sha: $(sha256sum $D/fast/matrix_safety126.py | cut -c1-16)"

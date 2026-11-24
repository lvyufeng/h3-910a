#!/usr/bin/env python3
"""Prepare private bit-copy FA wrapper/probe from R217 without mixing candidates."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
R217=HERE.parent/'round217'
wrapper=(R217/'fa217.py').read_text()
# Probe imports retain the fa217 name, but all caches/packages are fresh R218 paths.
(HERE/'fa218.py').write_text(wrapper)
probe=(R217/'probe217.py').read_text()
probe=probe.replace("Path('/tmp/h3_fa217')","Path('/tmp/h3_broadcast218')")
probe=probe.replace('import fa217 as fa','import fa218 as fa')
probe=probe.replace('R217_RESULT','R218_RESULT')
(HERE/'fa_probe218.py').write_text(probe)
for name in ['fa218.py','fa_probe218.py','broadcast_unit218.py']:
 ast.parse((HERE/name).read_text(),filename=name)
print('private FA wrapper/probe ASTs valid')

"""Guard the compact dynamic tile loop without changing FP32 K accumulation."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class DynamicStepContracts(unittest.TestCase):
    def test_only_tile_iteration_is_dynamic_not_K_init_dtype(self):
        text=(ROOT/'blocked_dynamic_steps83.py').read_text();ast.parse(text)
        self.assertIn('with t.for_range(0, steps) as step:',text)
        self.assertIn('for kt in range(k_tiles):',text)
        self.assertIn('init_l1out=(kt == 0)',text)
        self.assertNotIn('for step in range(steps):',text)
        self.assertNotIn('t.h_cast(',text)
        self.assertIn('f"blocked_dynamic83_',text)
    def test_dynamic_probe_has_new_source_and_writeonce_root(self):
        text=(ROOT/'probe_blocked_dynamic83.py').read_text()
        self.assertIn('import blocked_dynamic_steps83 as kernel',text)
        self.assertIn('/tmp/h3_round83_blocked_dynamic01',text)
        self.assertIn('CPU_reference_operator',text)
        self.assertIn("'resident_end_to_end_with_A_pack_and_output_unpack_s'",text)


if __name__=='__main__':unittest.main()

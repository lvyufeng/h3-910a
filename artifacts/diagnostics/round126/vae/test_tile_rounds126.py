#!/usr/bin/env python3
"""Portable regressions for unequal native tile counts; no Torch or model imports."""
import unittest
from tile_rounds126 import aligned_local_tile_rounds

class TileRoundsTests(unittest.TestCase):
    def test_real_eight_rank_28_21_split(self):
        counts=[28]*4+[21]*4
        sequences=[aligned_local_tile_rounds(n,max(counts)) for n in counts]
        self.assertTrue(all(len(s)==28 for s in sequences))
        self.assertEqual([sum(local for _,local in s) for s in sequences],counts)
        self.assertTrue(all([i for i,_ in s]==list(range(28)) for s in sequences))
        self.assertTrue(all(not local for _,local in sequences[7][21:]))
    def test_empty_rank_still_joins_peer_gates(self):
        self.assertEqual(aligned_local_tile_rounds(0,2),[(0,False),(1,False)])
    def test_uniform_counts(self):
        self.assertEqual(aligned_local_tile_rounds(2,2),[(0,True),(1,True)])
    def test_inverted_or_oversized_counts_rejected(self):
        for args in ((2,1),(-1,2),(0,129)):
            with self.assertRaises(ValueError):aligned_local_tile_rounds(*args)
    def test_boolean_count_rejected(self):
        with self.assertRaises(TypeError):aligned_local_tile_rounds(True,2)

if __name__=='__main__':unittest.main()

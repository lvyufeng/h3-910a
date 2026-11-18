"""Timeout monitor must distinguish an executable worker from its observer shell."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class TimeoutMonitorContracts(unittest.TestCase):
    def test_first_monitor_false_positive_is_preserved_and_identifiable(self):
        r=json.loads((ROOT/'static_timeout_state83.json').read_text())
        self.assertEqual(len(r['workers']),1)
        self.assertTrue(r['workers'][0]['command'].startswith('bash -c /home/mseco/vllm910a/bin/python -'))
        self.assertIn('Reconcile the timed-out static tile-unroll attempt',r['workers'][0]['command'])
    def test_exact_worker_classifier_does_not_match_the_observer(self):
        observer=json.loads((ROOT/'static_timeout_state83.json').read_text())['workers'][0]['command']
        prototype='/home/mseco/vllm910a/bin/python -u /tmp/h3_round83_blocked02/probe_blocked83.py --case up'
        def match(cmd):return cmd.startswith('/home/mseco/vllm910a/bin/python')and '/tmp/h3_round83_blocked02/probe_blocked83.py'in cmd
        self.assertFalse(match(observer));self.assertTrue(match(prototype))
    def test_timeout_is_not_falsely_reported_as_precision_or_speed_failure(self):
        r=json.loads((ROOT/'static_timeout_diagnosis83.json').read_text())
        self.assertEqual(r['exit_code'],124);self.assertFalse(r['kernel_math_changed_for_retry'])
        self.assertTrue(r['does_not_claim_timedout_shape_as_measured']);self.assertTrue(r['same_root_replay_forbidden'])


if __name__=='__main__':unittest.main()

"""Atomic evidence ownership tests using only owned temporary directories."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evidence_io63 import EvidenceWriter


class EvidenceIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h3_evidence63_test_', dir=Path(tempfile.gettempdir()).resolve(strict=True))
        self.root = Path(self.tmp.name).resolve(strict=True)
        self.assertEqual(self.root, Path(self.tmp.name).absolute())
        self.assertTrue(self.root.name.startswith('h3_evidence63_test_'))

    def tearDown(self):
        # Verify our exact owned target before TemporaryDirectory removes it.
        self.assertEqual(self.root.resolve(strict=True), self.root)
        self.assertTrue(self.root.name.startswith('h3_evidence63_test_'))
        self.tmp.cleanup()

    def test_publishes_complete_revisions_and_refuses_reuse(self):
        path = self.root / 'result.json'
        writer = EvidenceWriter(path)
        writer.write({'revision': 1, 'completed': False})
        self.assertEqual(json.loads(path.read_text())['revision'], 1)
        writer.write({'revision': 2, 'completed': True})
        self.assertEqual(json.loads(path.read_text()), {'revision': 2, 'completed': True})
        with self.assertRaises(FileExistsError):
            EvidenceWriter(path)
        self.assertTrue(writer.claim.exists())

    def test_duplicate_start_cannot_claim_before_first_snapshot(self):
        path = self.root / 'result.json'
        first = EvidenceWriter(path)
        with self.assertRaises(FileExistsError):
            EvidenceWriter(path)
        first.write({'owner': 'first'})
        self.assertEqual(json.loads(path.read_text()), {'owner': 'first'})

    def test_external_mutation_is_not_silently_overwritten(self):
        path = self.root / 'result.json'
        writer = EvidenceWriter(path)
        writer.write({'value': 1})
        path.write_text('{"external": true}\n')
        with self.assertRaises(RuntimeError):
            writer.write({'value': 2})
        self.assertEqual(json.loads(path.read_text()), {'external': True})

    def test_failed_serialization_keeps_prior_json(self):
        path = self.root / 'result.json'
        writer = EvidenceWriter(path)
        writer.write({'value': 1})
        with self.assertRaises(TypeError):
            writer.write({'bad': object()})
        self.assertEqual(json.loads(path.read_text()), {'value': 1})
        self.assertEqual(list(self.root.glob('.evidence63_*')), [])

    def test_failed_atomic_replace_keeps_prior_and_cleans_owned_temp(self):
        path = self.root / 'result.json'
        writer = EvidenceWriter(path)
        writer.write({'value': 1})
        with patch('evidence_io63.os.replace', side_effect=OSError('simulated publication failure')):
            with self.assertRaises(OSError):
                writer.write({'value': 2})
        self.assertEqual(json.loads(path.read_text()), {'value': 1})
        self.assertEqual(list(self.root.glob('.evidence63_*')), [])


if __name__ == '__main__':
    unittest.main()

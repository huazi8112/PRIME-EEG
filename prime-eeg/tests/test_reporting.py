import argparse
import json
import tempfile
import unittest
from pathlib import Path

from prime_eeg import io
from prime_eeg.cli import verify_results


class ReportingTests(unittest.TestCase):
    def test_missing_latency_serializes_as_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            io.write_json(path, {"latency_ci": [float("nan"), float("nan")]})
            self.assertEqual(json.loads(path.read_text())["latency_ci"], [None, None])

    def test_reference_ledger_arithmetic(self):
        references = Path(__file__).resolve().parents[1] / "reference"
        if not references.exists():
            self.skipTest("Reference ledgers are in the repository checkout, not the wheel")
        verify_results(argparse.Namespace(reference=str(references), recomputed=None))


if __name__ == "__main__":
    unittest.main()

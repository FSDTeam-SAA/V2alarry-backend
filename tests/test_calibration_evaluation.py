import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class EvaluationBudgetTests(unittest.TestCase):
    def test_dry_run_never_makes_provider_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            result = subprocess.run([sys.executable, "scripts/evaluate_coaching_calibration.py", "--output", str(output)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(output.read_text())["calls"], [])

    def test_exhausted_budget_exits_unsuccessfully_without_a_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            result = subprocess.run([sys.executable, "scripts/evaluate_coaching_calibration.py", "--live",
                                     "--scenario", "trusted_capacity", "--max-requests", "0", "--max-total-tokens", "0",
                                     "--output", str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["calls"], [])


if __name__ == "__main__":
    unittest.main()

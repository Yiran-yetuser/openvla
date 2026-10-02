"""CPU checks using the immutable real reference, no GPU or training."""
import copy
import json
import unittest
from reproduction.compare_small_fit_lr import ROOT, RESULT, validate_report, require_finite


class ComparisonEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(RESULT.read_text())
        self.rows = json.loads((ROOT / "reproduction/results/training_chain_cpu_audit_v1.json").read_text())["rows"]

    def test_real_reference_summaries(self):
        validate_report(self.report, self.rows)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            require_finite({"nested": [float("nan")]})

    def test_wrong_summary_rejected(self):
        changed = copy.deepcopy(self.report)
        changed["evaluations"][0]["summary"]["fit"]["teacher_token_accuracy"] = 1.0
        with self.assertRaises(AssertionError):
            validate_report(changed, self.rows)

    def test_frame_identity_mismatch_rejected(self):
        changed = copy.deepcopy(self.report)
        changed["evaluations"][0]["rows"][0]["image_sha256"] = "wrong"
        with self.assertRaises(AssertionError):
            validate_report(changed, self.rows)

    def test_production_stage_or_loader_mismatch_rejected(self):
        for key, value in (("step", 25), ("loader", "training_BF16_autocast")):
            changed = copy.deepcopy(self.report)
            changed["production_evaluation"][key] = value
            with self.assertRaises(AssertionError):
                validate_report(changed, self.rows)


if __name__ == "__main__":
    unittest.main()

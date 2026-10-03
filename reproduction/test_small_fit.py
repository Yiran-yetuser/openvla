"""CPU checks for bounded fitting and immutable resume evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from reproduction.fit_small_sample import (atomic_json, plan_spec, completed_snapshot, verify_snapshot,
                                          matched_control_spec, matched_initial_evaluation)
from reproduction.audit_training_runtime import file_hash


class SmallFitGuardsTest(unittest.TestCase):
    def test_update_bound(self):
        for updates in (0, 51):
            with self.assertRaises(ValueError):
                plan_spec(Path("fixture"), updates, 1e-4)

    def test_fixed_learning_rate(self):
        with self.assertRaises(ValueError):
            plan_spec(Path("fixture"), 50, 5e-4)

    def test_explicit_control_changes_only_learning_rate(self):
        reference = plan_spec(Path("fixture"), 50, 1e-4)
        control = plan_spec(Path("fixture"), 50, 5e-4, "small_fit_lr5e4_v1")
        matched_control_spec(control, reference)
        for updates, rate in ((25, 5e-4), (50, 1e-4)):
            with self.assertRaises(ValueError):
                plan_spec(Path("fixture"), updates, rate, "small_fit_lr5e4_v1")

    def test_control_rejects_other_configuration_changes(self):
        reference = plan_spec(Path("fixture"), 50, 1e-4)
        control = plan_spec(Path("fixture"), 50, 5e-4, "small_fit_lr5e4_v1")
        with self.assertRaises(ValueError):
            matched_control_spec({**control, "augmentation": True}, reference)
        with self.assertRaises(ValueError):
            matched_control_spec({**control, "source_checkpoint": "another"}, reference)

    def test_control_step_zero_must_match(self):
        reference = {"evaluations": [{"step": 0, "rows": [{"action": [0]}]}]}
        matched_initial_evaluation(reference["evaluations"][0], reference)
        with self.assertRaises(ValueError):
            matched_initial_evaluation({"step": 0, "rows": [{"action": [1]}]}, reference)

    def test_atomic_progress(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            atomic_json(path, {"step": 1})
            atomic_json(path, {"step": 2})
            self.assertEqual(json.loads(path.read_text()), {"step": 2})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_partial_checkpoint_not_resumable(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "step_010.partial").mkdir()
            self.assertIsNone(completed_snapshot(run))

    def test_resume_hash_and_spec(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "step_010"
            path.mkdir()
            tensor = path / "fixture.bin"
            tensor.write_bytes(b"local fixture")
            atomic_json(path / "snapshot.json", {"step": 10, "spec": {"x": 1},
                                                 "sha256": {"fixture.bin": file_hash(tensor)}})
            self.assertEqual(verify_snapshot(path, {"x": 1})["step"], 10)
            with self.assertRaises(ValueError):
                verify_snapshot(path, {"x": 2})
            tensor.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_snapshot(path, {"x": 1})


if __name__ == "__main__":
    unittest.main()

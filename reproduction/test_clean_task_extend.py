"""CPU guards for the immutable 50-to-200 update extension contract."""
import copy
import json
import unittest
from unittest.mock import patch

from reproduction import extend_clean_task
from reproduction.train_clean_task import sample_schedule, update_limit
from reproduction.verify_clean_task_extend import (production_equality, verify_loss_rows,
                                                    verify_parent_predictions)


class CleanTaskExtensionTests(unittest.TestCase):
    def test_extension_spec_has_verified_parent_and_budget(self):
        audit = json.loads(extend_clean_task.core.AUDIT.read_text())
        spec = extend_clean_task.specification(audit)
        self.assertEqual(spec["experiment"], "clean_task_extend_v1")
        self.assertEqual(spec["optimizer_updates"], 200)
        self.assertEqual(spec["parent_updates"], 50)
        self.assertEqual(spec["new_updates"], 150)
        self.assertEqual(spec["milestones"], [50, 100, 150, 200])
        self.assertEqual(spec["parent_spec"]["optimizer_updates"], 50)
        self.assertEqual(spec["parent_snapshot_sha256"],
                         extend_clean_task.sha256(extend_clean_task.PARENT / "snapshot.json"))

    def test_extension_sampling_keeps_parent_prefix(self):
        self.assertEqual(sample_schedule(992, 200, maximum=200)[:800],
                         sample_schedule(992, 50, maximum=50))
        self.assertEqual(update_limit({"experiment": "clean_task_extend_v1", "optimizer_updates": 200}), 200)
        with self.assertRaises(ValueError):
            update_limit({"experiment": "clean_task_extend_v1", "optimizer_updates": 50})

    def test_extension_preserves_training_configuration(self):
        audit = json.loads(extend_clean_task.core.AUDIT.read_text())
        spec = extend_clean_task.specification(audit)
        for field in ("learning_rate", "rank", "lora_alpha", "seed", "microbatch", "accumulation",
                      "nf4", "compute_dtype", "double_quant", "augmentation", "center_crop",
                      "train_episodes", "validation_episodes", "audit_sha256"):
            self.assertEqual(spec[field], spec["parent_spec"][field])

    def test_parent_must_match_previously_verified_snapshot(self):
        spec = extend_clean_task.core.specification(json.loads(extend_clean_task.core.AUDIT.read_text()))
        evidence = json.loads(extend_clean_task.PARENT_VERIFICATION.read_text())
        record = copy.deepcopy(evidence["snapshots"]["50"])
        record["sha256"]["adapter_model.safetensors"] = "0" * 64
        with self.assertRaises(ValueError):
            extend_clean_task.verify_parent(record, spec, evidence, evidence["final_report_sha256"])

    def test_parent_prediction_change_rejected(self):
        parent = json.loads((extend_clean_task.PARENT / "evaluation.json").read_text())
        altered = copy.deepcopy(parent)
        altered["splits"]["validation"]["teacher_rows"][0]["predicted_tokens"][0] += 1
        with self.assertRaises(ValueError):
            verify_parent_predictions(altered, parent)

    def loss_rows(self):
        schedule = sample_schedule(992, 200, maximum=200)
        rows = [{"optimizer_step": step, "sample_indices": schedule[(step - 1) * 16:step * 16],
                 "microbatch_losses": [0.5] * 8, "mean_loss": 0.5} for step in range(51, 201)]
        return rows, schedule

    def test_recomputed_loss_mean_required(self):
        rows, schedule = self.loss_rows()
        verify_loss_rows(rows, schedule)
        rows[0]["mean_loss"] = 0.1
        with self.assertRaises(ValueError):
            verify_loss_rows(rows, schedule)

    def test_uncheckpointed_duplicate_updates_not_silently_accepted(self):
        rows, schedule = self.loss_rows()
        with self.assertRaises(ValueError):
            verify_loss_rows(rows[:1] + rows, schedule)

    def test_loss_sampling_and_finiteness_required(self):
        rows, schedule = self.loss_rows()
        rows[0]["sample_indices"][0] = 992
        with self.assertRaises(ValueError):
            verify_loss_rows(rows, schedule)
        rows, schedule = self.loss_rows()
        rows[-1]["microbatch_losses"][0] = float("nan")
        with self.assertRaises(ValueError):
            verify_loss_rows(rows, schedule)

    def test_production_loader_difference_retained(self):
        end = {"step": 200, "splits": {s: {"autoregressive_rows": [{"action": [0.] * 7}]} for s in ("train", "validation")}}
        production = copy.deepcopy(end)
        production["loader"] = "production_no_autocast"
        self.assertTrue(production_equality([end], production))
        production["splits"]["train"]["autoregressive_rows"][0]["action"][0] = 1.
        self.assertFalse(production_equality([end], production))
        production["loader"] = "training_BF16_autocast"
        with self.assertRaises(ValueError):
            production_equality([end], production)

    def test_gpu_preflight_refuses_fast3r_even_when_not_using_cuda(self):
        with patch("reproduction.run_clean_task_closed_loop.subprocess.check_output",
                   return_value="/home/yyz/miniconda3/envs/fast3r/bin/python -u scripts/prepare_co3d51_source_v1.py\n") as mock:
            with self.assertRaises(RuntimeError):
                extend_clean_task.gpu_preflight()
            self.assertEqual(mock.call_count, 1)

    def test_verifier_refuses_missing_result(self):
        from reproduction.verify_clean_task_extend import verify
        if extend_clean_task.RESULT.exists():
            self.skipTest("A completed extension result is present")
        with self.assertRaises(FileNotFoundError):
            verify()


if __name__ == "__main__":
    unittest.main()

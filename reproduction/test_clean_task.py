"""CPU-only guard checks; these are not evidence that GPU training succeeded."""
import copy
import json
import unittest
from reproduction.train_clean_task import AUDIT, specification, sample_schedule


class CleanTaskTests(unittest.TestCase):
    def audit(self):
        return json.loads(AUDIT.read_text())

    def test_fresh_bounded_spec(self):
        spec = specification(self.audit())
        self.assertEqual(spec["optimizer_updates"], 50)
        self.assertEqual(spec["microbatch"] * spec["accumulation"], 16)
        self.assertIn("fresh LoRA", spec["initialization"])
        self.assertEqual(spec["validation_frames"], 246)

    def test_overlap_rejected(self):
        audit = self.audit()
        audit["selection"]["validation"][0] = copy.deepcopy(audit["selection"]["train"][0])
        with self.assertRaises(ValueError):
            specification(audit)

    def test_failed_audit_rejected(self):
        audit = self.audit()
        audit["checks"]["train_only_statistics"] = False
        with self.assertRaises(ValueError):
            specification(audit)

    def test_schedule_bounded_train_only(self):
        schedule = sample_schedule(992)
        self.assertEqual(len(schedule), 800)
        self.assertEqual(len(set(schedule)), 800)
        self.assertEqual(schedule, sample_schedule(992))
        self.assertTrue(all(0 <= i < 992 for i in schedule))
        with self.assertRaises(ValueError):
            sample_schedule(992, 51)


if __name__ == "__main__":
    unittest.main()

"""CPU checks for the 10 tasks x 50 rollouts acceptance gate."""
import unittest

from reproduction.libero_eval_results import parse_eval_text


def fixture(counts):
    lines = ["Task suite: libero_spatial"]
    total = successes = 0
    for task_id, count in enumerate(counts):
        task_successes = 0
        for trial in range(1, count + 1):
            success = trial == 1
            task_successes += success
            successes += success
            total += 1
            lines += [f"Task: task {task_id}", f"Starting episode {trial}...",
                      f"Success: {success}", f"# episodes completed so far: {total}",
                      f"# successes: {successes} ({successes / total * 100:.1f}%)"]
        if count == 50:
            lines += [f"Current task success rate: {task_successes / count}",
                      f"Current total success rate: {successes / total}"]
    return "\n".join(lines)


class EvalLogResultsTest(unittest.TestCase):
    def test_exact_full_suite(self):
        r = parse_eval_text(fixture([50] * 10))
        self.assertEqual(r["status"], "complete")
        self.assertEqual((r["episodes"], r["completed_task_summaries"]), (500, 10))
        self.assertEqual(r["final_success_rate"], 10 / 500)

    def test_partial_log_has_no_final_rate(self):
        r = parse_eval_text(fixture([50] * 7 + [32]))
        self.assertEqual((r["episodes"], r["completed_task_summaries"]), (382, 7))
        self.assertIsNone(r["final_success_rate"])

    def test_unequal_task_coverage_rejected(self):
        with self.assertRaises(ValueError):
            parse_eval_text(fixture([100] * 5))

    def test_duplicate_trial_rejected(self):
        with self.assertRaises(ValueError):
            parse_eval_text(fixture([50]).replace("Starting episode 2...", "Starting episode 1..."))

    def test_counter_jump_rejected(self):
        with self.assertRaises(ValueError):
            parse_eval_text(fixture([50]).replace("so far: 2\n", "so far: 500\n"))

    def test_success_counter_tampering_rejected(self):
        with self.assertRaises(ValueError):
            parse_eval_text(fixture([50]).replace("# successes: 1 (100.0%)", "# successes: 2 (100.0%)"))

    def test_summary_disagreement_rejected(self):
        with self.assertRaises(ValueError):
            parse_eval_text(fixture([50]).replace("Current task success rate: 0.02", "Current task success rate: 0.5"))

    def test_missing_last_summary_is_incomplete(self):
        r = parse_eval_text(fixture([50] * 10).rsplit("Current task success rate:", 1)[0])
        self.assertEqual(r["episodes"], 500)
        self.assertEqual(r["status"], "incomplete")

    def test_runtime_exception_blocks_final_result(self):
        r = parse_eval_text(fixture([50] * 10) + "\nCaught exception: simulator failure")
        self.assertEqual(r["status"], "incomplete")
        self.assertIsNone(r["final_success_rate"])


if __name__ == "__main__":
    unittest.main()

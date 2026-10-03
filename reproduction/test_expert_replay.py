"""CPU guards for expert action provenance and replay identity."""
import unittest

import numpy as np

from reproduction.audit_expert_correspondence import array_hash, keep_action_indices, selected_episodes


class ExpertReplayTests(unittest.TestCase):
    def test_official_noop_filter_keeps_gripper_changes(self):
        actions = np.array([[0, 0, 0, 0, 0, 0, -1],
                            [.2, 0, 0, 0, 0, 0, -1],
                            [0, 0, 0, 0, 0, 0, -1],
                            [0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1]])
        self.assertEqual(keep_action_indices(actions), [1, 3])

    def test_nonfinite_actions_fail(self):
        actions = np.zeros((1, 7))
        actions[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            keep_action_indices(actions)

    def test_action_hash_preserves_full_order(self):
        actions = np.arange(14).reshape(2, 7)
        self.assertEqual(array_hash(actions), array_hash(actions.astype('<f4')))
        self.assertNotEqual(array_hash(actions), array_hash(actions[::-1]))

    def test_partition_selection_ignores_metadata(self):
        selection = {"train": [{"episode_index": 1}], "validation": [{"episode_index": 2}],
                     "same_task_candidates": 46, "duplicate_policy": "content hash"}
        self.assertEqual(set(selected_episodes(selection)), {1, 2})

    def test_repeated_episode_rejected(self):
        with self.assertRaises(ValueError):
            selected_episodes({"train": [{"episode_index": 1}], "validation": [{"episode_index": 1}]})


if __name__ == "__main__":
    unittest.main()

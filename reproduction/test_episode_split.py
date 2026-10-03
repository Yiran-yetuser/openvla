"""Pure CPU safeguards for the clean episode-level experiment."""
import unittest
import numpy as np
from reproduction.audit_episode_split import TASK, action_statistics, select_episodes


class EpisodeSplitTests(unittest.TestCase):
    def examples(self):
        return [{"instruction": TASK, "eligible_frames": 10, "content_sha256": str(i),
                 "episode_index": i} for i in range(15)]

    def test_deterministic_disjoint(self):
        a, b = select_episodes(self.examples())
        self.assertEqual((a, b), select_episodes(self.examples()))
        self.assertEqual((len(a), len(b)), (8, 2))
        self.assertTrue({e["content_sha256"] for e in a}.isdisjoint(e["content_sha256"] for e in b))

    def test_duplicate_content_deduplicated(self):
        rows = self.examples()
        a, b = select_episodes(rows + rows)
        self.assertEqual(len({e["content_sha256"] for e in a + b}), 10)

    def test_insufficient_distinct_episodes_rejected(self):
        with self.assertRaises(ValueError):
            select_episodes(self.examples()[:9] * 2)

    def test_wrong_task_excluded(self):
        rows = self.examples()
        rows.append(dict(rows[0], instruction="different task", content_sha256="other"))
        self.assertEqual(select_episodes(self.examples()), select_episodes(rows))

    def test_stats_train_only_gripper_and_mask(self):
        train = np.arange(70).reshape(10, 7).astype(float)
        train[:, -1] = [-1, 1] * 5
        before = train.copy()
        stats = action_statistics(train)
        validation = np.full((2, 7), 1e6)
        self.assertEqual(stats, action_statistics(train))
        self.assertNotEqual(stats["q99"][:6], action_statistics(np.concatenate([train, validation]))["q99"][:6])
        np.testing.assert_array_equal(train, before)
        self.assertEqual(stats["mask"], [True] * 6 + [False])
        self.assertEqual(stats["mean"][-1], .5)

    def test_invalid_actions_rejected(self):
        for actions in ([], np.zeros((2, 7)), np.full((2, 7), np.nan)):
            with self.assertRaises(ValueError):
                action_statistics(actions)


if __name__ == "__main__":
    unittest.main()

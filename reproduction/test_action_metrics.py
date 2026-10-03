"""CPU regression tests; no pretrained weights, CUDA, or TensorFlow required."""
import unittest
import numpy as np
from reproduction.action_metrics import training_target, normalized_prediction, simulator_action, metrics


class ActionSpacesTest(unittest.TestCase):
    def setUp(self):
        self.stats = {"q01": [-1] * 7, "q99": [1] * 7, "mask": [True] * 6 + [False]}

    def test_gripper_round_trip(self):
        for raw_gripper, training_gripper in [(-1, 1), (1, 0)]:
            raw = np.array([0.] * 6 + [raw_gripper])
            target = training_target(raw, self.stats)
            self.assertEqual(target[-1], training_gripper)
            prediction = np.array([0.] * 6 + [target[-1]])
            np.testing.assert_array_equal(simulator_action(prediction), raw)
            report = metrics(prediction, raw, self.stats)
            self.assertTrue(report["gripper_correct"])
            self.assertEqual(report["simulator_mae_7d"], 0.)

    def test_old_metric_was_not_zero_for_perfect_policy(self):
        raw = np.array([0.] * 6 + [-1.])
        prediction = np.array([0.] * 6 + [1.])
        self.assertAlmostEqual(np.abs(prediction - raw).mean(), 2 / 7)
        self.assertEqual(metrics(prediction, raw, self.stats)["simulator_mae_7d"], 0.)

    def test_clipping_motion_not_gripper(self):
        target = training_target(np.array([3.] * 6 + [-1.]), self.stats)
        np.testing.assert_array_equal(target, [1.] * 7)

    def test_quantized_gripper_endpoint(self):
        for prediction, raw in [(.996, -1), (0., 1)]:
            report = metrics([0.] * 6 + [prediction], [0.] * 6 + [raw], self.stats)
            self.assertTrue(report["gripper_correct"])

    def test_unnormalization_inverse(self):
        stats = {"q01": [-2.] * 7, "q99": [4.] * 7, "mask": [True] * 6 + [False]}
        normalized = np.array([-.5, -.2, 0, .2, .5, .8, 1])
        physical = normalized.copy()
        physical[:6] = .5 * (normalized[:6] + 1) * 6 - 2
        np.testing.assert_allclose(normalized_prediction(physical, stats), normalized)


if __name__ == "__main__":
    unittest.main()

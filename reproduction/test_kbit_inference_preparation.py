"""Small interface contract for opt-in inference preparation."""
import unittest
from unittest.mock import patch

from experiments.robot.openvla_utils import prepare_vla_for_kbit_inference


class DummyModel:
    def __init__(self):
        self.evaluated = False

    def eval(self):
        self.evaluated = True
        return self


class InferencePreparationTests(unittest.TestCase):
    def test_opt_in_uses_peft_preparation_and_eval_mode(self):
        model = DummyModel()
        with patch("peft.prepare_model_for_kbit_training", return_value=model) as prepare:
            actual = prepare_vla_for_kbit_inference(model)
        prepare.assert_called_once_with(model, use_gradient_checkpointing=True)
        self.assertIs(actual, model)
        self.assertTrue(actual.evaluated)


if __name__ == "__main__":
    unittest.main()

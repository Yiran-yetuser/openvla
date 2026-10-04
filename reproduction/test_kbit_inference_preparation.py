"""Small interface contract for opt-in inference preparation."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from experiments.robot.openvla_utils import get_vla_action, prepare_vla_for_kbit_inference


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
        self.assertTrue(actual._openvla_kbit_inference_prepared)

    def test_prepared_action_generation_enters_bf16_autocast(self):
        model = MagicMock()
        model._openvla_kbit_inference_prepared = True
        processor = MagicMock()
        processor.return_value.to.return_value = {}
        context = MagicMock()
        with patch("experiments.robot.openvla_utils.torch.autocast", return_value=context) as autocast:
            get_vla_action(model, processor, "openvla-7b", {"full_image": np.zeros((2, 2, 3), dtype=np.uint8)}, "task", "stats")
        autocast.assert_called_once_with("cuda", dtype=torch.bfloat16)
        context.__enter__.assert_called_once()
        model.predict_action.assert_called_once_with(unnorm_key="stats", do_sample=False)

    def test_default_action_generation_does_not_enable_autocast(self):
        model = MagicMock()
        model._openvla_kbit_inference_prepared = False
        processor = MagicMock()
        processor.return_value.to.return_value = {}
        with patch("experiments.robot.openvla_utils.torch.autocast") as autocast:
            get_vla_action(model, processor, "openvla-7b", {"full_image": np.zeros((2, 2, 3), dtype=np.uint8)}, "task", "stats")
        autocast.assert_not_called()


if __name__ == "__main__":
    unittest.main()

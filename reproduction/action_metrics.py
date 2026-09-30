"""Action spaces used by LIBERO TFDS, OpenVLA training, and deployment.

Raw simulator gripper: -1=open, +1=close. Training: 1=open, 0=close.
Keep motion and gripper metrics separate: their units are not interchangeable.
"""
import numpy as np


def training_target(raw, stats):
    target = np.asarray(raw, dtype=np.float64).copy()
    target[-1] = 1 - np.clip(target[-1], 0, 1)
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", [True] * 6 + [False]), dtype=bool)
    target[mask] = np.clip(2 * (target[mask] - low[mask]) / (high[mask] - low[mask] + 1e-8) - 1, -1, 1)
    return target


def normalized_prediction(action, stats):
    action = np.asarray(action, dtype=np.float64).copy()
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", [True] * 6 + [False]), dtype=bool)
    action[mask] = 2 * (action[mask] - low[mask]) / (high[mask] - low[mask]) - 1
    return action


def simulator_action(prediction):
    action = np.asarray(prediction, dtype=np.float64).copy()
    action[-1] = -np.sign(2 * action[-1] - 1)
    return action


def metrics(prediction, raw, stats):
    deployed = simulator_action(prediction)
    raw = np.asarray(raw)
    norm_error = normalized_prediction(prediction, stats) - training_target(raw, stats)
    return {
        "motion_mae_6d": float(np.abs(deployed[:6] - raw[:6]).mean()),
        "simulator_mae_7d": float(np.abs(deployed - raw).mean()),
        "normalized_l1_7d": float(np.abs(norm_error).mean()),
        "gripper_correct": bool(deployed[-1] == raw[-1]),
        "translation_norm": float(np.linalg.norm(deployed[:3])),
        "normalized_abs_error": np.abs(norm_error).tolist(),
    }

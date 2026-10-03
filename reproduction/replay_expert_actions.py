"""Bounded expert-action replay: one train demo and one validation demo.

No OpenVLA model, training, or policy rollout is run. Use each original expert
demo's full state, ten settling steps, and the official filtered action sequence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.audit_episode_split import sha256
from reproduction.audit_expert_correspondence import array_hash, keep_action_indices
from reproduction.fetch_expert_demo import DESTINATION, DIGEST
from reproduction.run_clean_task_closed_loop import gpu_preflight

RUN = ROOT / "runs/expert_replay_v1"
OUTPUT = ROOT / "reproduction/results/expert_replay_v1.json"
ASSETS = ROOT / "reproduction/results/expert_replay_frames_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"expert_demos": 2, "training_episode": 185, "validation_episode": 394,
                          "model_loads": 0, "optimizer_updates": 0, "policy_rollouts": 0}, indent=2))
        return
    if OUTPUT.exists() or RUN.exists() or ASSETS.exists():
        raise FileExistsError("Existing run/evidence must not be repeated or overwritten")
    gpu_preflight()
    correspondence_path = ROOT / "reproduction/results/expert_correspondence_v1.json"
    correspondence = json.loads(correspondence_path.read_text())
    assert correspondence["status"] == "verified_unique_full_action_correspondence"
    assert sha256(DESTINATION) == DIGEST
    selected = [next(r for r in correspondence["matches"] if r["episode_index"] == index)
                for index in (185, 394)]
    assert [r["partition"] for r in selected] == ["train", "validation"]
    RUN.mkdir()
    ASSETS.mkdir()
    progress_path = RUN / "progress.json"

    def progress(status, **extra):
        tmp = RUN / "progress.tmp.json"
        tmp.write_text(json.dumps({"status": status, "worker_pid": os.getpid(), **extra}, indent=2))
        tmp.replace(progress_path)

    progress("initializing_expert_replay")
    # TensorFlow only resizes images; never give it CUDA devices/memory.
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from libero.libero import benchmark
    import robosuite
    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_dummy_action
    from reproduction.analyze_failure_videos import contact_sheet
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(6)
    assert task.language == correspondence["task_instruction"]
    results = []
    try:
        with h5py.File(DESTINATION, "r") as original:
            source_args = json.loads(original["data"].attrs["env_args"])["env_kwargs"]
            controller = robosuite.load_controller_config(default_controller="OSC_POSE")
            source_controller = source_args["controller_configs"]
            assert source_args["control_freq"] == 20 and source_args["robots"] == ["Panda"]
            assert all(controller[key] == value for key, value in source_controller.items())
            for match in selected:
                progress("replaying", episode_index=match["episode_index"], demo_name=match["demo_name"])
                demo = original["data"][match["demo_name"]]
                raw = demo["actions"][()]
                kept = keep_action_indices(raw)
                assert kept == match["kept_original_indices"]
                actions = raw[kept]
                assert array_hash(actions) == match["action_sha256_f32"]
                initial = np.asarray(demo["states"][0])
                assert initial.shape == (92,) and np.isfinite(initial).all()
                assert hashlib.sha256(initial.tobytes()).hexdigest() == match["initial_state_sha256"]
                # Verify the training/policy gripper conversion returns the raw
                # expert simulator convention. Execute RAW actions, not twice inverted ones.
                policy_gripper = 1 - np.clip(actions[:, -1], 0, 1)
                assert np.array_equal(-np.sign(2 * policy_gripper - 1), actions[:, -1])
                env, _ = get_libero_env(task, "openvla", resolution=256)
                try:
                    env.reset()
                    assert env.get_sim_state().shape == initial.shape
                    obs = env.set_init_state(initial)
                    for _ in range(10):
                        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                    trace, frames = [], []
                    for step, action in enumerate(actions):
                        image = get_libero_image(obs, 224)
                        frames.append(image)
                        bowl = env.sim.data.body_xpos[env.env.obj_body_id["akita_black_bowl_1"]].copy()
                        plate = env.sim.data.body_xpos[env.env.obj_body_id["plate_1"]].copy()
                        trace.append({"step": step, "original_action_index": kept[step],
                                      "simulator_action": action.tolist(), "eef_pos": obs["robot0_eef_pos"].tolist(),
                                      "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                                      "bowl_pos_m": bowl.tolist(), "plate_pos_m": plate.tolist(),
                                      "goal_before_action": bool(env.check_success()),
                                      "input_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
                        obs, reward, done, info = env.step(action.tolist())
                        trace[-1]["goal_after_action"] = bool(env.check_success())
                        trace[-1]["done_after_action"] = bool(done)
                    success = bool(env.check_success())
                    assert success == bool(done)
                    final_bowl = env.sim.data.body_xpos[env.env.obj_body_id["akita_black_bowl_1"]].copy()
                    final_plate = env.sim.data.body_xpos[env.env.obj_body_id["plate_1"]].copy()
                    final_asset = ASSETS / f"episode{match['episode_index']}_final_post_action.png"
                    from PIL import Image
                    Image.fromarray(get_libero_image(obs, 224)).save(final_asset)
                    keyframes = sorted(set(np.linspace(0, len(frames) - 1, 12, dtype=int).tolist()))
                    sheet = ASSETS / f"episode{match['episode_index']}_expert.png"
                    contact_sheet(frames, trace, keyframes,
                                  f"EXPERT | episode {match['episode_index']} | {match['demo_name']} | final success={success}").save(sheet)
                    trace_file = ROOT / f"reproduction/results/expert_replay_episode{match['episode_index']}_v1.json"
                    with trace_file.open("x") as handle:
                        json.dump({"episode_index": match["episode_index"], "demo_name": match["demo_name"],
                                   "initial_state_sha256": match["initial_state_sha256"],
                                   "actions": trace, "success": success}, handle, indent=2, allow_nan=False)
                    results.append({"partition": match["partition"], "episode_index": match["episode_index"],
                                    "demo_name": match["demo_name"], "initial_state_sha256": match["initial_state_sha256"],
                                    "action_count": len(actions), "action_sha256_f32": array_hash(actions),
                                    "success_at_end": success,
                                    "first_goal_after_action_step": next((r["step"] for r in trace if r["goal_after_action"]), None),
                                    "final_bowl_pos_m": final_bowl.tolist(), "final_plate_pos_m": final_plate.tolist(),
                                    "trace_file": str(trace_file.relative_to(ROOT)), "trace_sha256": sha256(trace_file),
                                    "contact_sheet": str(sheet.relative_to(ROOT)), "contact_sheet_sha256": sha256(sheet),
                                    "final_post_action_image": str(final_asset.relative_to(ROOT)), "final_image_sha256": sha256(final_asset)})
                    print(f"expert {match['demo_name']}: {len(actions)} actions; success={success}", flush=True)
                finally:
                    env.close()
        assert sha256(DESTINATION) == DIGEST
        report = {"schema_version": 1, "status": "completed_bounded_expert_replay",
                  "correspondence_sha256": sha256(correspondence_path), "hdf5_sha256": DIGEST,
                  "hdf5_hash_unchanged": True, "task_id": 6, "instruction": task.language,
                  "env_seed": 0, "controller": "OSC_POSE", "control_frequency_hz": 20,
                  "controller_matches_original": True, "settling_steps": 10,
                  "no_op_filter": "official threshold 1e-4 with gripper-change preservation",
                  "expert_replays": 2, "successes": sum(r["success_at_end"] for r in results),
                  "model_loads": 0, "optimizer_updates": 0, "policy_rollouts": 0, "results": results,
                  "limits": ["Two expert demonstrations, not a model success-rate or full-suite evaluation.",
                             "Expert demos use their own full original states, different from the prior two policy trials.",
                             "Exact selected-RLDS action provenance is established; regenerated image identity is not established.",
                             "A successful expert replay validates these two control trajectories, not every possible configuration."]}
        with OUTPUT.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        progress(report["status"], successes=report["successes"], expert_replays=2)
        print(json.dumps({"status": report["status"], "successes": report["successes"], "expert_replays": 2}, indent=2))
    except Exception:
        progress("failed_requires_diagnosis", error=traceback.format_exc(), completed_replays=len(results))
        raise


if __name__ == "__main__":
    main()

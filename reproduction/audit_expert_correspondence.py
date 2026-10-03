"""CPU-only matching of selected RLDS episodes to original HDF5 actions.

Full ordered float32 action sequences after the official no-op filter must match
uniquely. This is action provenance, not image identity or simulator replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.audit_episode_split import episode_fingerprint, sha256
from reproduction.fetch_expert_demo import DESTINATION, DIGEST, SIZE

OUTPUT = ROOT / "reproduction/results/expert_correspondence_v1.json"


def keep_action_indices(actions, threshold=1e-4):
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 7 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite original Nx7 actions")
    kept, previous = [], None
    for index, action in enumerate(values):
        noop = np.linalg.norm(action[:-1]) < threshold
        if previous is not None:
            noop = noop and action[-1] == previous[-1]
        if not noop:
            kept.append(index)
            previous = action
    return kept


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def selected_episodes(selection):
    result = {}
    # selection also contains count/policy metadata, not only episode lists.
    for split in ("train", "validation"):
        for episode in selection[split]:
            index = episode["episode_index"]
            if index in result:
                raise ValueError("Duplicate episode across partitions")
            result[index] = (split, episode)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Recompute existing evidence without writing")
    args = parser.parse_args()
    if args.execute and args.verify:
        parser.error("Choose execute or verify")
    if not args.execute and not args.verify:
        print(json.dumps({"selected_episodes": 10, "GPU_models": 0, "new_rollouts": 0,
                          "output": str(OUTPUT)}, indent=2))
        return
    if OUTPUT.exists() and not args.verify:
        raise FileExistsError("Inspect existing evidence; do not overwrite")
    if DESTINATION.stat().st_size != SIZE or sha256(DESTINATION) != DIGEST:
        raise ValueError("Original HDF5 source changed")
    audit_path = ROOT / "reproduction/results/clean_task_split_audit_v1.json"
    audit = json.loads(audit_path.read_text())
    manifest_path = ROOT / "reproduction/results/expert_demo_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sha256"] == DIGEST and manifest["bytes"] == SIZE
    originals = {}
    with h5py.File(DESTINATION, "r") as source:
        info = json.loads(source["data"].attrs["problem_info"])
        assert info["language_instruction"] == audit["task_instruction"]
        for name in sorted(source["data"], key=lambda k: int(k.split("_")[1])):
            demo = source["data"][name]
            raw = demo["actions"][()]
            initial = np.asarray(demo["states"][0])
            assert initial.shape == (92,) and np.isfinite(initial).all()
            indices = keep_action_indices(raw)
            filtered = np.asarray(raw[indices], dtype="<f4")
            originals[name] = {"values": filtered, "original_action_count": len(raw),
                               "filtered_action_count": len(filtered), "action_sha256_f32": array_hash(filtered),
                               "kept_original_indices": indices,
                               "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                               "initial_state_shape": list(initial.shape), "initial_state_dtype": str(initial.dtype)}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    builder = tfds.builder("libero_spatial_no_noops", data_dir=audit["dataset"]["path"])
    dataset = builder.as_dataset(split="train", shuffle_files=False, decoders={
        "steps": {"observation": {"image": tfds.decode.SkipDecoding(),
                                   "wrist_image": tfds.decode.SkipDecoding()}}})
    selected = selected_episodes(audit["selection"])
    matches = []
    for index, episode in enumerate(dataset):
        if index not in selected:
            continue
        split, row = selected[index]
        steps = list(episode["steps"].as_numpy_iterator())
        assert episode_fingerprint(steps) == row["content_sha256"]
        assert all(s["language_instruction"].decode() == audit["task_instruction"] for s in steps)
        actions = np.asarray([s["action"] for s in steps], dtype="<f4")
        exact = [name for name, value in originals.items() if np.array_equal(actions, value["values"])]
        if len(exact) != 1:
            raise ValueError(f"episode {index}: {len(exact)} exact full-action matches; no weakened correspondence")
        name = exact[0]
        record = {k: v for k, v in originals[name].items() if k != "values"}
        record.update(partition=split, episode_index=index, tfds_content_sha256=row["content_sha256"],
                      demo_name=name, tfds_action_count=len(actions), exact_unique_full_action_match=True)
        matches.append(record)
        print(f"episode {index} -> {name}; all {len(actions)} actions match", flush=True)
    assert {m["episode_index"] for m in matches} == set(selected)
    # Explicitly compare to the old policy trials without pretending equal states.
    comparison = json.loads((ROOT / "reproduction/results/clean_task_closed_loop_v1.json").read_text())
    policy_states = []
    for row in comparison["paths"]["default"]["rows"]:
        exact = [name for name, value in originals.items() if value["initial_state_sha256"] == row["init_state_sha256"]]
        policy_states.append({"trial": row["trial"], "initial_state_sha256": row["init_state_sha256"],
                              "exact_original_demo_initial_state_matches": exact})
    result = {"schema_version": 1, "status": "verified_unique_full_action_correspondence",
              "source_hdf5_sha256": DIGEST, "source_manifest_sha256": sha256(manifest_path),
              "split_audit_sha256": sha256(audit_path), "task_instruction": audit["task_instruction"],
              "original_demos": len(originals), "matches": sorted(matches, key=lambda r: r["episode_index"]),
              "policy_trial_initial_state_comparison": policy_states,
              "no_op_filter_reference": "experiments/robot/libero/regenerate_libero_dataset.py:is_noop",
              "no_op_filter_source_sha256": sha256(ROOT / "experiments/robot/libero/regenerate_libero_dataset.py"),
              "new_rollouts": 0, "optimizer_updates": 0,
              "limits": ["Full action identity only; regenerated images/states are not asserted equal to original HDF5.",
                         "Original demo initial states are not the previous two benchmark policy states.",
                         "Replay must use each demo's full simulator initial state, not RLDS end-effector state."]}
    if args.verify:
        assert json.loads(OUTPUT.read_text()) == result, "Existing correspondence differs from complete recomputation"
    else:
        with OUTPUT.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "matched_selected_episodes": len(matches)}, indent=2))


if __name__ == "__main__":
    main()

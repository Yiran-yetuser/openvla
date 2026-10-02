"""CPU-only, episode-disjoint single-task preparation; never starts training.

Source HDF5 metadata is not a unique episode ID. TFDS order and full content
fingerprints identify episodes. Statistics use only nonterminal training frames.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

TASK = "pick up the black bowl next to the cookie box and place it on the plate"
BASE = Path("/home/yyz/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_fingerprint(steps):
    digest = hashlib.sha256()
    for step in steps:
        values = (step["language_instruction"], step["observation"]["image"],
                  step["observation"]["wrist_image"],
                  np.asarray(step["action"], dtype="<f4").tobytes(),
                  np.asarray(step["observation"]["state"], dtype="<f4").tobytes(),
                  bytes([bool(step[key]) for key in ("is_first", "is_last", "is_terminal")]))
        for value in values:
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
    return digest.hexdigest()


def select_episodes(episodes, instruction=TASK, train_count=8, validation_count=2, seed=7):
    if train_count < 1 or validation_count < 1:
        raise ValueError("Both partitions must contain whole episodes")
    # Byte-identical duplicate trajectories can never cross the partition.
    candidates = {}
    for episode in episodes:
        if episode["instruction"] == instruction and episode["eligible_frames"]:
            candidates.setdefault(episode["content_sha256"], episode)
    pool = list(candidates.values())
    if len(pool) < train_count + validation_count:
        raise ValueError("Insufficient distinct eligible demonstrations")
    random.Random(seed).shuffle(pool)
    return pool[:train_count], pool[train_count:train_count + validation_count]


def action_statistics(raw_actions):
    actions = np.asarray(raw_actions, dtype=np.float64).copy()
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected finite, nonempty Nx7 actions")
    actions[:, -1] = 1 - np.clip(actions[:, -1], 0, 1)
    result = {key: operation(actions, axis=0).tolist() for key, operation in
              (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max))}
    result.update(q01=np.quantile(actions, .01, axis=0).tolist(),
                  q99=np.quantile(actions, .99, axis=0).tolist(), mask=[True] * 6 + [False])
    if np.any(np.asarray(result["q99"])[:6] <= np.asarray(result["q01"])[:6]):
        raise ValueError("Degenerate motion normalization bounds")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reproduction/results/clean_task_split_audit_v1.json"))
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--data", type=Path, default=Path("/home/yyz/modified_libero_rlds"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing evidence must not be overwritten")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds

    index = json.loads((args.base / "model.safetensors.index.json").read_text())
    assert not any("lora_" in key for key in index["weight_map"])
    assert not (args.base / "adapter_config.json").exists()
    base_files = sorted(set(index["weight_map"].values())) + [
        "model.safetensors.index.json", "config.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "modeling_prismatic.py", "configuration_prismatic.py", "processing_prismatic.py",
    ]
    provenance = {name: {"bytes": (args.base / name).stat().st_size,
                          "sha256": sha256(args.base / name)} for name in base_files}
    print("Base checkpoint files hashed; scanning TFDS episodes", flush=True)
    builder = tfds.builder("libero_spatial_no_noops", data_dir=str(args.data))
    dataset = builder.as_dataset(split="train", shuffle_files=False, decoders={
        "steps": {"observation": {"image": tfds.decode.SkipDecoding(),
                                   "wrist_image": tfds.decode.SkipDecoding()}}})
    inventory, actions_by_episode, frames_by_episode = [], {}, {}
    for episode_id, episode in enumerate(dataset):
        digest = hashlib.sha256()
        actions, frame_rows, instructions = [], [], set()
        steps = list(episode["steps"].as_numpy_iterator())
        assert steps and bool(steps[0]["is_first"]) and bool(steps[-1]["is_last"])
        for timestep, step in enumerate(steps):
            instruction = step["language_instruction"].decode()
            instructions.add(instruction)
            raw = np.asarray(step["action"], dtype="<f4")
            state = np.asarray(step["observation"]["state"], dtype="<f4")
            assert raw.shape == (7,) and np.isfinite(raw).all() and np.isfinite(state).all()
            encoded = step["observation"]["image"]
            # Include both views and flags so duplicate detection is conservative.
            for value in (instruction.encode(), encoded, step["observation"]["wrist_image"],
                          raw.tobytes(), state.tobytes(),
                          bytes([bool(step[key]) for key in ("is_first", "is_last", "is_terminal")])):
                digest.update(len(value).to_bytes(8, "little"))
                digest.update(value)
            if not bool(step["is_last"]) and not bool(step["is_terminal"]):
                actions.append(raw.tolist())
                frame_rows.append({"episode_index": episode_id, "timestep": timestep,
                                   "encoded_image_sha256": hashlib.sha256(encoded).hexdigest(),
                                   "raw_action": raw.tolist()})
        assert len(instructions) == 1
        item = {"episode_index": episode_id, "instruction": next(iter(instructions)),
                "source_hdf5_metadata": episode["episode_metadata"]["file_path"].numpy().decode(),
                "content_sha256": digest.hexdigest(), "total_frames": len(steps),
                "eligible_frames": len(actions)}
        inventory.append(item)
        # Retain small action arrays only; do not retain/decode the dataset's images.
        actions_by_episode[episode_id] = actions
        frames_by_episode[episode_id] = frame_rows
        if (episode_id + 1) % 50 == 0:
            print(f"Scanned {episode_id + 1} episodes", flush=True)
    assert len(inventory) == builder.info.splits["train"].num_examples
    train, validation = select_episodes(inventory)
    assert {e["content_sha256"] for e in train}.isdisjoint(e["content_sha256"] for e in validation)
    train_actions = [a for e in train for a in actions_by_episode[e["episode_index"]]]
    validation_actions = np.asarray([a for e in validation for a in actions_by_episode[e["episode_index"]]])
    stats = action_statistics(train_actions)
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    clipped = (validation_actions[:, :6] < low[:6]) | (validation_actions[:, :6] > high[:6])
    fixed_frames = {}
    for partition, episodes in (("train", train), ("validation", validation)):
        fixed_frames[partition] = []
        for e in episodes:
            rows = frames_by_episode[e["episode_index"]]
            for position in sorted(set([0, len(rows) // 2, len(rows) - 1])):
                fixed_frames[partition].append(rows[position])
    report = {
        "status": "cpu_split_verified_training_not_started", "experiment": "clean_task_split_v1",
        "seed": 7, "task_instruction": TASK,
        "dataset": {"name": builder.name, "version": str(builder.version), "path": str(args.data),
                    "order": "TFDS train, shuffle_files=False", "episode_count": len(inventory)},
        "frame_policy": "exclude is_last OR is_terminal; no additional no-op filtering",
        "base": {"repo_id": "openvla/openvla-7b", "revision": args.base.name,
                 "path": str(args.base), "files": provenance, "adapter_present": False},
        "selection": {"train": train, "validation": validation,
                      "same_task_candidates": sum(e["instruction"] == TASK for e in inventory),
                      "duplicate_policy": "one representative per full content fingerprint before seeded selection"},
        "training_statistics": {"action": stats, "num_transitions": len(train_actions),
                                "num_trajectories": len(train), "source": "selected training episodes only"},
        "validation": {"num_transitions": len(validation_actions),
                       "outside_train_q01_q99_fraction_6d": clipped.mean(axis=0).tolist()},
        "fixed_frames": fixed_frames, "episode_inventory": inventory,
        "checks": {"finite_actions_states": True, "episode_language_consistent": True,
                   "episode_boundary_flags": True, "complete_inventory": True,
                   "content_disjoint": True, "train_only_statistics": True,
                   "base_files_present_and_hashed": True},
        "limits": ["No GPU model loading, gradient update, new adapter or task success result yet.",
                   "Episode holdout within one exploratory task, not unseen-task generalization.",
                   "Source HDF5 metadata is shared across demonstrations and is not a local file verification.",
                   "OXE base pretraining data provenance is not re-audited here.",
                   "Historical full-data adapters must not initialize this clean experiment."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "train_frames": len(train_actions),
                      "validation_frames": len(validation_actions), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()

"""Decode existing rollout videos and align frames with pre-action traces; no GPU.

This reader does not infer grasp/contact/success from joint positions. It emits
source hashes, command events, and losslessly saved keyframe contact sheets for
human visual inspection. MP4 frames are lossy and cannot match input RGB hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reproduction/results/clean_task_closed_loop_v1.json"
OUTPUT = ROOT / "reproduction/results/clean_task_video_audit_v1.json"
ASSETS = ROOT / "reproduction/results/clean_task_video_frames_v1"
KEYFRAMES = (0, 10, 20, 40, 60, 80, 100, 120, 150, 180, 200, 219)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_events(actions):
    if not actions or [a["step"] for a in actions] != list(range(len(actions))):
        raise ValueError("Trace must have nonempty consecutive policy steps")
    result = []
    previous = None
    for row in actions:
        action = np.asarray(row["simulator_action"], dtype=float)
        eef = np.asarray(row["eef_pos"], dtype=float)
        qpos = np.asarray(row["gripper_qpos"], dtype=float)
        if action.shape != (7,) or eef.shape != (3,) or qpos.shape != (2,):
            raise ValueError("Unexpected trace shape")
        if not all(np.isfinite(v).all() for v in (action, eef, qpos)):
            raise ValueError("Nonfinite trace")
        if action[-1] not in (-1, 1):
            raise ValueError("Expected binarized LIBERO gripper command")
        current = "close" if action[-1] > 0 else "open"
        if current != previous:
            result.append({"policy_step": row["step"], "command": current,
                           "eef_pos_m": eef.tolist(),
                           "finger_joint_separation_m": float(abs(qpos[0] - qpos[1]))})
            previous = current
    return result


def decode_video_frames(reader, expected_count):
    # Do not use list(reader): imageio's ffmpeg reader reports an infinite
    # length hint, which list() may try to preallocate before decoding.
    frames = []
    for frame in reader:
        if len(frames) >= expected_count:
            raise ValueError("Video has more frames than trace")
        frames.append(frame)
    if len(frames) != expected_count:
        raise ValueError("Video has fewer frames than trace")
    return frames


def video_for_row(row):
    trace_path = ROOT / row["file"]
    logs = list(trace_path.parent.glob("*.txt"))
    if len(logs) != 1:
        raise ValueError("Expected one source log for each path")
    videos = re.findall(r"^Saved rollout MP4 at path (.+)$", logs[0].read_text(), re.M)
    if len(videos) != 2 or row["trial"] not in (0, 1):
        raise ValueError("Video identities missing")
    video = (ROOT / videos[row["trial"]]).resolve()
    if not video.is_relative_to(ROOT) or not video.is_file():
        raise ValueError("Source video missing or outside project")
    return trace_path, logs[0], video


def contact_sheet(frames, actions, selected, title):
    width, height = 224, 224
    columns, margin, label_height = 4, 8, 38
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (width + margin) + margin,
                              42 + rows * (height + label_height + margin)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 6), title, fill="black")
    draw.text((margin, 23), "Pre-action images; labels are policy steps, NOT simulated seconds.", fill="black")
    for i, step in enumerate(selected):
        x = margin + (i % columns) * (width + margin)
        y = 42 + (i // columns) * (height + label_height + margin)
        source = Image.fromarray(frames[step])
        if source.size != (width, height):
            raise ValueError("Unexpected video dimensions; do not silently resize")
        sheet.paste(source, (x, y))
        row = actions[step]
        command = "CLOSE" if row["simulator_action"][-1] > 0 else "OPEN"
        separation = abs(row["gripper_qpos"][0] - row["gripper_qpos"][1]) * 1000
        draw.text((x, y + height + 2), f"step {step:03d}: {command}; joints {separation:.1f} mm", fill="black")
        draw.text((x, y + height + 18), f"EEF z={row['eef_pos'][2]:.3f} m", fill="black")
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"source": str(REPORT), "videos": 4, "new_rollouts": 0,
                          "optimizer_updates": 0, "output": str(OUTPUT)}, indent=2))
        return
    if OUTPUT.exists() or (ASSETS.exists() and any(ASSETS.iterdir())):
        raise FileExistsError("Inspect existing evidence; no overwrite")
    report = json.loads(REPORT.read_text())
    if report["status"] != "bounded_comparison_complete" or report["rollouts"] != 4:
        raise ValueError("Expected completed bounded comparison")
    ASSETS.mkdir(exist_ok=True)
    entries = []
    for path_name in ("default", "compatible"):
        for row in report["paths"][path_name]["rows"]:
            trace_path, log_path, video_path = video_for_row(row)
            if sha256(trace_path) != row["sha256"]:
                raise ValueError("Source trace changed")
            trace = json.loads(trace_path.read_text())
            actions = trace["actions"]
            events = command_events(actions)
            with imageio.get_reader(video_path) as reader:
                metadata = reader.get_meta_data()
                frames = decode_video_frames(reader, len(actions))
            if len(frames) != len(actions) or len(actions) != row["steps"]:
                raise ValueError("Video/trace length mismatch")
            selected = [step for step in KEYFRAMES if step < len(frames)]
            asset = ASSETS / f"{path_name}_trial{row['trial']}.png"
            contact_sheet(frames, actions, selected,
                          f"Task 6 | {path_name} | trial {row['trial']} | success={trace['success']}").save(asset)
            eef = np.asarray([a["eef_pos"] for a in actions])
            entries.append({"path": path_name, "trial": row["trial"], "success": trace["success"],
                            "trace_file": str(trace_path.relative_to(ROOT)), "trace_sha256": sha256(trace_path),
                            "log_file": str(log_path.relative_to(ROOT)), "log_sha256": sha256(log_path),
                            "video_file": str(video_path.relative_to(ROOT)), "video_sha256": sha256(video_path),
                            "video_frames": len(frames), "trace_actions": len(actions),
                            "video_playback_fps": metadata["fps"], "frame_dimensions": list(frames[0].shape),
                            "eef_start_m": eef[0].tolist(), "eef_end_pre_action_m": eef[-1].tolist(),
                            "eef_z_min_m": float(eef[:, 2].min()), "eef_z_max_m": float(eef[:, 2].max()),
                            "gripper_command_events": events,
                            "keyframes": [{"policy_step": step,
                                           "decoded_rgb_sha256": hashlib.sha256(frames[step].tobytes()).hexdigest()}
                                          for step in selected],
                            "contact_sheet": str(asset.relative_to(ROOT)), "contact_sheet_sha256": sha256(asset)})
    result = {"schema_version": 1, "status": "decoded_aligned_existing_videos",
              "source_report_sha256": sha256(REPORT), "new_rollouts": 0, "optimizer_updates": 0,
              "entries": entries, "limits": [
                  "Video frames are pre-action policy observations; final post-action scene is not recorded.",
                  "Videos use lossy H.264: decoded pixels cannot authenticate original input RGB hashes.",
                  "30 FPS is playback timing, not a claim about simulator control frequency.",
                  "Command events and finger joints alone cannot prove contact, grasp, or object motion.",
                  "Visual stage annotations must be separately recorded as manual observations, not machine ground truth."]}
    with OUTPUT.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "videos": len(entries),
                      "frames": sum(e["video_frames"] for e in entries), "new_rollouts": 0}, indent=2))


if __name__ == "__main__":
    main()

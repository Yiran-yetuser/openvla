"""CPU contracts for video/trace command annotations."""
import unittest

from reproduction.analyze_failure_videos import command_events, decode_video_frames


def row(step, command):
    return {"step": step, "simulator_action": [0.] * 6 + [command],
            "eef_pos": [0., 0., .2], "gripper_qpos": [.02, -.02]}


class FailureVideoTests(unittest.TestCase):
    def test_events_are_commands_not_grasps(self):
        events = command_events([row(0, -1), row(1, -1), row(2, 1), row(3, -1)])
        self.assertEqual([(e["policy_step"], e["command"]) for e in events],
                         [(0, "open"), (2, "close"), (3, "open")])
        self.assertAlmostEqual(events[1]["finger_joint_separation_m"], .04)
        self.assertNotIn("grasp_success", events[1])

    def test_nonconsecutive_steps_rejected(self):
        with self.assertRaises(ValueError):
            command_events([row(0, -1), row(2, 1)])

    def test_nonfinite_rejected(self):
        item = row(0, 1)
        item["eef_pos"][0] = float("nan")
        with self.assertRaises(ValueError):
            command_events([item])

    def test_nonbinary_gripper_rejected(self):
        with self.assertRaises(ValueError):
            command_events([row(0, .5)])

    def test_decode_does_not_use_reader_length_hint(self):
        class Reader:
            def __len__(self):
                raise MemoryError("infinite imageio length hint")

            def __iter__(self):
                yield 1
                yield 2
        self.assertEqual(decode_video_frames(Reader(), 2), [1, 2])

    def test_video_length_mismatch_rejected(self):
        for frames in ([1], [1, 2, 3]):
            with self.assertRaises(ValueError):
                decode_video_frames(iter(frames), 2)


if __name__ == "__main__":
    unittest.main()

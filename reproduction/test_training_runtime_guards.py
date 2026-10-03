"""CPU regression tests for one-update audit safety guards."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from reproduction.audit_training_runtime import preflight, file_hash


class RuntimeGuardsTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="openvla-guard-test-")
        self.root = Path(self.directory.name)
        (self.root / "adapter_model.safetensors").write_bytes(b"test fixture")

    def tearDown(self):
        self.directory.cleanup()

    @patch("reproduction.audit_training_runtime.subprocess.check_output", return_value="123, other_job, 8000 MiB")
    def test_busy_gpu_is_refused(self, _):
        with self.assertRaisesRegex(RuntimeError, "do not preempt"):
            preflight(self.root, self.root)

    @patch("reproduction.audit_training_runtime.subprocess.check_output", return_value="")
    @patch("reproduction.audit_training_runtime.shutil.disk_usage", return_value=SimpleNamespace(free=100))
    def test_low_disk_is_refused(self, *_):
        with self.assertRaisesRegex(RuntimeError, "reserve"):
            preflight(self.root, self.root)

    @patch("reproduction.audit_training_runtime.subprocess.check_output", return_value="")
    @patch("reproduction.audit_training_runtime.shutil.disk_usage", return_value=SimpleNamespace(free=4 * 2**30))
    def test_idle_with_disk_reserve(self, *_):
        result = preflight(self.root, self.root)
        self.assertGreater(result["required_disk_bytes"], 2**30)
        self.assertLess(result["required_disk_bytes"], result["free_disk_bytes"])

    def test_checkpoint_hash_detects_change(self):
        path = self.root / "adapter_model.safetensors"
        before = file_hash(path)
        path.write_bytes(b"different fixture")
        self.assertNotEqual(before, file_hash(path))


if __name__ == "__main__":
    unittest.main()

"""Standard-library-only fixture checks for selective artifact verification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from verify_artifacts import CHECKPOINT_FILES, MANIFESTS, verify_manifest


def record(payload):
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class ArtifactVerificationTests(unittest.TestCase):
    def test_default_cli_needs_only_validation_and_selection_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / MANIFESTS[0]
            manifest.parent.mkdir(parents=True)
            payload = b"frozen validation fixture"
            (manifest.parent / "val_l2.bin").write_bytes(payload)
            manifest.write_text(json.dumps({"val_l2.bin": record(payload)}))
            checkpoints = root / MANIFESTS[1]
            checkpoints.write_text(json.dumps({name: record(payload) for name in CHECKPOINT_FILES.values()}))
            command = [sys.executable, str(SCRIPTS / "verify_artifacts.py"), "--root", str(root)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("PASS:"), 1)
            result = subprocess.run([*command, "--checkpoints", "3b"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing artifact", result.stderr)
            model = checkpoints.parent / CHECKPOINT_FILES["3b"]
            model.parent.mkdir()
            model.write_bytes(payload)
            result = subprocess.run([*command, "--checkpoints", "3b"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("PASS:"), 2)
            result = subprocess.run([*command, "--checkpoints", "all"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("50b", result.stderr)

    def test_unselected_file_can_be_missing_but_selected_hash_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"checkpoint fixture"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"available.pt": record(payload), "missing.pt": record(payload)}))
            model = root / "available.pt"
            model.write_bytes(payload)
            self.assertEqual(verify_manifest(manifest, ["available.pt"]), 1)
            with self.assertRaisesRegex(FileNotFoundError, "missing.pt"):
                verify_manifest(manifest)
            model.write_bytes(b"X" + payload[1:])
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_manifest(manifest, ["available.pt"])
            with self.assertRaisesRegex(ValueError, "Unknown manifest entries"):
                verify_manifest(manifest, ["unknown.pt"])

    def test_paths_and_record_checks_apply_even_to_unselected_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "assets"
            base.mkdir()
            payload = b"fixture artifact"
            (base / "safe.bin").write_bytes(payload)
            manifest = base / "manifest.json"
            for unsafe in ("../outside.bin", "/outside.bin", ".", "sub/../outside.bin"):
                manifest.write_text(json.dumps({"safe.bin": record(payload), unsafe: record(payload)}))
                with self.assertRaisesRegex(ValueError, "Unsafe relative"):
                    verify_manifest(manifest, ["safe.bin"])
            outside = root / "outside.bin"
            outside.write_bytes(payload)
            (base / "escape.bin").symlink_to(outside)
            manifest.write_text(json.dumps({"safe.bin": record(payload), "escape.bin": record(payload)}))
            with self.assertRaisesRegex(ValueError, "outside its manifest directory"):
                verify_manifest(manifest, ["safe.bin"])
            manifest.write_text(json.dumps({"safe.bin": record(payload), "missing.bin": {"bytes": -1, "sha256": "bad"}}))
            with self.assertRaisesRegex(ValueError, "Invalid byte count or SHA256"):
                verify_manifest(manifest, ["safe.bin"])


if __name__ == "__main__":
    unittest.main()

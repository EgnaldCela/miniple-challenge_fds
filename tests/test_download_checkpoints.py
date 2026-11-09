"""Offline checks that unverified downloads never become baseline checkpoints."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_checkpoints import download_checkpoint, release_url
from verify_artifacts import CHECKPOINT_FILES


class CheckpointDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.payload = b"fixture checkpoint"
        self.baselines = self.root / "baselines"
        self.baselines.mkdir()
        record = {"bytes": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest()}
        (self.baselines / "manifest.json").write_text(json.dumps(
            {name: record for name in CHECKPOINT_FILES.values()}))
        self.destination = self.baselines / CHECKPOINT_FILES["3b"]
        self.url = release_url("example/course", "v1.0", "3b")

    def test_verified_download_and_existing_file_skip_network(self):
        with patch("download_checkpoints.urlopen", return_value=io.BytesIO(self.payload)) as request:
            self.assertEqual(download_checkpoint(self.root, "3b", self.url), self.destination)
            self.assertEqual(self.destination.read_bytes(), self.payload)
            download_checkpoint(self.root, "3b", self.url)
            self.assertEqual(request.call_count, 1)
        self.assertFalse((self.baselines / CHECKPOINT_FILES["50b"]).exists())
        self.assertFalse(list(self.baselines.rglob("*.part")))

    def test_invalid_or_failed_download_leaves_no_checkpoint(self):
        for payload, error in [(b"short", "Byte count"),
                               (b"X" + self.payload[1:], "SHA256"),
                               (self.payload + b"extra", "exceeds")]:
            with self.subTest(error=error):
                with patch("download_checkpoints.urlopen", return_value=io.BytesIO(payload)):
                    with self.assertRaisesRegex(ValueError, error):
                        download_checkpoint(self.root, "3b", self.url)
                self.assertFalse(self.destination.exists())
                self.assertFalse(list(self.baselines.rglob("*.part")))
        with patch("download_checkpoints.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(URLError):
                download_checkpoint(self.root, "3b", self.url)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.baselines.rglob("*.part")))

    def test_corrupt_existing_file_is_preserved_and_rejected(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"unrelated local file")
        with patch("download_checkpoints.urlopen") as request:
            with self.assertRaisesRegex(ValueError, "Byte count mismatch"):
                download_checkpoint(self.root, "3b", self.url)
            request.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"unrelated local file")

    def test_release_url_rejects_invalid_repository_and_encodes_tag(self):
        self.assertEqual(release_url("example/course", "course/v1.0", "50b"),
                         "https://github.com/example/course/releases/download/course%2Fv1.0/miniple-50b.pt")
        for repository in ("https://github.com/example/course", "example/course/extra", "example/course?x=1"):
            with self.assertRaises(ValueError):
                release_url(repository, "v1.0", "3b")

    def test_file_created_during_download_is_not_overwritten(self):
        link = os.link

        def competing_write(source, destination):
            destination.write_bytes(b"another process wrote this")
            return link(source, destination)

        with patch("download_checkpoints.urlopen", return_value=io.BytesIO(self.payload)):
            with patch("download_checkpoints.os.link", side_effect=competing_write):
                with self.assertRaises(FileExistsError):
                    download_checkpoint(self.root, "3b", self.url)
        self.assertEqual(self.destination.read_bytes(), b"another process wrote this")
        self.assertFalse(list(self.baselines.rglob("*.part")))


if __name__ == "__main__":
    unittest.main()

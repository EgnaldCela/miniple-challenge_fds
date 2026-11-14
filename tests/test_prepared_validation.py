"""CPU checks for rejecting altered cached holdouts before training starts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train
import validation_reference
from data_progress import publish_tokens


class PreparedValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.validation = bytes(range(18))
        self.tokenizer = b'{"fixture": "tokenizer"}'
        self.val_path = self.root / "val_l2.bin"
        self.tok_path = self.root / "tokenizer.json"
        self.val_path.write_bytes(self.validation)
        self.tok_path.write_bytes(self.tokenizer)
        self.meta = {
            "format_version": 2,
            "packing": {"split_documents_across_output_files": False},
            "vocab_size": 32768,
            "tokenizer": {"local_file": "tokenizer.json",
                          "sha256": hashlib.sha256(self.tokenizer).hexdigest()},
            "files": {"stage1_l2.bin": 4, "decay_l2.bin": 4,
                      "decay_l3_qa.bin": 4, "decay_l3_style.bin": 4, "val_l2.bin": 9},
        }
        for name, tokens in self.meta["files"].items():
            if name != "val_l2.bin":
                (self.root / name).write_bytes(bytes(tokens * 2))
        (self.root / "metadata.json").write_text(json.dumps(self.meta))
        for module, name, value in (
            (train, "VALIDATION_TOKENS", 9),
            (validation_reference, "VALIDATION_TOKENS", 9),
            (validation_reference, "VALIDATION_SHA256", hashlib.sha256(self.validation).hexdigest()),
            (validation_reference, "TOKENIZER_SHA256", hashlib.sha256(self.tokenizer).hexdigest()),
        ):
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_valid_data_keeps_existing_metadata_and_resume_fingerprint(self):
        before = copy.deepcopy(self.meta)
        train.validate_prepared_data(self.root, self.meta)
        self.assertEqual(self.meta, before)

    def test_same_length_changed_validation_cannot_supply_its_own_hash(self):
        changed = b"\xff" + self.validation[1:]
        self.val_path.write_bytes(changed)
        self.meta["validation_sha256"] = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, "Validation SHA256 mismatch"):
            train.validate_prepared_data(self.root, self.meta)

    def test_changed_tokenizer_is_rejected_even_when_metadata_matches(self):
        changed = b'{"fixture": "different"}'
        self.tok_path.write_bytes(changed)
        self.meta["tokenizer"]["sha256"] = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, "Tokenizer SHA256 mismatch"):
            train.validate_prepared_data(self.root, self.meta)

    def test_metadata_tokenizer_hash_check_is_retained(self):
        self.meta["tokenizer"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Tokenizer SHA256 mismatch"):
            train.validate_prepared_data(self.root, self.meta)

    def test_validation_cannot_be_omitted_or_shortened_in_metadata(self):
        for tokens in (None, 8):
            with self.subTest(tokens=tokens):
                if tokens is None:
                    self.meta["files"].pop("val_l2.bin")
                else:
                    self.meta["files"]["val_l2.bin"] = tokens
                with self.assertRaisesRegex(ValueError, "frozen validation stream"):
                    train.validate_prepared_data(self.root, self.meta)

    def test_progressive_validation_waits_then_checks_complete_bytes(self):
        self.meta["progressive_preparation"] = True
        self.val_path.write_bytes(self.validation[:8])
        for name, target in self.meta["files"].items():
            ready = 4 if name == "val_l2.bin" else target
            publish_tokens(self.root / name, ready, target, complete=ready == target)
        events = []

        def finish_validation(path, needed):
            self.assertEqual((path, needed), (self.val_path, 9))
            events.append("wait")
            path.write_bytes(self.validation)
            publish_tokens(path, needed, needed, complete=True)
            return needed

        def verify(path, expected_tokens):
            events.append("verify")
            return validation_reference.verify_validation_reference(path, expected_tokens)

        with patch.object(train, "wait_for_tokens", side_effect=finish_validation), \
                patch.object(train, "verify_validation_reference", side_effect=verify):
            train.validate_prepared_data(self.root, self.meta)
        self.assertEqual(events, ["wait", "verify"])

    def test_validation_only_exits_before_device_or_output_initialization(self):
        args = argparse.Namespace(data_dir=self.root, validate_data_only=True)
        with patch.object(train, "parse_args", return_value=args), \
                patch.object(train, "choose_device", side_effect=AssertionError("device accessed")), \
                patch.object(train.torch.cuda, "is_available", side_effect=AssertionError("CUDA accessed")), \
                patch.object(train, "MiniPLE", side_effect=AssertionError("model initialized")):
            train.main()
        self.assertFalse((self.root / "runs").exists())

    def test_normal_training_rejects_bad_validation_before_cuda_access(self):
        self.val_path.write_bytes(b"\xff" + self.validation[1:])
        args = argparse.Namespace(data_dir=self.root, validate_data_only=False)
        with patch.object(train, "parse_args", return_value=args), \
                patch.object(train.torch.cuda, "is_available", side_effect=AssertionError("CUDA accessed")):
            with self.assertRaisesRegex(ValueError, "Validation SHA256 mismatch"):
                train.main()

    def test_validation_only_flag_is_explicit(self):
        for flags, expected in (([], False), (["--validate-data-only"], True)):
            with patch.object(sys, "argv", ["train.py", "--data-dir", str(self.root), *flags]):
                self.assertIs(train.parse_args().validate_data_only, expected)


if __name__ == "__main__":
    unittest.main()

"""CPU fixtures for fail-fast preparation; no Hub access or corpus download."""
import contextlib
import hashlib
import io
import re
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare_data
import validation_reference
import data_progress


class FixtureTokenizer:
    def token_to_id(self, token):
        assert token == prepare_data.BOS_TOKEN
        return 0

    def get_vocab_size(self):
        return 32768

    def encode_batch(self, texts, add_special_tokens=False):
        assert add_special_tokens is False
        return [SimpleNamespace(ids=[int(token) for token in text.split()]) for text in texts]


def write_tokens(path, tokens):
    np.asarray(tokens, dtype=np.uint16).tofile(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(*arguments):
    with patch.object(sys, "argv", ["prepare_data.py", *arguments]):
        return prepare_data.parse_args()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        progress = patch.object(prepare_data, "tqdm")
        progress.start()
        self.addCleanup(progress.stop)

    def test_first_mismatch_stops_within_first_batch_before_training_bytes(self):
        consumed = []

        def documents():
            for index in range(20):
                consumed.append(index)
                yield {"content": "11 12 13 14"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.bin"
            expected_sha = write_tokens(reference, [0, 99, 12])
            validation, training = root / "val_l2.bin", root / "stage1_l2.bin"
            with self.assertRaisesRegex(ValueError, r"at token 1: expected 99, got 11"):
                prepare_data.pack_stream_across_targets(
                    documents(), FixtureTokenizer(), "content",
                    [(validation, 3), (training, 4)], 2, "fixture",
                    validation_reference=reference, validation_sha256=expected_sha)
            self.assertEqual(consumed, [0, 1])
            self.assertEqual(validation.read_bytes(), b"")
            self.assertEqual(training.read_bytes(), b"")
            self.assertEqual(hashlib.sha256(reference.read_bytes()).hexdigest(), expected_sha)

    def test_matching_reference_drops_document_tail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.bin"
            expected_sha = write_tokens(reference, [0, 11, 12])
            for batch_size in (1, 2):
                with self.subTest(batch_size=batch_size):
                    out = root / str(batch_size)
                    out.mkdir()
                    validation, training = out / "val_l2.bin", out / "stage1_l2.bin"
                    counts = prepare_data.pack_stream_across_targets(
                        [{"content": "11 12 13 14"}, {"content": "21 22 23 24"}],
                        FixtureTokenizer(), "content", [(validation, 3), (training, 4)],
                        batch_size, "fixture", validation_reference=reference,
                        validation_sha256=expected_sha)
                    self.assertEqual(counts, {"val_l2.bin": 3, "stage1_l2.bin": 4})
                    self.assertEqual(validation.read_bytes(), reference.read_bytes())
                    self.assertEqual(np.fromfile(training, dtype=np.uint16).tolist(), [0, 21, 22, 23])

    def test_final_sha_guard_remains_after_incremental_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.bin"
            write_tokens(reference, [0, 11, 12])
            validation, training = root / "val_l2.bin", root / "stage1_l2.bin"
            with self.assertRaisesRegex(ValueError, "Generated validation SHA256 mismatch"):
                prepare_data.pack_stream_across_targets(
                    [{"content": "11 12 13 14"}, {"content": "21 22 23"}],
                    FixtureTokenizer(), "content", [(validation, 3), (training, 4)],
                    2, "fixture", validation_reference=reference, validation_sha256="0" * 64)
            self.assertEqual(validation.read_bytes(), reference.read_bytes())
            self.assertEqual(training.read_bytes(), b"")

    def test_progressive_ready_records_follow_verified_document_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in ("token mismatch", "SHA mismatch", "match"):
                with self.subTest(condition=condition):
                    out = root / condition
                    out.mkdir()
                    reference = out / "reference.bin"
                    expected_sha = write_tokens(
                        reference, [0, 99, 12] if condition == "token mismatch" else [0, 11, 12])
                    if condition == "SHA mismatch":
                        expected_sha = "0" * 64
                    validation, training = out / "val_l2.bin", out / "stage1_l2.bin"

                    def publish(path, tokens, target, **kwargs):
                        if path == validation:
                            self.assertEqual(training.read_bytes(), b"")
                            self.assertFalse(data_progress.progress_path(training).exists())
                        else:
                            self.assertTrue(data_progress.read_progress(validation)["complete"])
                            self.assertEqual(validation.read_bytes(), reference.read_bytes())
                            self.assertEqual(np.fromfile(training, dtype=np.uint16).tolist(),
                                             [0, 21, 22, 23])
                        data_progress.publish_tokens(path, tokens, target, **kwargs)

                    with patch.object(prepare_data, "publish_tokens", side_effect=publish) as announced, \
                            contextlib.redirect_stdout(io.StringIO()):
                        call = lambda: prepare_data.pack_stream_across_targets(
                            [{"content": "11 12 13 14"}, {"content": "21 22 23 24"}],
                            FixtureTokenizer(), "content", [(validation, 3), (training, 4)],
                            2, "fixture", progressive=True, validation_reference=reference,
                            validation_sha256=expected_sha)
                        if condition != "match":
                            with self.assertRaises(ValueError):
                                call()
                            announced.assert_not_called()
                            self.assertFalse(data_progress.progress_path(validation).exists())
                            self.assertFalse(data_progress.progress_path(training).exists())
                            self.assertEqual(training.read_bytes(), b"")
                        else:
                            call()
                            self.assertEqual([c.args[0] for c in announced.call_args_list],
                                             [validation, training])
                            for path, count in ((validation, 3), (training, 4)):
                                record = data_progress.read_progress(path)
                                self.assertEqual(record["tokens"], count)
                                self.assertTrue(record["complete"])
                                self.assertIsNone(record["error"])

    def test_standalone_defaults_use_both_bundled_references(self):
        args = parse("--out-dir", "unused")
        self.assertEqual(args.validation_reference, ROOT / "baselines/validation/val_l2.bin")
        self.assertEqual(args.tokenizer_json, ROOT / "baselines/validation/tokenizer.json")
        self.assertFalse(args.check_validation_only)

    def test_corrupt_reference_or_tokenizer_is_rejected_before_hub_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, tokenizer = root / "reference.bin", root / "tokenizer.json"
            valid_sha = write_tokens(reference, [0, 11, 12])
            token_bytes = b"frozen tokenizer fixture"
            token_sha = hashlib.sha256(token_bytes).hexdigest()
            with patch.multiple(validation_reference, VALIDATION_TOKENS=3,
                                VALIDATION_SHA256=valid_sha, TOKENIZER_SHA256=token_sha):
                for corrupt in ("validation", "tokenizer"):
                    with self.subTest(corrupt=corrupt):
                        write_tokens(reference, [0, 11, 99] if corrupt == "validation" else [0, 11, 12])
                        tokenizer.write_bytes(b"corrupt tokenizer" if corrupt == "tokenizer" else token_bytes)
                        args = parse("--out-dir", str(root / corrupt), "--val-tokens", "2",
                                     "--validation-reference", str(reference),
                                     "--tokenizer-json", str(tokenizer))
                        with patch.object(prepare_data, "parse_args", return_value=args), \
                                patch.object(prepare_data, "resolve_dataset_revision") as resolve, \
                                patch.object(prepare_data, "l2_stream") as stream, \
                                patch.object(prepare_data, "load_fixed_nanochat_tokenizer") as load:
                            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                                prepare_data.main()
                            resolve.assert_not_called()
                            stream.assert_not_called()
                            load.assert_not_called()

    def test_validation_only_resolves_no_l3_and_creates_no_training_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, tokenizer, out = root / "reference.bin", root / "tokenizer.json", root / "out"
            valid_sha = write_tokens(reference, [0, 11, 12])
            tokenizer.write_bytes(b"frozen tokenizer fixture")
            args = parse("--out-dir", str(out), "--val-tokens", "2", "--check-validation-only",
                         "--validation-reference", str(reference), "--tokenizer-json", str(tokenizer))
            with patch.multiple(validation_reference, VALIDATION_TOKENS=3,
                                VALIDATION_SHA256=valid_sha,
                                TOKENIZER_SHA256=hashlib.sha256(tokenizer.read_bytes()).hexdigest()), \
                    patch.object(prepare_data, "parse_args", return_value=args), \
                    patch.object(prepare_data, "version", side_effect=lambda name:
                                 prepare_data.DATASETS_VERSION if name == "datasets" else "fixture"), \
                    patch.object(prepare_data, "load_fixed_nanochat_tokenizer",
                                 return_value=(FixtureTokenizer(), tokenizer)), \
                    patch.object(prepare_data, "resolve_dataset_revision", return_value="fixed-l2") as resolve, \
                    patch.object(prepare_data, "l2_stream",
                                 return_value=iter([{"content": "11 12 13 14"}])) as l2, \
                    patch.object(prepare_data, "l3_stream") as l3, \
                    patch.object(prepare_data, "pack_single") as single, \
                    contextlib.redirect_stdout(io.StringIO()):
                prepare_data.main()
                resolve.assert_called_once_with(args.l2_dataset, args.l2_revision)
                l2.assert_called_once_with(args, 1338, "fixed-l2")
                l3.assert_not_called()
                single.assert_not_called()
            self.assertEqual((out / "val_l2.bin").read_bytes(), reference.read_bytes())
            self.assertFalse((out / "metadata.json").exists())
            self.assertEqual(list(out.glob("*l3*")), [])
            self.assertFalse((out / "stage1_l2.bin").exists())
            self.assertFalse((out / "decay_l2.bin").exists())

    def test_incompatible_datasets_is_rejected_before_hub_access(self):
        for installed in ("4.8.4", "5.2.0"):
            with self.subTest(installed=installed), tempfile.TemporaryDirectory() as directory:
                self.check_rejected_version(installed, directory)

    def check_rejected_version(self, installed, directory):
        args = parse("--out-dir", directory, "--check-validation-only")
        with patch.object(prepare_data, "parse_args", return_value=args), \
                patch.object(prepare_data, "version", side_effect=lambda name:
                             installed if name == "datasets" else "fixture"), \
                patch.object(prepare_data, "resolve_dataset_revision") as resolve, \
                patch.object(prepare_data, "l2_stream") as stream, \
                patch.object(prepare_data, "load_fixed_nanochat_tokenizer") as load, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(
                    RuntimeError, rf"requires datasets=={re.escape(prepare_data.DATASETS_VERSION)}; found {re.escape(installed)}"):
                prepare_data.main()
            resolve.assert_not_called()
            stream.assert_not_called()
            load.assert_not_called()
        self.assertEqual(list(Path(directory).iterdir()), [])

    def test_l2_and_l3_read_only_projected_text_with_bounded_buffering(self):
        class ScanOptions:
            def __init__(self, **kwargs):
                self.options = kwargs

        source, shuffled = Mock(), object()
        source.shuffle.return_value = shuffled
        load = Mock(return_value=source)
        datasets = SimpleNamespace(load_dataset=load, Features=dict,
                                   Value=lambda dtype: ("feature", dtype))
        arrow_dataset = SimpleNamespace(ParquetFragmentScanOptions=ScanOptions)
        args = parse("--out-dir", "unused", "--l2-text-field", "l2_text",
                     "--l3-text-field", "l3_text")
        with patch.dict(sys.modules, {"datasets": datasets,
                                     "pyarrow": SimpleNamespace(dataset=arrow_dataset),
                                     "pyarrow.dataset": arrow_dataset}):
            self.assertIs(prepare_data.l2_stream(args, 1338, "l2-revision"), shuffled)
            self.assertIs(prepare_data.l3_stream(args, "qa-config", 1339, "l3-revision"), shuffled)
        for call, dataset_args, field, split, revision in (
                (load.call_args_list[0], (args.l2_dataset,), "l2_text", "en", "l2-revision"),
                (load.call_args_list[1], (args.l3_dataset, "qa-config"), "l3_text", "train", "l3-revision")):
            self.assertEqual(call.args, dataset_args)
            self.assertEqual(call.kwargs["split"], split)
            self.assertEqual(call.kwargs["revision"], revision)
            self.assertIs(call.kwargs["streaming"], True)
            self.assertEqual(call.kwargs["columns"], [field])
            self.assertEqual(call.kwargs["features"], {field: ("feature", "string")})
            self.assertEqual(call.kwargs["batch_size"], 2000)
            self.assertEqual(call.kwargs["fragment_scan_options"].options,
                             {"pre_buffer": False, "use_buffered_stream": True, "buffer_size": 65536})
        self.assertEqual([c.kwargs for c in source.shuffle.call_args_list], [
            {"seed": 1338, "buffer_size": 20000, "max_buffer_input_shards": 10},
            {"seed": 1339, "buffer_size": 20000, "max_buffer_input_shards": 10}])

    def test_shuffle_explicitly_uses_ten_input_shards(self):
        class Stream:
            def shuffle(self, *, seed, buffer_size, max_buffer_input_shards=99):
                return seed, buffer_size, max_buffer_input_shards

        self.assertEqual(prepare_data._shuffle(Stream(), 1338, 20_000), (1338, 20_000, 10))
        unshuffled = Mock()
        self.assertIs(prepare_data._shuffle(unshuffled, 1338, 0), unshuffled)
        unshuffled.shuffle.assert_not_called()


if __name__ == "__main__":
    unittest.main()

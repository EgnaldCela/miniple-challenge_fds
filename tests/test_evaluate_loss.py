"""Small CPU checks for the validation protocol; no model/data download is needed."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_loss
from model import MiniPLE, MiniPLEConfig
from train import MemmapTokenLoader, evaluate


class ValidationContractTests(unittest.TestCase):
    def test_full_vocabulary_loss_is_independent_of_microbatch_and_restores_state(self):
        torch.manual_seed(1337)
        model = MiniPLE(MiniPLEConfig(
            vocab_size=32, max_seq_len=8, n_layers=1, d_model=8, d_ff=16,
            n_heads=2, n_kv_heads=1, head_dim=4, local_window=4,
            attention_pattern="G", ple_dim=4, attn_backend="sdpa", loss_backend="chunked"))
        model.train()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val.bin"
            tokens = np.arange(32, dtype=np.uint16)
            tokens.tofile(path)
            x = torch.from_numpy(tokens[:24].astype(np.int64)).reshape(3, 8)
            y = torch.from_numpy(tokens[1:25].astype(np.int64)).reshape(3, 8)
            with torch.no_grad():
                expected = model(x, y).item()
            for batch_size in (1, 2, 3):
                loader = MemmapTokenLoader(path, batch_size, 8, torch.device("cpu"), seed=1337)
                loader.next_batch()
                old_state = loader.state_dict()
                actual = evaluate(model, loader, 3, torch.device("cpu"), torch.float32)
                self.assertAlmostEqual(actual, expected, places=5)
                self.assertEqual(loader.state_dict(), old_state)
                self.assertTrue(model.training)

    def test_frozen_tokenizer_and_validation_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = b"test tokenizer fixture"
            validation = bytes((evaluate_loss.SEQUENCES * evaluate_loss.SEQ_LEN + 1) * 2)
            tok_sha = hashlib.sha256(tokenizer).hexdigest()
            val_sha = hashlib.sha256(validation).hexdigest()
            (root / "tokenizer.json").write_bytes(tokenizer)
            (root / "val_l2.bin").write_bytes(validation)
            (root / "metadata.json").write_text(json.dumps({
                "format_version": 2, "vocab_size": 32768,
                "tokenizer": {"sha256": tok_sha},
                "files": {"val_l2.bin": len(validation) // 2},
            }))
            with patch.object(evaluate_loss, "TOKENIZER_SHA256", tok_sha), \
                    patch.object(evaluate_loss, "VALIDATION_SHA256", val_sha):
                self.assertEqual(evaluate_loss.validation_files(root), (root / "val_l2.bin", val_sha))
                (root / "tokenizer.json").write_bytes(b"different tokenizer")
                with self.assertRaisesRegex(ValueError, "Tokenizer"):
                    evaluate_loss.validation_files(root)
                (root / "tokenizer.json").write_bytes(tokenizer)
                (root / "val_l2.bin").write_bytes(b"\x01" + validation[1:])
                with self.assertRaisesRegex(ValueError, "frozen course stream"):
                    evaluate_loss.validation_files(root)
                (root / "val_l2.bin").write_bytes(validation[:-2])
                with self.assertRaisesRegex(ValueError, "token count"):
                    evaluate_loss.validation_files(root)


if __name__ == "__main__":
    unittest.main()

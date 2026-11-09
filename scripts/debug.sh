#!/usr/bin/env bash
# Fast environment/data check; does not train or reserve GPU memory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
"${PYTHON:-$PWD/.venv/bin/python}" - "${DATA_DIR:-data/ultrafineweb_3b}" <<'PY'
import sys
from pathlib import Path

import torch
from model import MiniPLEConfig
from train import load_metadata, validate_prepared_data

data_dir = Path(sys.argv[1])
cfg = MiniPLEConfig()
cfg.validate()
print(f"PyTorch {torch.__version__}; CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16: {torch.cuda.is_bf16_supported()}")
print(f"Model: {cfg.n_layers} layers, {cfg.attention_pattern} layout, PLE {cfg.ple_dim}")
print(f"Position encoding: local=RoPE; global={'NoPE' if cfg.global_nope else 'RoPE'}")
print(f"Attention windows: local={cfg.local_window}; global/context={cfg.max_seq_len}")
print(f"Learned attention sinks: {cfg.n_layers * cfg.n_heads} (one logit per head)")
meta = load_metadata(data_dir)
validate_prepared_data(data_dir, meta)
print(f"Data OK: {data_dir}; {len(meta['files'])} token files")
PY

#!/usr/bin/env bash
# Tested on Linux x86_64, Python 3.12, and RTX 3090.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python3.12}"
"$python_bin" -c 'import platform,sys; assert sys.version_info[:2] == (3,12) and platform.system() == "Linux" and platform.machine() == "x86_64", "Use Linux x86_64 and Python 3.12"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt packaging ninja psutil einops==0.8.2
.venv/bin/python -m pip install --no-deps liger-kernel==0.8.4
.venv/bin/python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
.venv/bin/python -c 'import torch,flash_attn; from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; assert torch.cuda.is_available(); print(torch.__version__, flash_attn.__version__, torch.cuda.get_device_name(0))'

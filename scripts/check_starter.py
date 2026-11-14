"""Check syntax and the 3B launch contract without a GPU, dependencies, or downloads."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def parser_defaults(filename, arguments):
    tree = ast.parse((ROOT / filename).read_text(), filename)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = {"argparse": argparse, "Path": Path,
                 "__file__": str(ROOT / filename),
                 "NANOCHAT_TOKENIZER_REPO": "unused", "NANOCHAT_TOKENIZER_REVISION": "unused"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    old_argv = sys.argv
    try:
        sys.argv = [filename, *arguments]
        return namespace["parse_args"]()
    finally:
        sys.argv = old_argv


def main():
    sources = [*ROOT.glob("*.py"), *(ROOT / "scripts").glob("*.py")]
    for source in sources:
        ast.parse(source.read_text(), str(source))
    for script in (ROOT / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    for name, args in [("train.py", ["--data-dir", "unused"]),
                       ("prepare_data.py", ["--out-dir", "unused"])]:
        defaults = parser_defaults(name, args)
        assert (defaults.stage1_tokens, defaults.decay_tokens, defaults.mix_start_tokens) == (
            2_100_000_000, 900_000_000, 1_500_000_000)
        assert defaults.seed == 1337
        if name == "train.py":
            assert defaults.warmup_tokens == 10_000_000 and defaults.resume is None
            assert defaults.validate_data_only is False
        else:
            assert defaults.l2_revision == "02c85641e3d19a854be2e09139c25adaa9518063"
            assert defaults.l3_revision == "bc3b1ba986fcaef6871b9790a413b16267c2de0f"
            assert defaults.validation_reference == ROOT / "baselines/validation/val_l2.bin"
            assert defaults.tokenizer_json == ROOT / "baselines/validation/tokenizer.json"

    # Run the real shell wrapper in an isolated tree with inert GPU/Python commands.
    with tempfile.TemporaryDirectory(prefix="miniple-launch-check-") as directory:
        root = Path(directory)
        (root / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/train.sh", root / "scripts/train.sh")
        bindir = root / ".venv/bin"
        bindir.mkdir(parents=True)
        (bindir / "python").write_text(
            f"#!{sys.executable}\nimport json, os, sys\n"
            "with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if '--validate-data-only' in sys.argv: sys.exit(int(os.environ.get('MOCK_VALIDATE_EXIT', '0')))\n"
            "if 'train.py' in sys.argv: sys.exit(int(os.environ.get('MOCK_TRAIN_EXIT', '0')))\n")
        (bindir / "nvidia-smi").write_text(
            '#!/bin/bash\nprintf "%s\\n" "$*" >>"$GPU_LOG"\n'
            'if [[ "$*" == *"--query-gpu=uuid"* ]]; then echo GPU-MOCK; fi\n')
        (bindir / "flock").write_text("#!/bin/bash\nexit 0\n")
        for file in bindir.iterdir():
            file.chmod(0o755)
        env = dict(os.environ)
        for key in ("PYTHON", "OUT_DIR", "DATA_DIR", "STAGE1_TOKENS", "DECAY_TOKENS",
                    "MIX_START_TOKENS", "WARMUP_TOKENS", "GPU_ID"):
            env.pop(key, None)
        log = root / "calls.jsonl"
        gpu_log = root / "gpu_calls.log"
        env.update(PATH=str(bindir) + os.pathsep + env["PATH"], CALL_LOG=str(log),
                   GPU_LOG=str(gpu_log))
        command = ["bash", str(root / "scripts/train.sh")]
        missing = subprocess.run(command, env=env, capture_output=True, text=True)
        assert missing.returncode != 0 and "Missing frozen course reference" in missing.stderr
        assert not log.exists() and not gpu_log.exists()
        reference = root / "baselines/validation"
        reference.mkdir(parents=True)
        (reference / "val_l2.bin").touch()
        (reference / "tokenizer.json").touch()
        subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        prepare, validate, train = [json.loads(line) for line in log.read_text().splitlines()]
        assert prepare[:2] == ["-u", "prepare_data.py"]
        assert prepare[prepare.index("--validation-reference") + 1] == "baselines/validation/val_l2.bin"
        assert prepare[prepare.index("--tokenizer-json") + 1] == "baselines/validation/tokenizer.json"
        assert validate == ["-u", "train.py", "--data-dir", "data/ultrafineweb_3b", "--validate-data-only"]
        assert train[:2] == ["-u", "train.py"]
        expected = {"--data-dir": "data/ultrafineweb_3b", "--out-dir": "runs/baseline_3b",
                    "--stage1-tokens": "2100000000", "--decay-tokens": "900000000",
                    "--mix-start-tokens": "1500000000", "--warmup-tokens": "10000000",
                    "--micro-batch": "20", "--grad-accum": "4", "--seq-len": "2048",
                    "--seed": "1337"}
        for flag, value in expected.items():
            assert train[train.index(flag) + 1] == value, (flag, train)
        assert "--resume" not in train and "--compile" in train
        assert "--full-softmax-at-decay" in train
        run = root / "runs/baseline_3b"
        assert (run / "status").read_text().strip() == "complete"
        (run / "metrics.jsonl").touch()
        refused = subprocess.run(command, env=env, capture_output=True, text=True)
        assert refused.returncode != 0 and "already contains" in refused.stderr
        checkpoint = run / "latest.pt"
        checkpoint.touch()
        subprocess.run([*command, "--resume", str(checkpoint)], env=env,
                       check=True, capture_output=True, text=True)
        resumed = json.loads(log.read_text().splitlines()[-1])
        assert resumed[-2:] == ["--resume", str(checkpoint)]
        data = root / "data/ultrafineweb_3b"
        data.mkdir(parents=True)
        metadata = data / "metadata.json"
        metadata.write_text("{}")
        before = len(log.read_text().splitlines())
        gpu_before = gpu_log.read_text()
        # Cached data must pass CPU validation before any GPU query or training call.
        failed = subprocess.run(command, env=dict(env, OUT_DIR="runs/invalid_data", MOCK_VALIDATE_EXIT="7"),
                                capture_output=True, text=True)
        assert failed.returncode == 7
        assert len(log.read_text().splitlines()) == before + 1
        assert json.loads(log.read_text().splitlines()[-1]) == validate
        assert gpu_log.read_text() == gpu_before
        assert metadata.read_text() == "{}"
        assert (root / "runs/invalid_data/status").read_text().strip() == "failed (exit 7)"
        failed_train = subprocess.run(command, env=dict(env, OUT_DIR="runs/failed_train", MOCK_TRAIN_EXIT="9"),
                                      capture_output=True, text=True)
        assert failed_train.returncode == 9
    print(f"PASS: {len(sources)} Python files, shell syntax, 3B defaults, fresh/resume guards, mandatory references, CPU preflight before GPU access, error propagation.")
    print("No model was trained; no GPU, network, or third-party Python package was used.")


if __name__ == "__main__":
    main()

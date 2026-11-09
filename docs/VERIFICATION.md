# Handout verification — 7 November 2026

This record describes packaging checks, not a new training result. The revised handout excludes training data; the frozen validation set/tokenizer and both checkpoints remain.

## Passed

- Starter syntax and shell checks; explicit 3B defaults, seed and pinned source revisions; inert fresh-run protection and resume-invocation checks; preparation with and without bundled reference files; existing-data handling and trainer error propagation.
- Three small tests: frozen validation/tokenizer rejection checks; full-vocabulary loss invariance across microbatch sizes with loader/model state restoration; artifact-verifier success/corruption/missing/unsafe-path checks.
- All three files listed in `baselines/validation/manifest.json` match their byte counts and SHA256 hashes. Both full baseline checkpoints match their historical recorded byte counts and SHA256 hashes.
- The retained evaluation-only metadata, frozen validation hash, and tokenizer hash are accepted by the loss evaluator. Training binaries have been removed; preparation remains a student step.
- Both checkpoints load strictly into the student model implementation on CPU and contain 146,496,016 unique parameters. Each completes a finite 32-target CPU forward/loss probe. These tiny probes are compatibility checks, not benchmark scores; outputs are in `baselines/checkpoint_load_check.json`.
- Model, attention sinks, Muon, sampled-softmax, tokenizer, and four-task evaluator source hashes match the historical 3B manifest. The original source archive's members match that manifest. Current generation code includes later sampling defaults/top-p support; current training/data code includes later preparation safeguards and the documented student defaults.
- Baseline final losses, token budgets, schedules, benchmark figures, and checkpoint hashes were independently checked against the supplied raw records. The 3B plot was visually inspected.
- Relative Markdown links resolve and all packaged JSON documents parse. No access password is stored in student-facing code or documentation.

## Commands

```bash
python3 scripts/check_starter.py
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py
```

The small Python tests and artifact checks were run with the existing local Python 3.10 environment; the training installer targets Linux/Python 3.12 as documented. The artifact verifier uses portable standard-library hashing.

## Repository distribution update — 9 November 2026

- The source repository includes the frozen validation/tokenizer reference. Full checkpoints, training data, generated runs, caches, and Office lock files are excluded from Git.
- A clean export of the Git index passed the starter check and default artifact verification with no checkpoints present.
- All ten unit tests passed on local Python 3.12/PyTorch 2.11.0. Downloader tests cover successful selected downloads, skipping already verified files, rejected corrupt/truncated/oversized downloads, network failure cleanup, and preservation of existing files, including a concurrent write at download completion. Verifier tests cover explicit selection, missing unselected checkpoints, checksum failures, and manifest/path protections.
- The existing 3B and 50B checkpoints passed the downloader's verification without network access. Both checkpoints and all frozen validation assets retain their original recorded hashes.
- `source_manifest.json` records byte counts and SHA256 values for the versioned files, excluding itself. Historical manifests and archives remain unchanged.

The downloader uses public GitHub Release URLs from `distribution.json`; private-release access is documented separately. Repository publication and live release downloads are pending. No training or GPU benchmark was run for this packaging update.

## Limits

No full data reconstruction, new 3B/50B GPU training, full-checkpoint validation, four-task benchmark rerun, or fresh Linux/CUDA installation was performed while assembling this revision. Data-preparation arguments and launcher behavior were checked without downloads; complete source reconstruction was not rerun. The supplied benchmark scores and curves are the completed historical runs, with their raw evidence and environment metadata. SFT, Italian training/evaluation, and proposed architecture improvements are assignment tasks, not implemented solutions.

Course staff should run the documented setup and a matched baseline on the allocated course environment before grading speed claims. No hidden test set or course schedule/GPU entitlement was created. These release decisions are listed in `INSTRUCTOR.md`.

# Instructor release notes

The source repository contains editable pretraining code defaulting to a fresh 3B run, historical curves and benchmark outputs, and a frozen validation/tokenizer reference. The two full historical checkpoints are distributed separately as optional GitHub Release assets. Training data is not distributed: students reconstruct it using the supplied preparation code. SFT, Italian adaptation, and the proposed architecture changes remain student tasks.

The configured target is `spindro/miniple-challenge`, release `v1.0`. The package is prepared locally; repository visibility and publication are still pending. See [distribution](DISTRIBUTION.md) for asset names, selected downloads, and access instructions. No license has been selected or granted by this preparation; settle the intended licensing and applicable third-party notices before public publication.

## Defaults chosen for this assignment

- Six tracks; one primary track per project.
- Comparable size means total parameters within ±10% of 146,496,016. All experts, learned tables, and training-only heads count; frozen base parameters count for adapters. Adapter tuning is eligible for E/F even with a small trainable fraction.
- Pretraining quality is scored using frozen full-vocabulary held-out next-token CE. Training CE is still plotted, but its sampled/composite forms cannot fairly rank different objectives.
- A uses the exact 3B budget. B/C target the 3B final CE and share a 3B cap. D declares one of those objectives and identifies the changed data condition.
- E starts from 50B; F defaults to continued pretraining from 50B, with scratch Italian reported separately.
- The rubric in `DELIVERABLES.md` is a suggested research-quality rubric. No due date, group size, mandatory number of full runs, or GPU-hour entitlement has been invented.

Adjust these choices consistently before distributing the handout. The size band, compute allowance, and target precision should be stable once projects begin.

## Before teaching begins

1. Announce deadlines, team size, GPU access/allocation, total experiment budget, submission storage, and grading weights.
2. Publish the source repository with the frozen validation/tokenizer reference included, and attach `miniple-3b.pt` and `miniple-50b.pt` to release `v1.0`. Keep those full `.pt` files and generated training data out of Git. Confirm `distribution.json` matches the repository and tag. A source checkout is sufficient for scratch training after environment setup and data reconstruction; students download only the reference needed by their track. Test the published downloads and verification from a clean checkout, including authenticated access if private. Students need network access, preparation time, and local space to reconstruct the training corpus.
3. Allocate a short environment check and a baseline run on the actual course software/hardware image. Historical timings exclude earlier setup; C requires a newly matched control and common timing instrumentation. Do not rank against the old 11-hour number alone.
4. Freeze an additional document-disjoint test set for final claims if feasible. The shipped 128-sequence monitor is public and already used for development; the rest of the shipped file is public too. No hidden final test set has been created here.
5. Decide the permitted external-data and synthetic-data conditions for D/E/F and how preprocessing/teacher costs enter the class budget. The docs require disclosure and separate conditions; no paid API usage is required.
6. Decide whether final multi-seed confirmation fits the allocation. The handout recommends it when feasible, not unlimited compute.

## Historical artifacts and reproducibility

The 3B checkpoint hash and the 50B checkpoint hash are fixed in `BASELINE.md` and `baselines/manifest.json`. Original metrics, plots, task scores, evaluation metadata, launch plans, and source manifests are retained. The original 3B source archive is also included for audit.

Release asset names differ from their installed names: `miniple-3b.pt` installs as `baselines/3b/final_l3_mix.pt`, and `miniple-50b.pt` as `baselines/50b/final_l3_mix.pt`. Upload the original full checkpoints unchanged. After download, run `python3 scripts/verify_artifacts.py` with `--checkpoints 3b`, `--checkpoints 50b`, or `--checkpoints all`. It checks the selected files and always verifies frozen evaluation assets; `--checkpoints none` is the appropriate default for scratch-only students.

The original 3B data directory linked slices of larger existing files. The student folder retains only their provenance/hashes, not the training tokens. Fresh preparation uses the same pinned sources and frozen tokenizer, but different training slices; students must run a matched unchanged control on the reconstructed corpus. The launcher checks regenerated validation against `baselines/validation/val_l2.bin`, keeping scoring consistent. Historical scores alone cannot isolate a method's effect across the changed training corpus. The reconstructed metadata cannot be used to resume the historical optimizer checkpoint; do not defeat fingerprint checks.

The current student trainer retains later resume/data-handling safeguards. Default budgets, preparation revisions, and paths are changed only in the student copy. The core model and optimizer implementations are preserved. Current-source hashes are separate from historical launch manifests. Reproduction means the documented recipe/data condition, not a promise of bitwise-identical floating-point training across software versions.

## What is and is not implemented

Included: training/data preparation, generation for the baseline architecture, fixed loss evaluator, pinned four-task base benchmark evaluator, plotting, optional release-checkpoint downloads, selected artifact checks, CPU correctness checks, student documents, and agent guidance.

Student work: architecture/optimizer/data innovations, target-checkpoint saving/timing instrumentation when needed, SFT/chat training and generation evaluation, Italian adaptation and benchmarks, and any custom-model adapters.

No new 3B or 50B training run is part of preparing this handout. See `VERIFICATION.md` for the checks actually performed on this release.

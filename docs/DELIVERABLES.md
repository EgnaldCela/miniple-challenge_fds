# Deliverables and assessment

Use the course's announced team size, deadline, GPU allocation, and submission channel. Choose one main track. Additional tracks are optional and must fit the same announced resources.

## Before the final run

Complete [the proposal template](../templates/PROPOSAL.md): hypothesis, control, treatment, metric, exact token/time/parameter budget, data/checkpoint lineage, correctness check, and stopping rule. Record changes to the plan and their reasons.

## Submit

1. **Reproducible code:** changed source or a clean patch, exact commands, environment versions, seeds, configuration, and a small correctness check appropriate to the change.
2. **Research report:** use [the report template](../templates/REPORT.md). Include the question, method, control, results, ablation, interpretation, limitations, and references. Explain what failed as well as what succeeded.
3. **Raw evidence:** all relevant `metrics.jsonl`, evaluation JSON, dataset/source manifests, the filled [result metadata](../templates/result.json), and an experiment ledger including failed pilots and total compute cost.
4. **Plots:** full-vocabulary validation CE versus consumed tokens; CE versus elapsed time for C; training objectives split by type/candidate count. Mark target, schedules, and data boundaries. Include underlying data. E/F additionally need adaptation and task-specific evaluation curves/tables.
5. **Model artifact:** the final/qualifying checkpoint with SHA256 and loading instructions. Adapter submissions include the exact base checkpoint hash, adapter files, configuration, and inference command. Use the course artifact storage for large files rather than committing them to Git.
6. **Brief demonstration:** explain the hypothesis, show one reproducible evaluation, and discuss one meaningful failure case. Generated samples use fixed, disclosed prompts and settings.

For A, submit the exact-budget final checkpoint. For B/C, submit the first qualifying scheduled checkpoint and its preceding evaluation; if the target was not reached, submit the best observed result with that status clearly stated. D includes source selection and mixture accounting. E includes chat serialization/masks and instruction evaluation. F includes the Italian corpus, tokenizer statistics, and English retention.

## Suggested assessment rubric

The instructor can adopt or revise these weights before release. Quality of research is not equivalent to a leaderboard rank.

| Dimension | Suggested weight | Evidence |
|---|---:|---|
| Experimental validity | 30% | Fair control, correct accounting, no leakage, meaningful metric |
| Implementation and correctness | 25% | Appropriate change, relevant checks, sound model/data handling |
| Analysis and ablation | 25% | Explanation supported by results, failure analysis, uncertainty |
| Reproducibility and communication | 20% | Complete artifacts, clear report, repeatable commands |

A negative result can score well when the experiment is controlled and informative. A large apparent improvement with altered validation, missing costs, or incompatible metrics is not persuasive evidence. Multiple seeds strengthen a claim when feasible; one seed must be labelled preliminary.

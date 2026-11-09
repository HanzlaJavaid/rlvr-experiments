# Experiment G: native-thinking process rewards

Independent local fork of Experiment F’s completed v3 implementation. G asks the policy to select an appropriate framework, apply it, and check its reasoning within its native thinking trace. It produces a plain, self-contained final answer. There are no required `<model>`, `<apply>`, `<check>`, or `<answer>` sections, and no reward for headings or tag presence.

G has not been trained or evaluated on GPUs. Local tests use synthetic tokenizers and judge responses; they do not call Jev or rent a host. The inherited GPU runtime needs a separately authorized G pilot.

## Preservation of F

F remains in [the original source package](../science-gspo-token-f/README.md), with its completed run, final adapter, evaluation, logs, CSVs, and graphs in [the F result archive](../experiment-f-v3-59219018/FINAL_RESULTS.md). `F_BASELINE_MANIFEST.json` records checksums of F’s source/config/test/documentation files before creating G. This source also matched F’s verified full-run source archive.

G uses package `science_gspo_g`, command `science-gspo-g`, experiment ID `G`, new prompt/rubric/data/completion versions, and distinct protocol hashes. Default paths are `runs_g`, `runs_g_pilots`, and `data/science_g_qwen35_v1`. F configurations, F prepared-data formats, and F checkpoints cannot be reused as G. G preparation recomputes prompt-length eligibility; F’s 171-update count is not assumed.

## Judge inputs

| Evaluator | Candidate content | Other inputs | Target Opus reference |
|---|---|---|---|
| Scientific correctness | Plain final answer only | Question and relevant context | Yes, explicitly fallible |
| Modeling | Entire native thinking trace | Question and relevant context | No |
| Applying | Same entire thinking trace | Question and relevant context | No |
| Checking | Same entire thinking trace | Question and relevant context | No |

Process states use an explicit allowlist; the target `reference_answer` is never copied into them. Prior conversation turns remain fallible context, including earlier assistant answers when needed to interpret a follow-up. Future turns and the current target answer never enter the policy prompt or process payload. The reference’s own thinking is stripped during preparation, as in F.

Modeling scores frameworks, definitions, assumptions, and conditions. Applying scores the reasoning actually performed. Checking scores substantive tests, limitations, alternatives, or repairs, rather than declarations of confidence. All three accept concise work appropriate to the task, without requiring named stages. These are whole-trace scores, not step-level annotations. Four independent evaluator calls retain per-component failure and retry handling. `show-prompts` prints policy instructions; exact reward instructions live in `src/science_gspo_g/judge.py` and each future run’s `rubric.json`.

## Output parsing and credit

The actual `</think>` token separates thinking from the plain final answer. A generated `<think>` opening is optional because the chat template normally supplies it. Reward boundaries use original token IDs, never a decode/re-encode approximation.

Each metric is standardized independently within a question’s four completions, requiring at least two available scores. Population standard deviation has floor0.10; each process z-score is clipped to ±2, with coefficient0.25/3:

```text
advantage[token] = z(scientific_correctness)
                 + is_thinking_token * (0.25 / 3)
                   * (clip(z(modeling)) + clip(z(applying)) + clip(z(checking)))
```

Correctness credit covers the full completion. All process bonuses cover the same thinking span, excluding generated opening/closing markers, final answer, EOS, and padding. A missing score is unavailable, never a fabricated scientific zero. A valid closed trace can retain process feedback without a complete final answer; a complete final answer can retain correctness feedback with an empty trace. Repeated/missing native boundaries, misplaced openings, and unexpected special-token corruption are not silently repaired. Token spans and availability are journaled.

Removing the four visible sections does not itself solve repeated native closing tokens seen in F. G’s pilot must inspect boundary validity, score availability, effective policy signal, and finite nonzero gradients before making claims about improvement.

## Retained settings

Model/data revisions, eight domains, split procedure, LoRA rank16/alpha32 on eligible linear projections, FP32 adapters over the frozen BF16 text backbone, one epoch, LR2e-6 constant after warmup, global batch64 (16 questions ×4 completions), GSPO sequence clipping, KL0.01, and Jev1.14.0 are inherited from F.

Sampling remains temperature1, top_p0.95, top_k20, min_p0, presence_penalty1.5, repetition_penalty1, and max_new_tokens8192. The v3 token_truncate inference correction with cap3 and pinned kernels remain unchanged. The pilot is12 updates in a separate directory. No automatic calibration, evaluation, publishing, cloud rental, or credential discovery is added.

## Local use

From this directory, in Python3.12 with the lightweight test dependencies:

```bash
python -m pip install -e '.[test]'
python -m science_gspo_g.cli inspect-config --config configs/experiment_g_4x5090.yaml
python -m science_gspo_g.cli show-prompts --subject physics
python -m science_gspo_g.cli launch --config configs/experiment_g_4x5090.yaml --resume none --dry-run
python -m science_gspo_g.cli pilot --config configs/pilot_g_4x5090.yaml --dry-run
python -m pytest -q
```

The GPU installer uses `SCIENCE_G_ENV` (default `/opt/science-g-env`). A future authorized run should use fresh base-model initialization and a new G directory, not F’s trained adapter. Later resumptions must match G’s exact source and protocol hashes. Model/data preparation and actual judge/GPU execution have not been performed for G.

See [CHANGES.md](docs/CHANGES.md) and [VALIDATION.json](docs/VALIDATION.json).

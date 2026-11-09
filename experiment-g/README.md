# Experiment G

Experiment G fine-tuned the Qwen3.5-4B text model for one epoch on 2,736 science questions. Run `science_G_full_qwen35_4b_4x5090_v1` completed all 171 optimizer updates on Vast instance `59881279`.

G rewarded scientific correctness using an Opus reference and scored modeling, application, and verification on the same native thinking trace. Those three judges received the question and context without the target reference.

## Find what you need

| What | Where |
| --- | --- |
| Trained LoRA and tokenizer | [Adapter weights](https://github.com/HanzlaJavaid/rlvr-experiments/releases/download/experiment-g/adapter_model.safetensors) and [adapter config and tokenizer](full-run/adapter/) |
| Run both models in Colab | [Colab notebook](colab/run_experiment_g_colab.ipynb) and [adapter ZIP to upload](https://github.com/HanzlaJavaid/rlvr-experiments/releases/download/experiment-g/experiment_g_colab_adapter.zip) |
| Four-question evaluation | [Report](full-run/evaluation/eval_results.md), [selected questions](full-run/evaluation/selected_questions.json), [all generations and scores](full-run/evaluation/), and [supplemental correctness](full-run/evaluation/supplemental_correctness.json) |
| Training plots | [20-step rolling means](plots/rewards_20_step_rolling.png) and [earlier 10-step bucket plot](plots/rewards_10_step_buckets.png) |
| Training data and metrics | [Per-step rewards](full-run/evaluation/training_rewards_by_step.json), [metrics journal](full-run/training/metrics.jsonl), and [data and protocol record](full-run/training/protocol.json) |
| Exact training settings and prompts | [Resolved config](full-run/training/resolved_config.json), [policy prompts](full-run/training/prompts.json), and [judge rubrics](full-run/training/rubric.json) |
| Logs and run diagnostics | [Host logs and exit markers](full-run/host/), [trainer state](full-run/training/checkpoints/trainer_state.json), and [per-GPU diagnostics](full-run/training/ranks/) |
| Implementation and reproduction | [Source, configs, and tests](source/science-gspo-token-g/), plus [launch and evaluation scripts](launch/) |
| Manual follow-up | [Base and trained responses to the sliding-window coding question](follow-up/sliding-window-base-vs-trained.txt) |
| Original download | [Full results ZIP](https://github.com/HanzlaJavaid/rlvr-experiments/releases/download/experiment-g/g_eval4_59881279_results.zip), [73-file manifest](full-run/manifest.json), and [ZIP checksums](SHA256SUMS) |

The source README describes the original pre-run plan. The completion record and evaluation files above document what actually ran.

## Evaluation snapshot

| Question | Base correctness | Trained correctness |
| --- | ---: | ---: |
| Physics | 0.690 | 0.820 |
| Chemistry | 0.457 | 0.450 |
| Biology | 0.603 | 0.677 |
| Math | 0.580 | 0.707 |

The four-question mean rose by 0.081. This table uses the supplemental final-answer scoring, which recovered three outputs with duplicate thinking delimiters. The strict parser yielded two directly comparable correctness pairs. This was a small evaluation with one generation per model per question.

The per-step journal covers a Jev outage during steps 76–83. Consult score availability when interpreting that part of the curves.

## External inputs

- Base weights come from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a).
- Training data came from [angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k](https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k/tree/f0330e0ca46469b3928adef18c2b55f9476d6bd3).

The downloaded run archive contains the final adapter and trainer state. Base weights, the raw training dataset, and optimizer-state checkpoints are external to this folder.

## Download the model files

The two ZIPs and standalone adapter weights are attached to the [Experiment G release](https://github.com/HanzlaJavaid/rlvr-experiments/releases/tag/experiment-g). For Colab, download the adapter ZIP and upload it when the notebook prompts you.

To reproduce the Desktop folder layout, place the results ZIP in `archives/`, the Colab adapter ZIP in `colab/`, and the standalone weights in `full-run/adapter/`, using the filenames from the release. Run `shasum -a 256 -c SHA256SUMS` from this directory to verify the three downloads.

"""Paired evaluation of the final Experiment G adapter on the four F dev questions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

ARMS = ("base", "checkpoint_171")
OUTCOME = (("scientific_correctness", 1.0),)
# Experiment F question IDs, subjects, and UTF-8 SHA-256 digests of the question and reference.
QUESTIONS = [('conv_4882_message_2',
  'physics',
  '50d4f1a40a8dfd5b265f14158354ed5e662f7d0316b8134ba2a46d2085772bd2',
  '84da9f64052e6fe6963f9a655e44c9eec9ac638343c255d4f9ed4d8667d8eb62'),
 ('conv_2500_message_4',
  'chemistry',
  '0ab28ce77469bcc685b838f1aa8b5e4510c5df8ef5af574d1243f27b8f5323b3',
  '0136c82200c4c8e273916dec8be96785915e482317c78d4a0e9473d3ae77accb'),
 ('conv_3371_message_2',
  'biology',
  'adb1080a0fafe9fbcdad35ea0e2f0c80d95a0546585e8c78c2d9d5a00e4b505c',
  'a194583ddd197e022bd3f26d9b4e7c4df8001760e1c51023e3f917e1096ef0a5'),
 ('conv_6474_message_4',
  'math',
  'cc20b51fd2944635915f1ac9bcbc01b59fca408f99f44e9bb20ac9685e3c97db',
  '3f6018580a59923400ef1f6c89795ce6e578221fc440e97080736f4475c3204b')]


def paired_summary(records):
    pairs={}
    for row in records:
        pair=pairs.setdefault(row['example_id'],{})
        if row['arm'] not in ARMS or row['arm'] in pair:raise ValueError('Duplicate or invalid evaluation arm')
        pair[row['arm']]=row
    if len(pairs)!=4 or any(set(pair)!=set(ARMS) for pair in pairs.values()):
        raise ValueError('Require exactly four complete base/trained pairs')
    output=[]
    for example_id,pair in pairs.items():
        base,trained=(pair[arm] for arm in ARMS)
        common=[(name,w) for name,w in OUTCOME if all(pair[arm]['scores'].get(name) is not None for arm in ARMS)]
        weight=sum(w for _,w in common)
        outcomes={arm:sum(pair[arm]['scores'][name]*w for name,w in common)/weight if weight else None for arm in ARMS}
        deltas={name:(trained['scores'][name]-value if value is not None and trained['scores'].get(name) is not None else None)
                for name,value in base['scores'].items()}
        output.append({'example_id':example_id,'category':base['category'],'base_status':base['status'],
                       'trained_status':trained['status'],'common_outcome_criteria':[name for name,_ in common],
                       'outcomes_on_common_criteria':outcomes,
                       'outcome_delta':outcomes[ARMS[1]]-outcomes[ARMS[0]] if weight else None,
                       'metric_deltas':deltas})
    deltas=[r['outcome_delta'] for r in output if r['outcome_delta'] is not None]
    metrics=sorted({name for row in output for name in row['metric_deltas']})
    return {'questions':4,'generations':8,'paired_questions_with_outcome_scores':len(deltas),
            'mean_paired_outcome_delta':sum(deltas)/len(deltas) if deltas else None,
            'numerically_higher':sum(d>1e-12 for d in deltas),'numerically_lower':sum(d< -1e-12 for d in deltas),
            'numerically_tied':sum(abs(d)<=1e-12 for d in deltas),
            'mean_paired_metric_deltas':{name:sum(v)/len(v) if (v:=[r['metric_deltas'][name] for r in output if r['metric_deltas'].get(name) is not None]) else None for name in metrics},
            'pairs':output,'limitations':['Four questions, one sample per model: exploratory only.',
            'Identical prompts/decoding and paired seeds; no reward calibration or extra training.',
            'The same training judge/rubrics are reused; these are not independent scientific gold labels.',
            'Missing metrics are unavailable, never synthetic zeros; deltas use common available criteria.']}


def generate_pair(index,row,assets,checkpoint,cfg,out_dir):
    out=Path(out_dir)
    if all((out/f'generation_{index}_{arm}.json').is_file() for arm in ARMS):return
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import PeftModel
    from science_gspo_g.parsing import parse_completion
    from science_gspo_g.io import atomic_json
    torch.set_num_threads(4);torch.cuda.set_device(index)
    tokenizer=AutoTokenizer.from_pretrained(assets['model_snapshot'],local_files_only=True)
    from science_gspo_g.model import load_text_policy
    base,audit=load_text_policy(assets['model_snapshot'],cfg,index)
    base.config.use_cache=True
    atomic_json(out/f'model_loader_{index}.json',audit)
    policy=PeftModel.from_pretrained(base,str(checkpoint),is_trainable=False)
    policy.eval();policy.set_adapter('default')
    prompt_ids=tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,return_dict=False,enable_thinking=True)
    assert len(prompt_ids)==row['prompt_tokens']
    assert len(prompt_ids)<=cfg['model']['max_prompt_tokens']
    inputs=torch.tensor([prompt_ids],device=f'cuda:{index}');attention=torch.ones_like(inputs)
    eos=policy.generation_config.eos_token_id
    eos_ids=[eos] if isinstance(eos,int) else list(eos)
    if tokenizer.eos_token_id not in eos_ids:eos_ids.append(tokenizer.eos_token_id)
    seed=123+index
    settings=cfg['generation']
    kwargs=dict(max_new_tokens=settings['max_new_tokens'],do_sample=True,temperature=settings['temperature'],
                top_p=settings['top_p'],top_k=settings['top_k'],use_cache=True,
                eos_token_id=eos_ids,pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    from transformers import LogitsProcessor,LogitsProcessorList
    class PresencePenalty(LogitsProcessor):
        def __call__(self,input_ids,scores):
            for batch in range(input_ids.shape[0]):
                seen=input_ids[batch,len(prompt_ids):].unique()
                scores[batch,seen]-=settings['presence_penalty']
            return scores
    kwargs.update(min_p=settings['min_p'],repetition_penalty=settings['repetition_penalty'],logits_processor=LogitsProcessorList([PresencePenalty()]))
    for arm in ARMS:
        target=out/f'generation_{index}_{arm}.json'
        if target.exists():continue
        set_seed(seed);torch.cuda.synchronize(index);start=time.monotonic()
        from contextlib import nullcontext
        with (policy.disable_adapter() if arm=='base' else nullcontext()),torch.inference_mode():
            generated=policy.generate(input_ids=inputs,attention_mask=attention,**kwargs)
        torch.cuda.synchronize(index)
        ids=generated[0,len(prompt_ids):].tolist()
        parsed=parse_completion(ids,tokenizer,settings['max_new_tokens'],eos_ids)
        atomic_json(target,{'arm':arm,'selection_index':index,'seed':seed,
                    'generation_seconds':time.monotonic()-start,'example_id':row['example_id'],'parsed':parsed})
        print(f'Generated question {index+1}/4, {arm}, tokens={len(ids)}, status={parsed["status"]}',flush=True)
    del policy,base;torch.cuda.empty_cache()



def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("generate", "score"))
    args = parser.parse_args()

    from science_gspo_g.io import atomic_json

    run = args.run_dir
    out = run / "paired_eval_4_final171"
    out.mkdir(exist_ok=True)
    if (out / "eval_completed.json").exists():
        raise RuntimeError("This paired evaluation is already complete")
    marker = json.loads((run / "training_completed.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    checkpoint = run / "final_adapter"
    assert marker["global_step"] == 171 and marker["epoch"] == 1
    assert marker["protocol_hash"] == protocol["protocol_hash"]
    cfg = json.loads((run / "resolved_config.json").read_text())

    if args.phase == "generate":
        from science_gspo_g.data import load_prepared

        assets = json.loads((run / "assets.json").read_text())
        splits = load_prepared(assets["data_dir"])
        lookup = {row["example_id"]: row for row in splits["dev"]}
        train_groups = {row["leakage_group"] for row in splits["train"]}
        train_ids = {row["example_id"] for row in splits["train"]}
        rows = []
        for example_id, category, question_hash, reference_hash in QUESTIONS:
            row = lookup[example_id]
            assert row["category"] == category
            assert row["example_id"] not in train_ids and row["leakage_group"] not in train_groups
            assert digest(row["user_question"]) == question_hash
            assert digest(row["reference_answer"]) == reference_hash
            rows.append(row)

        def adapter_digest(directory):
            with (directory / "adapter_model.safetensors").open("rb") as file:
                return hashlib.file_digest(file, "sha256").hexdigest()

        final_hash = adapter_digest(checkpoint)
        initial_hash = adapter_digest(run / "initial_adapter")
        assert final_hash != initial_hash
        atomic_json(out / "eval_protocol.json", {
            "format": "science-g-paired-dev-4-final171-v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evaluator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_protocol_hash": protocol["protocol_hash"],
            "model_revision": assets["model_revision"],
            "checkpoint": "final_adapter_step_171",
            "checkpoint_policy_sha256": final_hash,
            "initial_policy_sha256": initial_hash,
            "question_ids": [row["example_id"] for row in rows],
            "selection": "Exact four Experiment F questions and references, verified by SHA-256 and held out in G",
            "arms": list(ARMS),
            "subject_prompts_identical_between_arms": True,
            "reference_in_policy_prompt": False,
            "seeds": list(range(123, 127)),
            "generation": cfg["generation"],
            "inference_backend": "G Qwen3.5 text loader, Transformers SDPA, one GPU per question",
            "comparison_scope": "G prompts and Jev rubrics shared by both arms",
            "judge_blinding": "Arm labels excluded from judge input and request order shuffled",
        })
        atomic_json(out / "selected_questions.json", rows)
        print("Verified the same four held-out questions. Starting paired generation.", flush=True)
        context = mp.get_context("spawn")
        workers = [
            context.Process(target=generate_pair, args=(i, row, assets, checkpoint, cfg, out))
            for i, row in enumerate(rows)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if any(worker.exitcode != 0 for worker in workers):
            raise RuntimeError("A generation worker failed")
        print("All eight generations saved.", flush=True)
        return

    from science_gspo_g.judge import JevEngine, score_parsed_rows

    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("The scoring phase requires a TypeSafe credential")
    rows = json.loads((out / "selected_questions.json").read_text())
    generated = [
        json.loads((out / f"generation_{i}_{arm}.json").read_text())
        for i in range(4) for arm in ARMS
    ]
    random.Random(456).shuffle(generated)
    cfg["judge"]["concurrency_per_rank"] = cfg["judge"]["global_max_concurrency"]
    with JevEngine(cfg, out / "jev_cache.jsonl") as engine:
        records, stats = score_parsed_rows(
            [rows[item["selection_index"]] for item in generated],
            [item["parsed"] for item in generated],
            engine,
            protocol["protocol_hash"],
        )
    for record, item in zip(records, generated, strict=True):
        record.update({key: item[key] for key in ("arm", "selection_index", "seed", "generation_seconds")})
    records.sort(key=lambda record: (record["selection_index"], record["arm"] != "base"))
    summary = paired_summary(records)
    summary["judge_stats"] = stats
    atomic_json(out / "eval_records.json", records)
    atomic_json(out / "eval_summary.json", summary)
    lines = ["# Experiment G paired evaluation", "", json.dumps(summary, indent=2), ""]
    for row, base, trained in zip(rows, records[::2], records[1::2], strict=True):
        lines.extend([f"## {row['category']} · {row['example_id']}", "", row["user_question"], ""])
        for record in (base, trained):
            parsed = record["parsed"]
            lines.extend([
                f"### {record['arm']}", "", f"Status · {record['status']}", "",
                "Scores · " + json.dumps(record["scores"]), "",
                "Thinking trace", "", parsed["reasoning"] or "(unavailable)", "",
                "Final answer", "", parsed["final_answer"] or "(unavailable)", "",
            ])
        lines.extend(["### Dataset reference", "", row["reference_answer"], ""])
    (out / "eval_results.md").write_text("\n".join(lines), encoding="utf-8")
    if stats["failed_requests"]:
        raise RuntimeError(f"Jev failed {stats['failed_requests']} requests. Completed scores were saved for retry.")
    atomic_json(out / "eval_completed.json", {
        "success": True, "questions": 4, "generations": 8,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

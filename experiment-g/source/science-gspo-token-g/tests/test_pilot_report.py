import json
from pathlib import Path
import pytest

from science_gspo_g.io import atomic_json, append_jsonl
from science_gspo_g.pilot_report import summarize_pilot, write_pilot_report


def config(updates=6):
    return {"training":{"max_steps":updates}, "pilot":{"max_updates":updates,"warmup_updates":2,"min_measured_updates":3}}


def fixture_run(root,updates=6,checkpoint=True):
    cfg=config(updates)
    atomic_json(root/'resolved_config.json',cfg)
    atomic_json(root/'protocol.json',{'protocol_hash':'p','data_manifest':{'optimizer_steps_per_epoch':171}})
    atomic_json(root/'launcher_metrics.json',{'status':'completed','returncode':0,'wall_seconds':1000,'hourly_usd':2.0})
    atomic_json(root/'training_completed.json',{'protocol_hash':'p','run_scope':'bounded_updates','global_step':updates})
    for rank in range(4):
        directory=root/'ranks'/f'rank_{rank:05d}'
        atomic_json(directory/'first_rollout_signal.json',{'passed':True,
                    'effective_policy_signal_sequences':64,'nonzero_policy_signal_sequences':64})
        append_jsonl(directory/'update_timings.jsonl',[
            {'rank':rank,'global_step':step,'update_seconds':step*10+rank,
             'generated_tokens':100*step,'generated_completions':16,
             'stages_seconds':{'generation_and_sync':step+rank,'judging_and_rank_wait':1+rank},
             'peak_allocated_gib':20+rank,'peak_reserved_gib':25+rank}
            for step in range(1,updates+1)])
        append_jsonl(directory/'training_generations.jsonl',[
            {'global_step':step-1,'batch_id':f'b{step}','status':'complete','judge_cache_keys':['a','b','c','d']}
            for step in range(1,updates+1) for sample in range(16)])
        append_jsonl(directory/'token_advantages.jsonl',[
            {'global_step':step-1,'active':True,'advantage_segments':[{'start':0,'stop':10,'advantage':0.1}]}
            for step in range(1,updates+1) for sample in range(16)])
        append_jsonl(directory/'startup_events.jsonl',[
            {'rank':rank,'session_id':'one','event':event,'elapsed_seconds':seconds+rank}
            for event,seconds in [('worker_started',0),('trainer_vllm_ready',40),('first_update_started',50),('first_update_completed',60)]])
    append_jsonl(root/'metrics.jsonl',[{'global_step':step,'loss':-0.2,'grad_norm':0.4,'g/effective_policy_signal_fraction':1.0} for step in range(1,updates+1)])
    if checkpoint:
        path=root/'checkpoints'/f'checkpoint-{updates}'
        atomic_json(path/'checkpoint_complete.json',{'protocol_hash':'p','world_size':4,'global_step':updates})
        for name in ['trainer_state.json','optimizer.pt','scheduler.pt','adapter_config.json','adapter_model.safetensors',*[f'rng_state_{r}.pth' for r in range(4)]]:
            (path/name).write_text('{}')
    return cfg


def test_slowest_rank_summed_tokens_and_provisional_projection(tmp_path):
    fixture_run(tmp_path)
    report=summarize_pilot(tmp_path)
    assert report['success'] and report['status']=='completed'
    assert report['warmup_updates_excluded']==[1,2]
    assert report['measured_updates']==[3,4,5,6]
    assert report['update_seconds']=={'min':33,'median':48,'mean':48,'max':63}
    assert report['end_to_end_generated_tokens_per_second']==pytest.approx(7200/192)
    projection=report['projection']
    assert projection['training_seconds']==48*171
    assert projection['training_rental_usd']==pytest.approx(48*171/3600*2)
    assert projection['provisional'] and not projection['range_is_confidence_interval']
    assert report['stage_seconds']['generation_and_sync']['mean']==7.5
    assert report['gpu_peak_memory'][3]['peak_allocated_gib']==23
    assert report['completion_status_counts']=={'complete':384}
    assert report['advantage_diagnostics']['nonzero_policy_signal_records']==384
    assert report['startup_by_rank'][3]['seconds_to_first_update']==53


def test_partial_step_is_excluded_from_time_quality_and_projection_counts(tmp_path):
    fixture_run(tmp_path)
    rank=tmp_path/'ranks/rank_00003/update_timings.jsonl'
    rank.write_text('\n'.join(rank.read_text().splitlines()[:-1])+'\n')
    report=summarize_pilot(tmp_path)
    assert report['completed_updates_all_ranks']==[1,2,3,4,5]
    assert report['partial_rank_updates_excluded']==[6]
    assert report['completion_status_counts']=={'complete':320}
    assert not report['success'] and report['projection'] is not None


def test_missing_rank_cannot_certify_any_completed_update(tmp_path):
    fixture_run(tmp_path)
    (tmp_path/'ranks/rank_00002/update_timings.jsonl').unlink()
    report=summarize_pilot(tmp_path)
    assert report['missing_timing_ranks']==[2]
    assert report['completed_updates_all_ranks']==[]
    assert report['projection'] is None and not report['success']


def test_two_warmed_updates_withholds_projection(tmp_path):
    fixture_run(tmp_path,updates=4)
    report=summarize_pilot(tmp_path)
    assert report['measured_updates']==[3,4]
    assert report['projection'] is None


def test_duplicate_resumed_step_is_excluded_everywhere(tmp_path):
    fixture_run(tmp_path)
    p=tmp_path/'ranks/rank_00001/update_timings.jsonl'
    append_jsonl(p,[json.loads(p.read_text().splitlines()[2])])
    report=summarize_pilot(tmp_path)
    assert report['duplicate_steps_excluded']==[3]
    assert 3 not in report['completed_updates_all_ranks']
    assert not report['success'] and report['projection'] is None
    assert report['completion_status_counts']=={'complete':320}


def test_launcher_failed_before_training_uses_launch_config(tmp_path):
    atomic_json(tmp_path/'launch_config.json',config(12))
    atomic_json(tmp_path/'launcher_metrics.json',{'returncode':1,'status':'preflight_failed','wall_seconds':3})
    report=write_pilot_report(tmp_path)
    assert report['status']=='startup_failed_or_incomplete'
    assert report['planned_updates']==12 and not report['success']
    assert report['projection'] is None
    assert (tmp_path/'pilot_report.json').is_file()
    assert 'projection withheld' in (tmp_path/'pilot_report.md').read_text()


def test_first_rollout_abort_is_visible_without_optimizer_updates(tmp_path):
    atomic_json(tmp_path/'launch_config.json',config(12))
    atomic_json(tmp_path/'launcher_metrics.json',{'returncode':1,'status':'training_failed'})
    for rank in range(4):
        atomic_json(tmp_path/'ranks'/f'rank_{rank:05d}'/'first_rollout_signal.json',
                    {'passed':False,'reason':'no_complete_scaffold','rank':rank,
                     'completion_records':64,'complete_completions':0,'nonzero_policy_signal_sequences':0})
    report=write_pilot_report(tmp_path)
    assert len(report['first_rollout_signal_checks'])==4
    assert report['completed_updates_all_ranks']==[]
    assert not report['success'] and report['projection'] is None
    text=(tmp_path/'pilot_report.md').read_text()
    assert 'stopped before backward: `no_complete_scaffold`' in text
    assert '0/64 complete thinking/final outputs' in text


def test_incomplete_checkpoint_blocks_success_even_when_updates_finished(tmp_path):
    fixture_run(tmp_path)
    (tmp_path/'checkpoints/checkpoint-6/rng_state_2.pth').unlink()
    report=summarize_pilot(tmp_path)
    assert report['status']=='completed_with_failed_checks'
    assert not report['success'] and not report['checks']['final_resumable_checkpoint_verified']


def test_full_epoch_marker_is_not_pilot_success(tmp_path):
    fixture_run(tmp_path)
    atomic_json(tmp_path/'training_completed.json',{'protocol_hash':'p','run_scope':'full_epoch','global_step':6})
    assert not summarize_pilot(tmp_path)['success']


def test_no_policy_signal_does_not_certify_startup_training_success(tmp_path):
    fixture_run(tmp_path)
    for path in tmp_path.glob('ranks/*/token_advantages.jsonl'):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:row['advantage_segments'][0]['advantage']=0
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report=summarize_pilot(tmp_path)
    assert report['checks']['all_planned_updates_completed']
    assert not report['checks']['nonzero_policy_advantages_observed'] and not report['success']
    assert report['projection'] is None
    assert 'nonzero_policy_advantages_observed_not_verified' in report['projection_withheld_reasons']


def test_nonfinite_loss_is_reported_without_emitting_invalid_json(tmp_path):
    fixture_run(tmp_path)
    with (tmp_path/'metrics.jsonl').open('a') as f:f.write('{"global_step":6,"loss":NaN}\n')
    report=write_pilot_report(tmp_path)
    assert report['loss_diagnostics']['nonfinite_observations']==1
    assert not report['success']
    json.dumps(report,allow_nan=False)


def test_unknown_epoch_never_assumes_old_171_update_count(tmp_path):
    fixture_run(tmp_path)
    atomic_json(tmp_path/'protocol.json',{'protocol_hash':'p','data_manifest':{}})
    assert summarize_pilot(tmp_path)['projection'] is None


def test_partial_tail_not_modified_and_corruption_prevents_success(tmp_path):
    fixture_run(tmp_path)
    path=tmp_path/'ranks/rank_00000/update_timings.jsonl'
    with path.open('a') as f:f.write('{"unfinished":')
    before=path.read_bytes()
    report=summarize_pilot(tmp_path)
    assert report['success'] and path.read_bytes()==before
    path.write_text('not json\n')
    report=summarize_pilot(tmp_path)
    assert report['journal_errors'] and not report['success']


def test_multiple_generation_attempts_are_not_counted_twice(tmp_path):
    fixture_run(tmp_path)
    append_jsonl(tmp_path/'ranks/rank_00000/training_generations.jsonl',[
        {'global_step':2,'batch_id':'another','status':'complete','judge_cache_keys':['a','b','c','d']}])
    report=summarize_pilot(tmp_path)
    assert report['ambiguous_generation_steps_excluded_from_quality']==[3]
    assert not report['success']
    assert report['completion_status_counts']=={'complete':368}


def test_no_files_is_not_started(tmp_path):
    report=summarize_pilot(tmp_path)
    assert report['status']=='not_started' and report['projection'] is None


def test_missing_or_zero_gradient_blocks_success(tmp_path):
    fixture_run(tmp_path)
    p=tmp_path/'metrics.jsonl'
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    for row in rows:row['grad_norm']=0
    p.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report=summarize_pilot(tmp_path)
    assert not report['checks']['nonzero_gradient_observed'] and not report['success']


def test_checkpoint_marker_must_match_final_step(tmp_path):
    fixture_run(tmp_path)
    atomic_json(tmp_path/'checkpoints/checkpoint-6/checkpoint_complete.json',
                {'protocol_hash':'p','world_size':4,'global_step':5})
    assert not summarize_pilot(tmp_path)['checks']['final_resumable_checkpoint_verified']


@pytest.mark.parametrize('filename', ['training_generations.jsonl','token_advantages.jsonl'])
def test_incomplete_rank_audit_blocks_readiness_and_projection(tmp_path,filename):
    fixture_run(tmp_path)
    path=tmp_path/'ranks/rank_00002'/filename
    rows=path.read_text().splitlines()
    path.write_text('\n'.join(rows[:-1])+'\n')
    report=summarize_pilot(tmp_path)
    assert not report['success'] and report['projection'] is None
    assert not report['checks']['complete_generation_and_advantage_audits']
    failed=[row for row in report['rank_update_audit_counts'] if not row['complete']]
    assert len(failed)==1 and failed[0]['rank']==2 and failed[0]['step']==6


def test_later_update_does_not_prove_first_update(tmp_path):
    fixture_run(tmp_path)
    for path in tmp_path.glob('ranks/*/update_timings.jsonl'):
        path.write_text('\n'.join(path.read_text().splitlines()[1:])+'\n')
    report=summarize_pilot(tmp_path)
    assert report['completed_updates_all_ranks']==[2,3,4,5,6]
    assert not report['checks']['first_update_completed_on_all_ranks']


def test_zero_learning_all_invalid_withholds_eta(tmp_path):
    fixture_run(tmp_path)
    for path in tmp_path.glob('ranks/*/training_generations.jsonl'):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:row.update(status='malformed',judge_cache_keys=[None]*4)
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report=summarize_pilot(tmp_path)
    assert report['projection'] is None
    assert 'jev_scoring_observed_not_verified' in report['projection_withheld_reasons']


def test_markdown_surfaces_preflight_and_startup_failure_details(tmp_path):
    atomic_json(tmp_path/'launch_config.json',config(12))
    atomic_json(tmp_path/'launcher_metrics.json',{'status':'timed_out','phase':'training',
                'returncode':-15,'timeout_kind':'startup','wall_seconds':900})
    atomic_json(tmp_path/'preflight.json',{'status':'failed','stage':'judge_connectivity',
                'http_status_code':451,'error_type':'APIStatusError',
                'error':'Judge connectivity check failed (HTTP 451)'})
    append_jsonl(tmp_path/'ranks/rank_00002/startup_events.jsonl',[
        {'session_id':'one','event':'model_loaded','elapsed_seconds':500}])
    report=write_pilot_report(tmp_path)
    text=(tmp_path/'pilot_report.md').read_text()
    for value in ['judge_connectivity','451','APIStatusError','model_loaded','startup','training']:
        assert value in text
    assert report['preflight']['http_status_code']==451
    assert report['startup_by_rank'][2]['last_event']=='model_loaded'


@pytest.mark.parametrize('metric,check,diagnostic', [
    ('loss','finite_loss_observed','loss_diagnostics'),
    ('grad_norm','nonzero_gradient_observed','gradient_diagnostics'),
])
def test_every_completed_step_needs_finite_loss_and_gradient(tmp_path,metric,check,diagnostic):
    fixture_run(tmp_path)
    path=tmp_path/'metrics.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    del rows[-1][metric]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report=summarize_pilot(tmp_path)
    assert not report['checks'][check] and not report['success']
    assert report['projection'] is None
    assert report[diagnostic]['missing_steps']==[6]


@pytest.mark.parametrize('metric,check,diagnostic', [
    ('loss','finite_loss_observed','loss_diagnostics'),
    ('grad_norm','nonzero_gradient_observed','gradient_diagnostics'),
])
def test_sanitized_nonfinite_flag_cannot_be_hidden_by_other_finite_records(tmp_path,metric,check,diagnostic):
    fixture_run(tmp_path)
    append_jsonl(tmp_path/'metrics.jsonl',[{'global_step':6,'nonfinite_metric_keys':[metric]}])
    report=summarize_pilot(tmp_path)
    assert not report['checks'][check] and not report['success']
    assert report['projection'] is None
    assert report[diagnostic]['nonfinite_observations']==1


def test_partial_reward_coverage_is_reported_as_scoring_not_missing_format(tmp_path):
    fixture_run(tmp_path)
    for path in tmp_path.glob('ranks/*/training_generations.jsonl'):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.update(status='malformed',reward_availability={'correctness':True,'check':False},
                       judge_failures=[{'error_type':'TimeoutError'}])
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report=summarize_pilot(tmp_path)
    assert report['success']
    assert report['judge_scored_completions']==384
    assert report['judge_scored_complete_answers']==0
    assert report['reward_availability_counts']=={'correctness':384}
    assert report['reward_unavailability_counts']=={'check':384}
    assert report['judge_failure_counts']=={'TimeoutError':384}


def test_keys_from_failed_requests_are_not_treated_as_scores(tmp_path):
    fixture_run(tmp_path)
    for path in tmp_path.glob('ranks/*/training_generations.jsonl'):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:row['reward_availability']={'correctness':False,'check':False}
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report=summarize_pilot(tmp_path)
    assert not report['checks']['jev_scoring_observed']
    assert not report['success'] and report['projection'] is None


def test_nonzero_raw_advantages_do_not_certify_zero_effective_policy_signal(tmp_path):
    fixture_run(tmp_path)
    metrics=tmp_path/'metrics.jsonl'
    rows=[json.loads(line) for line in metrics.read_text().splitlines()]
    for row in rows:row['g/effective_policy_signal_fraction']=0.0
    metrics.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for rank in range(4):
        atomic_json(tmp_path/'ranks'/f'rank_{rank:05d}'/'first_rollout_signal.json',
                    {'passed':False,'nonzero_policy_signal_sequences':64,
                     'effective_policy_signal_sequences':0,'reason':'no_effective_policy_signal'})
    report=summarize_pilot(tmp_path)
    assert report['checks']['nonzero_policy_advantages_observed']
    assert not report['checks']['first_rollout_effective_signal_verified']
    assert not report['checks']['effective_policy_signal_observed']
    assert not report['success']

"""CPU contracts using exact AST-extracted TRL 1.17.0 reward/loss hooks.

Generation/model-forward are lightweight stand-ins; no model is loaded and no
network service is called. This exercises the real reward callback dispatch and
compute_loss routing without installing a CUDA-only vLLM runtime on this Mac.
"""
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import logging
import json
import sys

import pytest
import torch

from science_gspo_g.trainer import make_trainer_class, _compact_segments, PilotSignalError, effective_policy_signal
from test_parsing import ByteTokenizer, TEXT


namespace = dict(torch=torch, nn=torch.nn, profiling_context=lambda *a: nullcontext(),
                 is_async_callable=lambda f: False, gather=lambda x: x,
                 logger=logging.getLogger('offline-upstream'))
exec(compile((Path(__file__).parent/'fixtures/trl_1_17_hooks.txt').read_text(),
             'trl_1_17_hooks.txt', 'exec'), namespace)


class UpstreamContractBase:
    _calculate_rewards = namespace['_calculate_rewards']
    compute_loss = namespace['compute_loss']

    def __init__(self, **kwargs):
        self.args=SimpleNamespace(use_bias_correction_kl=False,delta=None)
        self.loss_type='grpo';self.importance_sampling_level='sequence'
        self.accelerator=SimpleNamespace(device=torch.device('cpu'),process_index=0,
            reduce=lambda x,reduction:x)
        self.state=SimpleNamespace(global_step=0)
        self.model=SimpleNamespace(training=True)
        self.processing_class=ByteTokenizer()
        self.num_generations=4;self.environments=None
        self.epsilon_low=.0003;self.epsilon_high=.0004;self.beta=0.
        self.use_vllm=True;self.vllm_importance_sampling_correction=True
        self.current_gradient_accumulation_steps=16
        self.reward_funcs=[]
        for col in range(4):
            def callback(*,completion_ids,example_id,col=col,**kw):
                assert example_id==['q']*4
                return [1.,.6,.4,0.] if col==0 else [.9,.3,.7,.1]
            self.reward_funcs.append(callback)
        self.reward_processing_classes=[None]*4
        self.reward_func_names=[f'reward{i}' for i in range(4)]
        self._log_completion_extra=lambda *a:None
        self._log_metric=lambda *a:None
        self._metrics=defaultdict(lambda:defaultdict(list))
        self.calls=[]

    def _generate_and_score_completions(self,inputs):
        ids=[self.processing_class.encode(TEXT)+[9001] for _ in inputs]
        self._calculate_rewards(inputs,['prompt']*4,['completion']*4,ids)
        c=torch.tensor(ids)
        self._metrics['train']['reward'].append(123.)
        self._metrics['train']['reward_std'].append(123.)
        return dict(completion_ids=c,completion_mask=torch.ones_like(c),
                    advantages=torch.zeros(4),prompt_ids=torch.ones(4,2,dtype=torch.long),
                    prompt_mask=torch.ones(4,2),old_per_token_logps=torch.full(c.shape,-1.),
                    importance_sampling_ratio=torch.ones(c.shape),
                    sampling_mask=torch.stack((c,c+1),dim=-1))

    def _get_per_token_logps_and_entropies(self,model,input_ids,attention_mask,logits_to_keep,**kw):
        self.calls.append(kw)
        self.test_logps=torch.full((4,logits_to_keep),-1.,requires_grad=True)
        return self.test_logps,torch.ones_like(self.test_logps),None


def trainer(tmp_path, *, pilot=False):
    cls=make_trainer_class(UpstreamContractBase)
    return cls(experiment_config={'pilot':{'enabled':pilot},'generation':{'max_new_tokens':8192},'training':{
        'process_weight':.25,'advantage_std_floor':.1,'process_z_clip':2}},
        advantage_audit_path=tmp_path/'advantages.jsonl',eos_token_ids=[9001])


def zero_rewards(tr):
    tr.reward_funcs = [lambda *, completion_ids, **kw: [0.0]*len(completion_ids) for _ in range(4)]


def test_pilot_rejects_all_malformed_first_rollout_before_backward(tmp_path, monkeypatch):
    tr = trainer(tmp_path, pilot=True)
    tr.reward_funcs=[lambda **kw:[None]*4 for _ in range(4)]
    monkeypatch.setattr(sys.modules[__name__], 'TEXT', 'An answer without the native thinking closing boundary.')
    with pytest.raises(PilotSignalError, match='no_available_rewards'):
        tr._generate_and_score_completions([{'example_id':'q'}]*4)
    diagnostic = json.loads((tmp_path/'first_rollout_signal.json').read_text())
    assert diagnostic['passed'] is False and diagnostic['checked_before_backward'] is True
    assert diagnostic['complete_completions'] == 0
    assert diagnostic['nonzero_policy_signal_sequences'] == 0
    assert diagnostic['completion_status_counts'] == {'malformed':4}
    assert len((tmp_path/'advantages.jsonl').read_text().splitlines()) == 4
    assert tr.calls == []


def test_pilot_rejects_complete_but_zero_signal_first_rollout(tmp_path):
    tr = trainer(tmp_path, pilot=True)
    zero_rewards(tr)
    with pytest.raises(PilotSignalError, match='no_nonzero_policy_advantages'):
        tr._generate_and_score_completions([{'example_id':'q'}]*4)
    diagnostic = json.loads((tmp_path/'first_rollout_signal.json').read_text())
    assert diagnostic['complete_completions'] == diagnostic['scored_complete_completions'] == 4
    assert diagnostic['nonzero_policy_signal_sequences'] == 0
    assert tr.calls == []


def test_pilot_records_first_rollout_success_without_gating_later_batches(tmp_path):
    tr = trainer(tmp_path, pilot=True)
    out = tr._generate_and_score_completions([{'example_id':'q'}]*4)
    diagnostic = json.loads((tmp_path/'first_rollout_signal.json').read_text())
    assert diagnostic['passed'] is True and diagnostic['reason'] is None
    assert diagnostic['nonzero_policy_signal_sequences'] == 4
    assert out['advantages'].abs().sum() > 0
    zero_rewards(tr)
    later = tr._generate_and_score_completions([{'example_id':'q'}]*4)
    assert later['advantages'].abs().sum() == 0
    assert json.loads((tmp_path/'first_rollout_signal.json').read_text()) == diagnostic


def test_full_run_does_not_apply_pilot_first_rollout_gate(tmp_path):
    tr = trainer(tmp_path)
    zero_rewards(tr)
    output = tr._generate_and_score_completions([{'example_id':'q'}]*4)
    assert output['advantages'].abs().sum() == 0
    assert not (tmp_path/'first_rollout_signal.json').exists()


def test_pilot_signal_decision_uses_global_rewards_even_for_zero_signal_local_rank(tmp_path, monkeypatch):
    global_scores = torch.zeros(8,4)
    global_scores[4:, :1] = torch.tensor([1.0,0.6,0.4,0.0])[:,None]
    monkeypatch.setitem(namespace, 'gather', lambda _local: global_scores)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    def gather_metadata(parts, local):
        parts[:] = [local, local]
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather_metadata)
    for rank in range(2):
        path = tmp_path/f'rank_{rank}'
        tr = trainer(path, pilot=True)
        tr.accelerator.process_index = rank
        tr.accelerator.reduce=lambda counts,reduction:torch.tensor([4,4,4],device=counts.device)
        zero_rewards(tr)
        result = tr._generate_and_score_completions([{'example_id':'q'}]*4)
        diagnostic = json.loads((path/'first_rollout_signal.json').read_text())
        assert diagnostic['passed'] is True
        assert diagnostic['completion_records'] == diagnostic['complete_completions'] == 8
        assert diagnostic['nonzero_policy_signal_sequences'] == 4
        assert bool(result['advantages'].abs().sum() > 0) is bool(rank)


def test_pinned_reward_hook_to_token_advantages_and_pinned_loss_routing(tmp_path):
    tr=trainer(tmp_path)
    out=tr._generate_and_score_completions([{'example_id':'q'}]*4)
    assert out['advantages'].shape==out['completion_ids'].shape
    audit = json.loads((tmp_path/'advantages.jsonl').read_text().splitlines()[0])
    trace_start,trace_stop=audit['reasoning_token_span']
    assert out['advantages'][0,trace_start] != out['advantages'][0,-1]
    assert torch.all(out['advantages'][0,:trace_start] == out['advantages'][0,-1])
    assert torch.all(out['advantages'][0,trace_stop:] == out['advantages'][0,-1])
    assert audit['process_token_spans'] == [[[trace_start,trace_stop]]]*3
    assert tr._metrics['train']['g/effective_policy_signal_fraction'][-1] == 1.0
    assert out['advantages'][3,0]<0
    assert tr._metrics['train']['reward'][-1]<1
    loss=tr.compute_loss(tr.model,out)
    assert torch.isfinite(loss)
    loss.backward()
    assert tr.test_logps.grad[0].sum()<0
    assert tr.test_logps.grad[3].sum()>0
    assert tr.calls[0]['sampling_mask'] is out['sampling_mask']
    assert (tmp_path/'advantages.jsonl').read_text().count('\n')==4


def test_extra_tokens_after_eos_fail_alignment(tmp_path):
    tr=trainer(tmp_path)
    ids=[tr.processing_class.encode(TEXT)+[9001,9005] for _ in range(4)]
    with pytest.raises(RuntimeError,match='after EOS'):
        tr._calculate_rewards([{'example_id':'q'}]*4,['p']*4,['c']*4,ids)


def test_missing_inference_correction_is_fatal(tmp_path):
    tr=trainer(tmp_path)
    out=tr._generate_and_score_completions([{'example_id':'q'}]*4)
    del out['importance_sampling_ratio']
    with pytest.raises(RuntimeError,match='correction'):tr.compute_loss(tr.model,out)


def test_compact_advantages_exactly_reconstruct():
    values=[.1,.1,-.4,-.4,0,0]
    segments=_compact_segments(values,len(values))
    recovered=[None]*len(values)
    for item in segments: recovered[item['start']:item['stop']]=[item['advantage']]*(item['stop']-item['start'])
    assert recovered==values


def test_pinned_reward_hook_preserves_partial_nan_and_nonzero_gradients(tmp_path):
    tr=trainer(tmp_path)
    tr.reward_funcs[2]=lambda **kw:[None]*4
    tr.reward_funcs[0]=lambda **kw:[None,0.6,0.4,0.0]
    out=tr._generate_and_score_completions([{'example_id':'q'}]*4)
    assert torch.isfinite(out['advantages']).all()
    assert out['advantages'].abs().sum()>0
    loss=tr.compute_loss(tr.model,out)
    loss.backward()
    assert torch.isfinite(tr.test_logps.grad).all()
    assert tr.test_logps.grad.abs().sum()>0
    records=[json.loads(line) for line in (tmp_path/'advantages.jsonl').read_text().splitlines()]
    assert records[0]['reward_availability']==[False,True,False,True]


@pytest.mark.parametrize('kind', ['zero_correction', 'negligible_correction', 'singleton_support'])
def test_first_rollout_rejects_effectively_zero_policy_signal(tmp_path, monkeypatch, kind):
    original=UpstreamContractBase._generate_and_score_completions
    def masked(self,inputs):
        output=original(self,inputs)
        if kind=='zero_correction':
            output['importance_sampling_ratio'].zero_()
        elif kind=='negligible_correction':
            output['importance_sampling_ratio'].fill_(1e-10)
        else:
            output['sampling_mask']=output['completion_ids'].unsqueeze(-1)
        return output
    monkeypatch.setattr(UpstreamContractBase,'_generate_and_score_completions',masked)
    tr=trainer(tmp_path,pilot=True)
    with pytest.raises(PilotSignalError,match='no_effective_policy_signal'):
        tr._generate_and_score_completions([{'example_id':'q'}]*4)
    diagnostic=json.loads((tmp_path/'first_rollout_signal.json').read_text())
    assert diagnostic['nonzero_policy_signal_sequences']==4
    assert diagnostic['effective_policy_signal_sequences']==0
    assert diagnostic['corrected_policy_signal_sequences']==(4 if kind=='singleton_support' else 0)
    assert tr.calls==[]


def test_effective_policy_signal_masks_one_row_without_erasing_other_rows(tmp_path):
    tr=trainer(tmp_path)
    out=tr._generate_and_score_completions([{'example_id':'q'}]*4)
    out['importance_sampling_ratio'][0].zero_()
    assert effective_policy_signal(out).tolist()==[4,3,3]
    out['sampling_mask'][1,:,1]=-1
    assert effective_policy_signal(out).tolist()==[4,3,2]


@pytest.mark.parametrize('kind', ['nan', 'negative', 'shape', 'support_shape', 'support_missing', 'support_float'])
def test_effective_policy_signal_refuses_invalid_correction_or_support(tmp_path,kind):
    tr=trainer(tmp_path)
    out=tr._generate_and_score_completions([{'example_id':'q'}]*4)
    if kind=='nan':out['importance_sampling_ratio'][0,0]=float('nan')
    elif kind=='negative':out['importance_sampling_ratio'][0,0]=-1
    elif kind=='shape':out['importance_sampling_ratio']=torch.ones(4)
    elif kind=='support_shape':out['sampling_mask']=torch.ones_like(out['completion_ids'])
    elif kind=='support_missing':out['sampling_mask'][0,0]=-1
    else:out['sampling_mask']=out['sampling_mask'].float()
    with pytest.raises(ValueError):effective_policy_signal(out)


def test_trainer_routes_process_only_rewards_to_closed_trace_when_final_missing(tmp_path,monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], 'TEXT', '<think>Reason about μ, apply a law, and check units.</think>')
    tr=trainer(tmp_path)
    tr.reward_funcs[0]=lambda **kw:[None]*4
    out=tr._generate_and_score_completions([{'example_id':'q'}]*4)
    records=[json.loads(line) for line in (tmp_path/'advantages.jsonl').read_text().splitlines()]
    for i,record in enumerate(records):
        start,stop=record['reasoning_token_span']
        assert start==1  # Optional generated opening token excluded.
        assert record['status']=='empty_final'
        assert record['outcome_reward'] is None
        assert torch.all(out['advantages'][i,:start]==0)
        assert torch.all(out['advantages'][i,stop:]==0)
        assert out['advantages'][i,start:stop].abs().sum()>0
    loss=tr.compute_loss(tr.model,out)
    loss.backward()
    assert torch.isfinite(tr.test_logps.grad).all()
    assert tr.test_logps.grad.abs().sum()>0
    for i,record in enumerate(records):
        start,stop=record['reasoning_token_span']
        assert torch.all(tr.test_logps.grad[i,:start]==0)
        assert torch.all(tr.test_logps.grad[i,stop:]==0)


def test_trainer_keeps_final_accuracy_feedback_without_reasoning(tmp_path,monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], 'TEXT', '</think>The acceleration is 3 m/s².')
    tr=trainer(tmp_path)
    for stage in range(1,4):tr.reward_funcs[stage]=lambda **kw:[None]*4
    out=tr._generate_and_score_completions([{'example_id':'q'}]*4)
    records=[json.loads(line) for line in (tmp_path/'advantages.jsonl').read_text().splitlines()]
    for i,record in enumerate(records):
        assert record['reasoning_token_span'] is None
        assert record['reward_availability']==[True,False,False,False]
        assert torch.all(out['advantages'][i]==out['advantages'][i,0])
    assert out['advantages'].abs().sum()>0


def test_process_only_batch_omits_primary_reward_without_fabricating_zero(tmp_path):
    tr=trainer(tmp_path)
    tr.reward_funcs[0]=lambda **kw:[None]*4
    out=tr._generate_and_score_completions([{'example_id':'q'}]*4)
    metrics=tr._metrics['train']
    assert out['advantages'].abs().sum()>0
    assert 'reward' not in metrics
    assert 'reward_std' not in metrics
    assert 'g/outcome_reward' not in metrics
    assert metrics['g/scientific_correctness_availability_fraction']==[0.0]


def test_process_only_batch_preserves_previous_primary_metrics_in_logging_window(tmp_path):
    tr=trainer(tmp_path)
    tr._generate_and_score_completions([{'example_id':'q'}]*4)
    before={name:list(tr._metrics['train'][name]) for name in ('reward','reward_std','g/outcome_reward')}
    tr.reward_funcs[0]=lambda **kw:[None]*4
    tr._generate_and_score_completions([{'example_id':'q'}]*4)
    for name,values in before.items():
        assert tr._metrics['train'][name]==values
    assert tr._metrics['train']['g/scientific_correctness_availability_fraction']==[1.0,0.0]


def test_singleton_accuracy_is_reported_without_fabricating_relative_advantage(tmp_path):
    tr=trainer(tmp_path)
    tr.reward_funcs[0]=lambda **kw:[None,0.75,None,None]
    tr._generate_and_score_completions([{'example_id':'q'}]*4)
    metrics=tr._metrics['train']
    assert metrics['reward']==[.75]
    assert metrics['reward_std']==[0.0]
    assert metrics['g/outcome_reward']==[.75]
    assert metrics['g/scientific_correctness_availability_fraction']==[.25]
    records=[json.loads(line) for line in (tmp_path/'advantages.jsonl').read_text().splitlines()]
    assert all(record['outcome_advantage']==0 for record in records)
    assert all(record['outcome_reward'] is None for record in records)


def test_logging_keeps_previous_entries_if_upstream_does_not_append_primary_metric(tmp_path,monkeypatch):
    original=UpstreamContractBase._generate_and_score_completions
    def without_primary(self,inputs):
        output=original(self,inputs)
        self._metrics['train']['reward'].pop()
        self._metrics['train']['reward_std'].pop()
        return output
    monkeypatch.setattr(UpstreamContractBase,'_generate_and_score_completions',without_primary)
    tr=trainer(tmp_path)
    tr._metrics['train']['reward']=[.4]
    tr._metrics['train']['reward_std']=[.2]
    tr.reward_funcs[0]=lambda **kw:[None]*4
    tr._generate_and_score_completions([{'example_id':'q'}]*4)
    assert tr._metrics['train']['reward']==[.4]
    assert tr._metrics['train']['reward_std']==[.2]

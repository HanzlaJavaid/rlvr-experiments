"""Offline readiness contracts: no CUDA device or Jev service is contacted."""
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from science_gspo_g.io import read_jsonl
from science_gspo_g.preflight import (
    StartupAudit, check_bf16_device, check_collective,
    check_judge_connectivity, run_preflight, main,
)


class FakeCuda:
    def __init__(self, count=4, bf16=True):
        self.count, self.bf16 = count, bf16
        self.selected = []
    def device(self, index):
        self.selected.append(index)
        return nullcontext()
    def is_available(self): return True
    def device_count(self): return self.count
    def is_bf16_supported(self): return self.bf16
    def get_device_properties(self, index):
        return SimpleNamespace(name=f'fixture GPU {index}', total_memory=32*1024**3)
    def get_device_capability(self, index): return (12, 0)
    def synchronize(self): pass


def fixture_torch(count=4, bf16=True):
    return SimpleNamespace(cuda=FakeCuda(count, bf16), version=SimpleNamespace(cuda='fixture'),
        bfloat16=torch.bfloat16, float32=torch.float32,
        ones=lambda shape, *, device, dtype: torch.ones(shape, dtype=dtype))


class Client:
    def __init__(self, model='jev-1.14.0', value=1.0, error=None):
        self.model, self.value, self.error = model, value, error
        self.calls, self.closed = [], False
    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        if self.error: raise self.error
        return {'model': self.model, 'answers': {'startup_check': {'type': 'score', 'score': self.value}}}
    def close(self): self.closed = True


def test_startup_event_records_rank_session_and_elapsed(tmp_path):
    ticks = iter([10.0, 12.0, 15.0])
    audit = StartupAudit(tmp_path, 2, enabled=True, session_id='s', clock=lambda: next(ticks))
    assert audit.event('assets_ready', training_rows=32) == 2.0
    assert audit.event('trainer_vllm_ready') == 5.0
    rows = read_jsonl(tmp_path/'ranks/rank_00002/startup_events.jsonl')
    assert [r['event'] for r in rows] == ['assets_ready', 'trainer_vllm_ready']
    assert all(r['rank']==2 and r['session_id']=='s' and 'utc' in r for r in rows)


def test_disabled_startup_audit_has_no_filesystem_effect(tmp_path):
    audit = StartupAudit(tmp_path/'absent', 0, enabled=False)
    audit.event('worker_started')
    assert not (tmp_path/'absent').exists()


def test_startup_rejects_nonmonotonic_clock(tmp_path):
    ticks = iter([10., 9.])
    audit=StartupAudit(tmp_path,0,enabled=True,clock=lambda:next(ticks))
    with pytest.raises(ValueError,match='monotonic'): audit.event('bad')


def test_preflight_checks_every_gpu_without_a_judge_call(monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    runtime = fixture_torch()
    report = run_preflight(4, torch_module=runtime, package_checker=lambda:{'torch':'fixture'})
    assert report['success'] and report['status']=='passed'
    assert runtime.cuda.selected == [0,1,2,3]
    assert len(report['devices'])==4
    assert not report['judge_connectivity_tested']
    assert not report['typesafe_key_present']


@pytest.mark.parametrize('runtime', [fixture_torch(count=3),fixture_torch(bf16=False)])
def test_hardware_failure_returns_failed_stage(runtime):
    report=run_preflight(4,torch_module=runtime,package_checker=lambda:{})
    assert not report['success'] and report['stage']=='cuda_bf16'
    assert report['error_type']=='RuntimeError'


def test_collective_checks_all_rank_sum():
    runtime=fixture_torch()
    distributed=SimpleNamespace(ReduceOp=SimpleNamespace(SUM='sum'),
        all_reduce=lambda value,op:value.mul_(4))
    assert check_collective(runtime,distributed,0,4)['all_reduce_sum']==4
    with pytest.raises(RuntimeError,match='NCCL'):
        check_collective(runtime,distributed,0,3)


def test_explicit_canary_is_one_typed_request(monkeypatch):
    monkeypatch.setenv('TYPESAFE_API_KEY','fixture-key-not-real')
    client=Client()
    report=run_preflight(4,check_judge=True,torch_module=fixture_torch(),package_checker=lambda:{},
        judge_client_factory=lambda:client,score_factory=lambda **kw:kw)
    assert report['success'] and report['judge_connectivity_tested']
    assert report['judge']['requests']==1 and client.closed and len(client.calls)==1
    assert list(client.calls[0]['questions'])==['startup_check']
    assert 'fixture-key-not-real' not in json.dumps(report)
    assert 'score' not in report['judge']


@pytest.mark.parametrize('client', [Client(model='different'),Client(value=float('nan')),
    Client(error=RuntimeError('fixture-secret-in-service-error'))])
def test_canary_failure_is_sanitized(monkeypatch,client):
    monkeypatch.setenv('TYPESAFE_API_KEY','fixture-secret-key')
    report=run_preflight(4,check_judge=True,torch_module=fixture_torch(),package_checker=lambda:{},
        judge_client_factory=lambda:client,score_factory=lambda **kw:kw)
    assert report['status']=='failed' and report['stage']=='judge_connectivity'
    assert 'fixture-secret' not in json.dumps(report)
    assert client.closed


def test_canary_requires_key_before_client_construction(monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    with pytest.raises(RuntimeError,match='TYPESAFE_API_KEY'):
        check_judge_connectivity(client_factory=lambda:pytest.fail('client must not be created'),score_factory=dict)


def test_cli_writes_failure_json_and_returns_nonzero(tmp_path,monkeypatch,capsys):
    import science_gspo_g.preflight as module
    monkeypatch.setattr(module,'run_preflight',lambda *a,**kw:{'success':False,'status':'failed'})
    output=tmp_path/'preflight.json'
    with pytest.raises(SystemExit) as error:
        main(['--gpus','4','--check-judge','--output',str(output)])
    assert error.value.code==1
    assert json.loads(output.read_text())['status']=='failed'
    assert json.loads(capsys.readouterr().out)['success'] is False


def test_actual_timing_callback_records_first_update_once(tmp_path):
    import ast
    from pathlib import Path
    from unittest.mock import Mock
    from science_gspo_g.io import atomic_json
    tree=ast.parse((Path(__file__).parents[1]/'src/science_gspo_g/train.py').read_text())
    callback=next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name=='TimingCallback')
    startup=Mock()
    startup.event.return_value=12.5
    profiler=Mock()
    cuda=SimpleNamespace(synchronize=Mock(),reset_peak_memory_stats=Mock(),
                         max_memory_allocated=lambda:1024**3,max_memory_reserved=lambda:2*1024**3)
    session_path=tmp_path/'session.json'
    scope={'TrainerCallback':object,'torch':SimpleNamespace(cuda=cuda),'profiler':profiler,
           'startup':startup,'pilot_enabled':True,'session_record':{},'session_path':session_path,
           'atomic_json':atomic_json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[callback],type_ignores=[])),
                 'actual_timing_callback','exec'),scope)
    obj=scope['TimingCallback']()
    for before,after in [(0,1),(1,2)]:
        obj.on_step_begin(None,SimpleNamespace(global_step=before),None)
        obj.on_step_end(None,SimpleNamespace(global_step=after),None)
    assert [call.args[0] for call in startup.event.call_args_list]==['first_update_started','first_update_completed']
    assert profiler.begin.call_count==2 and profiler.end.call_count==2
    assert cuda.synchronize.call_count==4
    assert json.loads(session_path.read_text())['startup_seconds_to_first_update']==12.5


def test_safe_package_and_cuda_failure_reasons_are_retained():
    def package_failure():
        raise RuntimeError('Expected torch==2.13.0, found 2.15.0; use the pinned runtime')
    report=run_preflight(4,torch_module=fixture_torch(),package_checker=package_failure)
    assert report['stage']=='packages'
    assert report['error']=='Expected torch==2.13.0, found 2.15.0; use the pinned runtime'
    report=run_preflight(4,torch_module=fixture_torch(count=2),package_checker=lambda:{})
    assert report['visible_gpus']==2 and 'found 2' in report['error']


@pytest.mark.parametrize('status,location',[(451,'direct'),(429,'response')])
def test_http_status_preserved_without_service_details(monkeypatch,status,location):
    monkeypatch.setenv('TYPESAFE_API_KEY','fixture-secret-key')
    class ServiceError(Exception): pass
    error=ServiceError('Bearer fixture-secret-key at https://secret.example/private?token=fixture-secret-key')
    if location=='direct': error.status_code=status
    else: error.response=SimpleNamespace(status_code=status,body='fixture-secret-body')
    client=Client(error=error)
    report=run_preflight(4,check_judge=True,torch_module=fixture_torch(),package_checker=lambda:{},
        judge_client_factory=lambda:client,score_factory=lambda **kw:kw)
    assert not report['success'] and report['http_status_code']==status
    assert report['error_type']=='ServiceError' and f'HTTP {status}' in report['error']
    encoded=json.dumps(report)
    assert 'fixture-secret' not in encoded and 'https://' not in encoded and 'Bearer' not in encoded

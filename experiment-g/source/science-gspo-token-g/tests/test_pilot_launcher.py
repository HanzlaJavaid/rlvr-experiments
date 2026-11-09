from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest

from science_gspo_g.config import load_config, validate_config
from science_gspo_g.cli import main
from science_gspo_g.io import atomic_json, append_jsonl
from science_gspo_g import pilot

ROOT = Path(__file__).parents[1]


@pytest.fixture
def cfg(tmp_path):
    value=load_config(ROOT/'configs/pilot_g_4x5090.yaml')
    value['paths']['output_dir']=str(tmp_path)
    return value


def test_pilot_preserves_full_workload_except_duration_and_checkpoints(cfg):
    full=load_config(ROOT/'configs/experiment_g_4x5090.yaml')
    for key in ('model','data','generation','judge','experiment'):
        assert cfg[key]==full[key]
    changes={k for k in cfg['training'] if cfg['training'][k]!=full['training'][k]}
    assert changes=={'max_steps','save_steps'}
    assert cfg['training']['max_steps']==12
    assert cfg['pilot']['warmup_updates']==2
    assert cfg['launch']['timeout_minutes']==120
    assert cfg['run_name']!=full['run_name']


@pytest.mark.parametrize('key,value',[('warmup_updates',12),('min_measured_updates',2),('max_updates',8),('startup_timeout_minutes',121),('preflight_timeout_seconds',10000)])
def test_inconsistent_pilot_rejected(cfg,key,value):
    cfg['pilot'][key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_pilot_dry_run_has_no_side_effects(cfg,monkeypatch,capsys,tmp_path):
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:pytest.fail('No subprocess for dry run'))
    assert pilot.run_pilot('pilot.yaml',cfg,dry_run=True) is None
    assert not (tmp_path/cfg['run_name']).exists()
    out=capsys.readouterr().out
    assert '--check-judge' in out and '--nproc_per_node=4' in out


def test_cli_pilot_uses_pilot_config_without_launch(monkeypatch,capsys):
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:pytest.fail('No subprocess for dry run'))
    main(['pilot','--config',str(ROOT/'configs/pilot_g_4x5090.yaml'),'--dry-run'])
    assert 'science_G_pilot_qwen35_4b_4x5090_v1' in capsys.readouterr().out


def test_preflight_failure_prevents_training_and_writes_partial_report(cfg,monkeypatch,tmp_path):
    calls=[]
    def child(command,**kwargs):
        calls.append(command)
        return 9
    monkeypatch.setattr(pilot,'_run_child',child)
    with pytest.raises(subprocess.CalledProcessError):pilot.run_pilot('pilot.yaml',cfg)
    assert len(calls)==1 and 'science_gspo_g.preflight' in calls[0]
    run=tmp_path/cfg['run_name']
    meta=json.loads((run/'launcher_metrics.json').read_text())
    assert meta['status']=='preflight_failed' and meta['returncode']==9
    assert (run/'pilot_report.json').is_file()
    assert json.loads((run/'pilot_report.json').read_text())['success'] is False


def test_training_timeout_retains_reason_and_preflight_time(cfg,monkeypatch,tmp_path):
    calls=[]
    def child(command,**kwargs):
        calls.append(command)
        if len(calls)==1:return 0
        raise pilot.PilotTimeout('startup')
    monkeypatch.setattr(pilot,'_run_child',child)
    with pytest.raises(pilot.PilotTimeout):pilot.run_pilot('pilot.yaml',cfg)
    meta=json.loads((tmp_path/cfg['run_name']/'launcher_metrics.json').read_text())
    assert meta['status']=='timed_out' and meta['timeout_kind']=='startup'
    assert meta['preflight_seconds']>=0 and meta['training_process_seconds']>=0


def test_zero_exit_without_training_does_not_claim_success(cfg,monkeypatch,tmp_path):
    monkeypatch.setattr(pilot,'_run_child',lambda *a,**kw:0)
    with pytest.raises(RuntimeError,match='readiness'):pilot.run_pilot('pilot.yaml',cfg)
    assert json.loads((tmp_path/cfg['run_name']/'pilot_report.json').read_text())['success'] is False


def test_pilot_rejects_existing_run_before_subprocess(cfg,monkeypatch,tmp_path):
    run=tmp_path/cfg['run_name'];atomic_json(run/'launcher_metrics.json',{'old':True})
    monkeypatch.setattr(pilot,'_run_child',lambda *a,**kw:pytest.fail('Do not start'))
    with pytest.raises(ValueError,match='fresh run'):pilot.run_pilot('pilot.yaml',cfg)
    assert json.loads((run/'launcher_metrics.json').read_text())=={'old':True}


def test_successful_supervisor_collects_report_and_optional_cost(cfg,monkeypatch,tmp_path):
    from test_pilot_report import fixture_run
    calls=[]
    run=tmp_path/cfg['run_name']
    def child(command,**kwargs):
        calls.append(command)
        if len(calls)==1:
            atomic_json(run/'preflight.json',{'success':True,'status':'passed','stage':'complete'})
        else:
            fixture_run(run,updates=12)
        return 0
    monkeypatch.setattr(pilot,'_run_child',child)
    report=pilot.run_pilot('pilot.yaml',cfg,hourly_usd=3.0)
    assert report['success'] and len(calls)==2
    assert report['projection']['assumed_cluster_hourly_usd']==3.0
    assert report['launcher']['status']=='completed'
    assert report['completed_updates_all_ranks']==list(range(1,13))


def test_common_updates_require_every_rank(tmp_path):
    for rank in range(4):
        append_jsonl(tmp_path/'ranks'/f'rank_{rank:05d}'/'update_timings.jsonl',
                     [{'global_step':step} for step in range(1,4 if rank!=3 else 3)])
    assert pilot.common_updates(tmp_path)==[1,2]


def test_child_is_terminated_when_deadline_expires(monkeypatch,tmp_path):
    child=SimpleNamespace(pid=12345,wait=lambda **kw:0)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:child)
    monkeypatch.setattr(pilot.time,'monotonic',lambda:10.)
    terminated=[]
    monkeypatch.setattr(pilot,'_terminate',lambda p:terminated.append(p.pid))
    with (tmp_path/'log').open('w') as console:
        with pytest.raises(pilot.PilotTimeout,match='preflight'):
            pilot._run_child(['fake'],env={},console=console,deadline=9.,timeout_kind='preflight')
    assert terminated==[12345]

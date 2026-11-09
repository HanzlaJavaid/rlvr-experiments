import json
from pathlib import Path
from unittest.mock import patch
import pytest
from science_gspo_g.config import load_config, validate_config, batch_plan
from science_gspo_g.cli import main

CONFIG = Path(__file__).parents[1] / 'configs/experiment_g_4x5090.yaml'


def test_default_is_g_only_full_epoch():
    cfg = load_config(CONFIG)
    assert cfg['experiment'] == {'id':'G','method':'gspo_token','calibration':False,'comparison_arms':False}
    assert cfg['training'].get('max_steps', -1) == -1
    assert cfg['training']['epochs'] == 1
    assert batch_plan(cfg,4)['gradient_accumulation_steps'] == 16
    assert cfg['wandb']['mode'] == 'disabled'
    assert cfg['run_name'].endswith('_v1')
    assert cfg['paths']['data_dir'] == './data/science_g_qwen35_v1'
    assert cfg['model']['name'] == 'Qwen/Qwen3.5-4B'
    assert cfg['training']['learning_rate'] == 2e-6
    assert cfg['training']['lr_scheduler_type'] == 'constant_with_warmup'
    assert len(cfg['training']['reward_weights']) == 4
    assert 'outcome_weights' not in cfg['training']


@pytest.mark.parametrize('key,value', [('advantage_std_floor',0),('process_z_clip',float('nan')),('learning_rate',-1),('process_weight',-1),('epsilon',float('inf')),('beta',True),('max_steps',0),('outcome_weights',[1,0,0])])
def test_invalid_objective_values_fail_before_training(key,value):
    cfg=load_config(CONFIG);cfg['training'][key]=value
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('field', ['calibration','comparison_arms'])
def test_extra_experiments_rejected(field):
    cfg=load_config(CONFIG);cfg['experiment'][field]=True
    with pytest.raises(ValueError):validate_config(cfg)


def test_dry_launch_does_not_start_process(capsys):
    with patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
        main(['launch','--config',str(CONFIG),'--dry-run'])
    out=capsys.readouterr().out
    assert 'science_gspo_g.train' in out and '--nproc_per_node=4' in out
    assert 'science_grpo.train' not in out


def test_inspect_is_local_and_reports_judge_request_count(capsys):
    main(['inspect-config','--config',str(CONFIG)])
    report=json.loads(capsys.readouterr().out)
    assert report['judge_requests_per_update'] == 256
    assert report['batch_plan']['questions_per_update'] == 16


@pytest.mark.parametrize('key,value', [('temperature', .6), ('top_p', .8),
    ('top_k', 40), ('min_p', .1), ('presence_penalty', 0),
    ('repetition_penalty', 1.1), ('enable_thinking', False)])
def test_sampling_cannot_silently_depart_from_user_preset(key, value):
    cfg=load_config(CONFIG);cfg['generation'][key]=value
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('key,value', [('lr_scheduler_type','cosine'),
    ('warmup_ratio',float('nan')), ('warmup_ratio',1), ('warmup_ratio',True),
    ('reward_weights',[1.]*6)])
def test_obsolete_schedule_or_reward_schema_rejected(key,value):
    cfg=load_config(CONFIG);cfg['training'][key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_model_and_experiment_cannot_resume_e_by_accident():
    cfg=load_config(CONFIG);cfg['model']['name']='Qwen/Qwen3-4B-Thinking-2507'
    with pytest.raises(ValueError):validate_config(cfg)
    cfg=load_config(CONFIG);cfg['experiment']['id']='E'
    with pytest.raises(ValueError):validate_config(cfg)


def test_f_configuration_is_rejected_and_default_paths_are_separate():
    original = load_config(CONFIG)
    original['experiment']['id'] = 'F'
    with pytest.raises(ValueError, match='experiment G only'):
        validate_config(original)
    cfg = load_config(CONFIG)
    assert cfg['paths']['output_dir'] == './runs_g'
    assert cfg['paths']['data_dir'] == './data/science_g_qwen35_v1'
    pilot = load_config(CONFIG.parent / 'pilot_g_4x5090.yaml')
    assert pilot['paths']['output_dir'] == './runs_g_pilots'
    assert pilot['run_name'].startswith('science_G_pilot_')

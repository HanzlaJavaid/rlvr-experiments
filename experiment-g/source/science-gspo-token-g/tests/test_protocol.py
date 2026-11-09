"""Protocol and preservation contracts; no real model or judge calls."""
import json
from pathlib import Path

import pytest

from science_gspo_g.config import load_config
from science_gspo_g.execution import choose_checkpoint
from science_gspo_g.train import write_protocol


ROOT = Path(__file__).parents[1]


def write_fixture(root):
    cfg = load_config(ROOT / 'configs/experiment_g_4x5090.yaml')
    return write_protocol(cfg, root, {'model_revision': cfg['model']['revision']},
                          {'fixture': 'no real preparation'}, {}, {})


def test_protocol_records_g_visibility_and_same_trace_credit(tmp_path):
    protocol = write_fixture(tmp_path)
    assert protocol['arm'] == 'G'
    assert protocol['format'] == 'science-gspo-token-G-v1'
    assert 'reasoning_start<=token<reasoning_stop' in protocol['advantage_formula']
    assert 'first_visible_model_token' not in protocol['advantage_formula']
    for metric in ('model_quality', 'application_quality', 'verification_quality'):
        assert 'no target reference or final answer' in protocol['judge_visibility'][metric]
    assert 'fallible reference answer' in protocol['judge_visibility']['scientific_correctness']
    assert write_fixture(tmp_path) == protocol


def test_f_protocol_cannot_be_overwritten(tmp_path):
    original = json.dumps({'arm': 'F', 'protocol_hash': 'historical-f'})
    (tmp_path / 'protocol.json').write_text(original)
    with pytest.raises(ValueError, match='choose a new run name'):
        write_fixture(tmp_path)
    assert (tmp_path / 'protocol.json').read_text() == original
    assert not (tmp_path / 'resolved_config.json').exists()


def test_f_checkpoint_cannot_be_resumed_as_g(tmp_path):
    checkpoint = tmp_path / 'checkpoint-171'
    checkpoint.mkdir()
    for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'adapter_config.json',
                 'adapter_model.safetensors', *(f'rng_state_{i}.pth' for i in range(4))):
        (checkpoint / name).write_text('fixture')
    (checkpoint / 'checkpoint_complete.json').write_text(json.dumps(
        {'protocol_hash': 'historical-f', 'world_size': 4}))
    with pytest.raises(ValueError, match='another protocol'):
        choose_checkpoint(tmp_path, str(checkpoint), protocol_hash='new-g', world_size=4)

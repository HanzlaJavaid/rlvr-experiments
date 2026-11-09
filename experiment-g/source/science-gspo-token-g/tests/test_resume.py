"""CPU source contracts for the pinned PEFT save / Transformers reload layout.

Exact upstream method excerpts drive the path decisions; tiny scalar adapters
stand in for safetensors I/O. No real policy, tokenizer, CUDA or service is used.
"""
import __future__
import ast
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from science_gspo_g.resume import restore_checkpoint_policy


ROOT = Path(__file__).parents[1]
FIXTURES = ROOT/'tests/fixtures'


def peft_save_layout():
    """Execute PEFT's exact adapter-selection and output-directory AST nodes."""
    method = ast.parse((FIXTURES/'peft_0_21_2_save.py.txt').read_text()).body[0]
    selection = next(node for node in method.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == 'selected_adapters is None')
    output_dir = next(node for node in ast.walk(method) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'output_dir'
                              for target in node.targets))
    # The actual serializer/config writer is replaced by a scalar file writer.
    wrapper = ast.parse('''
def save_pretrained(self, save_directory, selected_adapters=None):
    pass
    for adapter_name in selected_adapters:
        pass
        self.write_scalar_fixture(output_dir, adapter_name)
''').body[0]
    wrapper.body[0] = deepcopy(selection)
    wrapper.body[1].body[0] = deepcopy(output_dir)
    namespace = {'os': os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 'pinned-peft-save-layout', 'exec'), namespace)
    return namespace['save_pretrained']


class ScalarAdapters:
    """Parameter identities and trainability behave like this vanilla LoRA case."""
    save_pretrained = peft_save_layout()

    def __init__(self, policy=0.0, reference=0.25, *, with_reference=True):
        self.peft_config = {'default': SimpleNamespace()}
        if with_reference:
            self.peft_config['ref'] = SimpleNamespace()
        self.active_adapters = ['default']
        self.weights = {name: torch.nn.Parameter(torch.tensor([policy if name == 'default' else reference]),
                        requires_grad=name == 'default') for name in self.peft_config}
        self.base = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.loads = []

    def named_parameters(self):
        yield 'base_model.model.q_proj.base_layer.weight', self.base
        for name, parameter in self.weights.items():
            yield f'base_model.model.q_proj.lora_A.{name}.weight', parameter

    def write_scalar_fixture(self, output_dir, adapter_name):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'adapter_config.json').write_text('{}')
        # This file deliberately contains JSON for the lightweight loader stub.
        (directory/'adapter_model.safetensors').write_text(json.dumps(
            {'value': self.weights[adapter_name].item()}))

    def load_adapter(self, model_id, adapter_name, is_trainable=False):
        value = json.loads((Path(model_id)/'adapter_model.safetensors').read_text())['value']
        self.weights[adapter_name].data.fill_(value)
        self.weights[adapter_name].requires_grad_(is_trainable)
        self.loads.append((Path(model_id), adapter_name, is_trainable))

    def set_adapter(self, adapter_name):
        self.active_adapters = [adapter_name]
        for name, parameter in self.weights.items():
            parameter.requires_grad_(name == adapter_name)


def trainer_classes():
    namespace = {
        '__package__': 'science_gspo_g', 'os': os,
        'logger': logging.getLogger('pinned-resume-contract'),
        'FSDP_MODEL_NAME': 'pytorch_model_fsdp', 'CONFIG_NAME': 'config.json',
        'ADAPTER_WEIGHTS_NAME': 'adapter_model.bin', 'ADAPTER_SAFE_WEIGHTS_NAME': 'adapter_model.safetensors',
        'WEIGHTS_NAME': 'pytorch_model.bin', 'WEIGHTS_INDEX_NAME': 'pytorch_model.bin.index.json',
        'SAFE_WEIGHTS_NAME': 'model.safetensors', 'SAFE_WEIGHTS_INDEX_NAME': 'model.safetensors.index.json',
        '_is_peft_model': lambda model: isinstance(model, ScalarAdapters),
        'is_sagemaker_mp_enabled': lambda: False,
    }
    source = (FIXTURES/'transformers_5_20_2_resume.py.txt').read_text()
    exec(compile(source, 'pinned-transformers-resume', 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)

    class UpstreamTrainer:
        _load_from_checkpoint = namespace['_load_from_checkpoint']
        is_fsdp_enabled = False
        def __init__(self, model):
            self.model = model

    # Compile the actual locally implemented nested override, so tests exercise
    # its super() ordering and helper call rather than duplicating that code.
    train_tree = ast.parse((ROOT/'src/science_gspo_g/train.py').read_text())
    trainer = next(node for node in ast.walk(train_tree)
                   if isinstance(node, ast.ClassDef) and node.name == 'InstrumentedTrainer')
    override = next(node for node in trainer.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_load_from_checkpoint')
    wrapper = ast.ClassDef(name='CorrectedTrainer', bases=[ast.Name(id='UpstreamTrainer', ctx=ast.Load())],
                           keywords=[], body=[deepcopy(override)], decorator_list=[])
    namespace['UpstreamTrainer'] = UpstreamTrainer
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 'local-instrumented-resume', 'exec'), namespace)
    return UpstreamTrainer, namespace['CorrectedTrainer']


def test_exact_peft_and_transformers_contract_reproduces_root_default_skip(tmp_path):
    saved = ScalarAdapters(policy=0.75, reference=0.25)
    saved.save_pretrained(str(tmp_path))
    assert (tmp_path/'adapter_model.safetensors').is_file()
    assert (tmp_path/'ref/adapter_model.safetensors').is_file()
    model = ScalarAdapters(policy=0.0, reference=0.25)
    upstream, _ = trainer_classes()
    upstream(model)._load_from_checkpoint(str(tmp_path))
    assert model.weights['default'].item() == 0.0  # Root policy was skipped.
    assert model.loads == [(tmp_path/'ref', 'ref', False)]


@pytest.mark.parametrize('with_reference', [False, True])
@pytest.mark.parametrize('explicit_model', [False, True])
def test_corrected_resume_restores_policy_and_preserves_reference(tmp_path, with_reference, explicit_model):
    saved = ScalarAdapters(policy=0.75, reference=0.25, with_reference=with_reference)
    saved.save_pretrained(str(tmp_path))
    model = ScalarAdapters(policy=0.0, reference=0.25, with_reference=with_reference)
    identities = {name: id(parameter) for name, parameter in model.weights.items()}
    _, corrected = trainer_classes()
    trainer = corrected(model)
    trainer._load_from_checkpoint(str(tmp_path), model=model if explicit_model else None)
    assert model.weights['default'].item() == 0.75
    assert model.weights['default'].requires_grad
    assert model.active_adapters == ['default']
    assert identities == {name: id(parameter) for name, parameter in model.weights.items()}
    assert not model.base.requires_grad
    if with_reference:
        assert model.weights['ref'].item() == 0.25
        assert not model.weights['ref'].requires_grad
    assert trainer._g_resume_adapter_state['policy_adapter'] == 'default'
    assert model.loads[-1] == (tmp_path, 'default', True)


def test_restore_rejects_missing_root_policy(tmp_path):
    ScalarAdapters().write_scalar_fixture(str(tmp_path/'ref'), 'ref')
    with pytest.raises(ValueError, match='checkpoint-root default'):
        restore_checkpoint_policy(ScalarAdapters(), tmp_path)


@pytest.mark.parametrize('defect', ['frozen_policy', 'trainable_reference', 'trainable_base'])
def test_restore_rejects_invalid_trainability(tmp_path, defect):
    saved = ScalarAdapters(policy=0.75)
    saved.save_pretrained(str(tmp_path))
    model = ScalarAdapters()
    original_set = model.set_adapter
    def broken_set(name):
        original_set(name)
        if defect == 'frozen_policy':
            model.weights['default'].requires_grad_(False)
        elif defect == 'trainable_reference':
            model.weights['ref'].requires_grad_(True)
        else:
            model.base.requires_grad_(True)
    model.set_adapter = broken_set
    with pytest.raises(RuntimeError):
        restore_checkpoint_policy(model, tmp_path)

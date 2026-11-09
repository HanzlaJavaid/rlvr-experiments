"""Offline architecture and sampler contracts; never load the 4B model."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from science_gspo_g.config import load_config
from science_gspo_g.model import (
    THINKING_GENERAL_SAMPLING, effective_sampling_settings, generation_kwargs,
    load_text_policy, validate_loading_info, validate_snapshot_config,
)
from science_gspo_g.preflight import check_qwen35_runtime
from science_gspo_g.weight_sync import qwen35_checkpoint_name, validate_qwen35_sync_target

ROOT = Path(__file__).parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT / 'configs/experiment_g_4x5090.yaml')


def snapshot():
    return {'model_type': 'qwen3_5', 'architectures': ['Qwen3_5ForConditionalGeneration'],
            'tie_word_embeddings': True,
            'text_config': {'model_type': 'qwen3_5_text', 'hidden_size': 2560,
                'num_hidden_layers': 32, 'vocab_size': 248320, 'num_attention_heads': 16,
                'num_key_value_heads': 4, 'max_position_embeddings': 262144,
                'layer_types': (['linear_attention']*3+['full_attention'])*8}}


def policy():
    return SimpleNamespace(config=SimpleNamespace(**snapshot()['text_config']),
        named_parameters=lambda: iter([('model.embed_tokens.weight', object()),
                                      ('model.norm.weight', object())]))


def test_load_uses_causal_text_class_and_explicit_checkpoint_prefix_conversion(cfg, tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(snapshot()))
    base = policy()
    text_config = SimpleNamespace(from_dict=lambda values: SimpleNamespace(**values))
    loader = Mock(return_value=(base, {'missing_keys': [], 'unexpected_keys': ['model.visual.a', 'mtp.x']}))
    transformers = SimpleNamespace(Qwen3_5TextConfig=text_config,
                                   Qwen3_5ForCausalLM=SimpleNamespace(from_pretrained=loader))
    loaded, audit = load_text_policy(tmp_path, cfg, 2, torch_module=SimpleNamespace(bfloat16='bf16'),
                                    transformers_module=transformers)
    assert loaded is base and loaded.config.use_cache is False
    kw = loader.call_args.kwargs
    assert kw['config'].model_type == 'qwen3_5_text'
    assert kw['config'].architectures == ['Qwen3_5ForCausalLM']
    assert kw['key_mapping'] == {r'^model\.language_model\.': 'model.'}
    assert kw['local_files_only'] and kw['output_loading_info']
    assert kw['device_map'] == {'': 2}
    assert 'use_kernels' not in kw  # no unpinned executable Hub downloads
    assert audit['vision_loaded'] is False and audit['mtp_loaded'] is False
    assert audit['gpu_forward_backward_tested'] is False


@pytest.mark.parametrize('info', [
    {'missing_keys':['model.layers.0.linear_attn.in_proj_qkv.weight']},
    {'mismatched_keys':['model.embed_tokens.weight']}, {'error_msgs':['bad load']},
    {'unexpected_keys':['model.layers.0.typo.weight']},
])
def test_incomplete_or_wrong_text_weights_are_fatal(info):
    with pytest.raises(RuntimeError): validate_loading_info(info)


def test_wrong_architecture_or_context_is_fatal(cfg):
    raw=snapshot();raw['text_config']['num_hidden_layers']=24
    with pytest.raises(ValueError,match='architecture'):validate_snapshot_config(raw,cfg)
    raw=snapshot();raw['text_config']['max_position_embeddings']=100
    with pytest.raises(ValueError,match='context'):validate_snapshot_config(raw,cfg)


def test_all_six_sampler_settings_are_explicit_and_audited(cfg):
    extra=generation_kwargs(cfg,[248044,248046])
    assert {k:extra[k] for k in THINKING_GENERAL_SAMPLING} == THINKING_GENERAL_SAMPLING
    backend=SimpleNamespace(generation_kwargs=extra)
    report=effective_sampling_settings(backend,cfg)
    assert report['effective']==THINKING_GENERAL_SAMPLING and report['thinking_enabled']
    backend.generation_kwargs.pop('presence_penalty')
    with pytest.raises(RuntimeError,match='sampling'):effective_sampling_settings(backend,cfg)


def test_requested_sampling_is_not_silently_overridden(cfg):
    cfg['generation']['presence_penalty']=0.0
    with pytest.raises(ValueError,match='exactly'):generation_kwargs(cfg,[248044])


def test_weight_stream_uses_checkpoint_namespace_without_manual_fusion():
    assert qwen35_checkpoint_name('model.layers.0.linear_attn.in_proj_qkv.weight') == 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight'
    assert qwen35_checkpoint_name('model.layers.0.linear_attn.in_proj_a.weight') == 'model.language_model.layers.0.linear_attn.in_proj_a.weight'
    assert qwen35_checkpoint_name('lm_head.weight') == 'lm_head.weight'
    for name in ['model.visual.weight','model.language_model.layers.0.weight','visual.weight']:
        with pytest.raises(ValueError):qwen35_checkpoint_name(name)


def test_runtime_preflight_only_checks_classes_not_model_loading():
    class Qwen3_5TextConfig: model_type='qwen3_5_text'
    class Qwen3_5ForCausalLM: pass
    report=check_qwen35_runtime(transformers_module=SimpleNamespace(
        Qwen3_5TextConfig=Qwen3_5TextConfig,Qwen3_5ForCausalLM=Qwen3_5ForCausalLM))
    assert report['model_weights_loaded'] is False and report['gpu_forward_backward_tested'] is False
    with pytest.raises(RuntimeError):check_qwen35_runtime(transformers_module=SimpleNamespace())


def sync_backend():
    config=SimpleNamespace(**snapshot()['text_config'])
    packed={'qkv_proj':['q_proj','k_proj','v_proj'],
            'gate_up_proj':['gate_proj','up_proj'],
            'in_proj_qkvz':['in_proj_qkv','in_proj_z'],
            'in_proj_ba':['in_proj_b','in_proj_a']}
    cls=type('Qwen3_5ForConditionalGeneration',(),{})
    target=cls()
    target.packed_modules_mapping=packed
    target.language_model=SimpleNamespace(model=SimpleNamespace(config=deepcopy(config)))
    target.named_parameters=lambda:iter([('language_model.model.embed_tokens.weight',object())])
    return SimpleNamespace(model=SimpleNamespace(config=config),llm=SimpleNamespace(
        llm_engine=SimpleNamespace(model_executor=SimpleNamespace(driver_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=target)))))),target


def test_initialized_sync_target_is_native_text_only_with_matching_gdn_packing():
    backend,target=sync_backend()
    audit=validate_qwen35_sync_target(backend)
    assert audit['vision_loaded'] is False
    assert audit['projection_fusion']=='native vLLM QKV/gate-up/GDN loader'
    target.packed_modules_mapping['in_proj_ba']=['in_proj_a','in_proj_b']
    with pytest.raises(RuntimeError,match='packing'):validate_qwen35_sync_target(backend)


def test_sync_audit_rejects_vision_weights_and_architecture_mismatch():
    backend,target=sync_backend()
    target.named_parameters=lambda:iter([('visual.patch_embed.weight',object())])
    with pytest.raises(RuntimeError,match='vision'):validate_qwen35_sync_target(backend)
    backend,target=sync_backend()
    target.language_model.model.config.hidden_size=2048
    with pytest.raises(RuntimeError,match='architectures differ'):validate_qwen35_sync_target(backend)


def test_exact_pinned_trl_generation_receives_presence_penalty_and_candidate_support(cfg):
    """Execute the pinned generation hook with a fake engine, never a GPU/model."""
    source=ROOT/'tests/fixtures/trl_1_17_vllm_generation.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VLLMGeneration')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    capture={}
    def sample(**kw):capture.update(kw);return SimpleNamespace(**kw)
    scope={'SamplingParams':sample,'nullcontext':nullcontext,'ProfilingContext':object,'is_peft_model':lambda _:False,
           'extract_logprobs':lambda _:([[[-0.25]]],[[[42]]])}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
    out=SimpleNamespace(prompt_token_ids=[1,2],outputs=[SimpleNamespace(
        token_ids=[42],sampling_mask=SimpleNamespace(token_ids=[[42,43]]))])
    backend=SimpleNamespace(**{k:v for k,v in THINKING_GENERAL_SAMPLING.items() if k!='presence_penalty'},
        mode='colocate',accelerator=SimpleNamespace(),model=object(),tensor_parallel_size=1,
        max_completion_length=8192,logprobs=0,generation_kwargs=generation_kwargs(cfg,[248044]),
        enable_sleep_mode=False,_llm_weights_sleeping=False,structured_outputs_regex=None,
        return_sampling_mask=True,llm=SimpleNamespace(generate=Mock(return_value=[out])))
    result=scope['generate'](backend,[[1,2]],None,4)
    assert {k:capture[k] for k in THINKING_GENERAL_SAMPLING}==THINKING_GENERAL_SAMPLING
    assert capture['stop_token_ids']==[248044]
    assert result[-1]==[[[42,43]]] and result[2]==[[[-0.25]]]

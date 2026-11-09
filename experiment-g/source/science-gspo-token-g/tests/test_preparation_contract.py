"""Asset preparation contracts without downloads, model loading or evaluators."""
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from science_gspo_g.preparation import prepare_assets


def test_preparation_pins_model_and_explicitly_opens_native_thinking(tmp_path):
    cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/experiment_g_4x5090.yaml').read_text())
    cfg['paths']={'output_dir':str(tmp_path/'runs'),'data_dir':str(tmp_path/'data')}
    calls=[]
    hub=ModuleType('huggingface_hub')
    hub.HfApi=lambda:SimpleNamespace(model_info=lambda name,revision:
        SimpleNamespace(sha=revision))
    hub.snapshot_download=lambda name,**kwargs:calls.append(('snapshot',name,kwargs))
    transformers=ModuleType('transformers')

    class Tokenizer:
        pad_token_id=0
        eos_token='EOS'
        def apply_chat_template(self,messages,**kwargs):
            calls.append(('template',messages,kwargs))
            assert kwargs['enable_thinking'] is True
            return '<|im_start|>assistant\n<think>\n'

    tokenizer=Tokenizer()
    transformers.AutoTokenizer=SimpleNamespace(from_pretrained=lambda path:tokenizer)
    with patch.dict('sys.modules',{'huggingface_hub':hub,'transformers':transformers}), \
         patch('science_gspo_g.preparation.prepare_data',return_value={'fixture':True}):
        assets,manifest=prepare_assets(cfg)
    assert manifest=={'fixture':True}
    snapshot=calls[0]
    assert snapshot[1]=='Qwen/Qwen3.5-4B'
    assert snapshot[2]['revision']==cfg['model']['revision']
    assert assets['model_loader']=='qwen3_5_text'
    assert assets['language_model_only'] is True
    assert assets['chat_template_kwargs']=={'enable_thinking':True}
    assert tokenizer.padding_side=='left'
    stored=json.loads((tmp_path/'runs'/cfg['run_name']/'assets.json').read_text())
    assert stored==assets


def test_preparation_rejects_template_with_thinking_disabled(tmp_path):
    cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/experiment_g_4x5090.yaml').read_text())
    cfg['paths']={'output_dir':str(tmp_path/'runs'),'data_dir':str(tmp_path/'data')}
    hub=ModuleType('huggingface_hub')
    hub.HfApi=lambda:SimpleNamespace(model_info=lambda name,revision:SimpleNamespace(sha=revision))
    hub.snapshot_download=lambda *a,**kw:None
    transformers=ModuleType('transformers')
    tokenizer=SimpleNamespace(pad_token_id=0,apply_chat_template=lambda *a,**kw:'<|im_start|>assistant\n')
    transformers.AutoTokenizer=SimpleNamespace(from_pretrained=lambda path:tokenizer)
    with patch.dict('sys.modules',{'huggingface_hub':hub,'transformers':transformers}), \
         patch('science_gspo_g.preparation.prepare_data',side_effect=AssertionError('must reject first')):
        with pytest.raises(ValueError,match='prefilled'):
            prepare_assets(cfg)

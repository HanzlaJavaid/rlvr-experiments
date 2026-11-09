"""Protect the local audit from dropping metrics added by upstream log()."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
from science_gspo_g.io import append_jsonl


def test_local_log_retains_upstream_reward_metrics(tmp_path):
    path = Path(__file__).parents[1] / 'src/science_gspo_g/train.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'InstrumentedTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'log')
    class Base:
        def log(self, logs, start_time=None):
            logs.update({'g/outcome_reward': 0.7, 'kl': 0.012, 'bad_numeric': float('nan')})
    node = ast.ClassDef(name='Audit', bases=[ast.Name(id='Base',ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
    scope = {'Base':Base,'append_jsonl':append_jsonl,'run_dir':tmp_path,'math':math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    trainer=scope['Audit']()
    trainer.accelerator=SimpleNamespace(is_main_process=True)
    trainer.state=SimpleNamespace(global_step=2)
    trainer.log({'loss':0.1})
    row=json.loads((tmp_path/'metrics.jsonl').read_text())
    assert row['g/outcome_reward']==0.7 and row['kl']==0.012 and row['global_step']==2
    assert 'bad_numeric' not in row
    assert row['nonfinite_metric_keys'] == ['bad_numeric']

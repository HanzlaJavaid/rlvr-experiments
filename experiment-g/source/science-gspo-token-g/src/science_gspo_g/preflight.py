"""Explicit pilot readiness checks and small, credential-free startup journals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

from .io import append_jsonl, atomic_json


class JudgeConnectivityError(RuntimeError):
    """Only sanitized service metadata crosses the connectivity-check boundary."""
    def __init__(self, source_error_type, status_code=None):
        self.source_error_type = source_error_type
        self.status_code = status_code
        suffix = f' (HTTP {status_code})' if status_code is not None else ''
        super().__init__(f'Judge connectivity check failed{suffix}')


def _http_status_code(exc):
    try:
        values = [getattr(exc, 'status_code', None),
                  getattr(getattr(exc, 'response', None), 'status_code', None)]
    except Exception:
        return None
    for value in values:
        if isinstance(value, str) and re.fullmatch(r'[1-5][0-9]{2}', value):
            value = int(value)
        if type(value) is int and 100 <= value <= 599:
            return value
    return None


def _safe_local_reason(exc):
    """Retain our own bounded diagnostics, never arbitrary dependency text."""
    message = str(exc)
    known = {
        'GPU count must be a positive integer',
        'BF16 is required on every GPU',
        'BF16 matrix-multiply validation failed',
        'TYPESAFE_API_KEY is required for the judge connectivity check',
        'Judge returned a different model version',
        'Judge returned an invalid typed response',
    }
    if message in known:
        return message
    if re.fullmatch(r'Need [0-9]+ visible CUDA GPUs; found [0-9]+ \(CUDA available: (True|False)\)', message):
        return message
    if re.fullmatch(r'Expected [A-Za-z0-9_.-]+==[A-Za-z0-9_.+\-]+, found [A-Za-z0-9_.+\-]+; use the pinned runtime', message):
        return message
    if type(exc).__name__ == 'PackageNotFoundError':
        name = getattr(exc, 'name', None)
        if isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            return f'Required package is not installed: {name}'
    return 'Preflight failed at the recorded stage; inspect the runtime and service availability.'


class StartupAudit:
    """A single-rank event stream; disabled instances have no file effects."""
    def __init__(self, run_dir, rank, *, enabled, session_id=None, clock=time.perf_counter):
        self.enabled = enabled
        self.rank = rank
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.clock = clock
        self.started = clock()
        self.path = Path(run_dir) / 'ranks' / f'rank_{rank:05d}' / 'startup_events.jsonl'

    def event(self, name, **details):
        elapsed = self.clock() - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Startup clock must be monotonic and finite')
        if self.enabled:
            append_jsonl(self.path, [{
                'rank': self.rank, 'session_id': self.session_id, 'event': name,
                'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': elapsed,
                **details,
            }])
        return elapsed


def check_bf16_device(torch, index):
    """Exercise a real small matrix multiplication on one explicitly chosen GPU."""
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 is required on every GPU')
        a = torch.ones((32, 32), device=f'cuda:{index}', dtype=torch.bfloat16)
        actual = float((a @ a).float().sum().item())
        if actual != 32768.0:
            raise RuntimeError('BF16 matrix-multiply validation failed')
        device = torch.cuda.get_device_properties(index)
        result = {'index': index, 'name': device.name, 'vram_bytes': device.total_memory,
                  'capability': list(torch.cuda.get_device_capability(index))}
        del a
        return result


def check_collective(torch, dist, local_rank, world_size):
    """Small all-rank sum verifies NCCL participation, not communication speed."""
    value = torch.ones(1, device=f'cuda:{local_rank}', dtype=torch.float32)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    if float(value.item()) != float(world_size):
        raise RuntimeError('NCCL all-reduce validation failed')
    return {'world_size': world_size, 'all_reduce_sum': float(value.item())}


def check_qwen35_runtime(*, transformers_module=None):
    """Check architecture availability without loading model weights or kernels."""
    if transformers_module is None:
        import transformers as transformers_module
    text_config = getattr(transformers_module, 'Qwen3_5TextConfig', None)
    text_model = getattr(transformers_module, 'Qwen3_5ForCausalLM', None)
    if text_config is None or text_model is None or getattr(text_config, 'model_type', None) != 'qwen3_5_text':
        raise RuntimeError('Pinned Transformers lacks the reviewed Qwen3.5 text architecture')
    return {'text_config_class': text_config.__name__, 'policy_class': text_model.__name__,
            'vision_required': False, 'model_weights_loaded': False,
            'gpu_forward_backward_tested': False,
            'training_kernel_note': 'DeltaNet training kernels and long-sequence memory require the GPU pilot; no Hub kernels are downloaded automatically.'}


def check_judge_connectivity(*, client_factory=None, score_factory=None):
    """One explicit typed-response canary; no science grading or calibration."""
    if not os.environ.get('TYPESAFE_API_KEY'):
        raise RuntimeError('TYPESAFE_API_KEY is required for the judge connectivity check')
    model = 'jev-1.14.0'
    if client_factory is None or score_factory is None:
        from typesafe_sdk import TypeSafeClient, RetryPolicy, Score
        client_factory = client_factory or (lambda: TypeSafeClient(
            model=model, timeout=30.0, retry=RetryPolicy(max_retries=0, timeout=45.0)))
        score_factory = score_factory or Score
    client = None
    try:
        client = client_factory()
        response = client.system_one(
            state={'message': 'Experiment G startup connectivity check.'},
            questions={'startup_check': score_factory(type='score',
                instructions='Check whether the supplied state contains a startup connectivity message.',
                criteria=['The state does not contain a startup connectivity message.',
                          'The state contains a startup connectivity message.'])})
        raw = response.model_dump(mode='json') if hasattr(response, 'model_dump') else response
        if raw.get('model') != model:
            raise RuntimeError('Judge returned a different model version')
        answer = raw.get('answers', {}).get('startup_check', {})
        score = answer.get('score')
        if (answer.get('type') != 'score' or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise RuntimeError('Judge returned an invalid typed response')
        return {'model': model, 'response_valid': True, 'requests': 1,
                'purpose': 'connectivity_only'}
    except Exception as exc:
        # SDK errors can contain request details. Never serialize their text or
        # traceback into a shareable preflight artifact.
        raise JudgeConnectivityError(type(exc).__name__, _http_status_code(exc)) from None
    finally:
        if client is not None:
            close = getattr(client, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass


def run_preflight(gpus=4, *, check_judge=False, torch_module=None, package_checker=None,
                  judge_client_factory=None, score_factory=None, model_checker=None):
    """Return a structured success/failure report, with no policy model loading."""
    started = time.perf_counter()
    report = {'success': False, 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'requested_gpus': gpus, 'stage': 'configuration', 'judge_connectivity_tested': False,
              'typesafe_key_present': bool(os.environ.get('TYPESAFE_API_KEY'))}
    try:
        if type(gpus) is not int or gpus < 1:
            raise ValueError('GPU count must be a positive integer')
        report['stage'] = 'packages'
        real_packages = package_checker is None
        if package_checker is None:
            from .train import package_versions
            package_checker = package_versions
        report['packages'] = package_checker()
        if real_packages or model_checker is not None:
            report['stage'] = 'model_runtime'
            report['model_runtime'] = (model_checker or check_qwen35_runtime)()
        report['stage'] = 'cuda_bf16'
        if torch_module is None:
            import torch as torch_module
        available = bool(torch_module.cuda.is_available())
        visible = int(torch_module.cuda.device_count())
        report.update(cuda_available=available, visible_gpus=visible)
        if not available or visible < gpus:
            raise RuntimeError(f'Need {gpus} visible CUDA GPUs; found {visible} (CUDA available: {available})')
        report['torch_cuda'] = torch_module.version.cuda
        report['devices'] = [check_bf16_device(torch_module, index) for index in range(gpus)]
        if check_judge:
            report['stage'] = 'judge_connectivity'
            report['judge_connectivity_tested'] = True
            report['judge'] = check_judge_connectivity(client_factory=judge_client_factory,
                                                       score_factory=score_factory)
        report.update(success=True, status='passed', stage='complete')
    except Exception as exc:
        if isinstance(exc, JudgeConnectivityError):
            report.update(status='failed', error_type=exc.source_error_type, error=str(exc))
            if exc.status_code is not None:
                report['http_status_code'] = exc.status_code
        else:
            report.update(status='failed', error_type=type(exc).__name__, error=_safe_local_reason(exc))
    report['elapsed_seconds'] = time.perf_counter() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--check-judge', action='store_true', help='Make one typed connectivity request to pinned Jev')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = run_preflight(args.gpus, check_judge=args.check_judge)
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

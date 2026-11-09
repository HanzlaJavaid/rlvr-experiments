"""Bounded Experiment G pilot supervisor. Importing it starts no work."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

from .config import batch_plan, run_directory
from .execution import torchrun_command
from .io import atomic_json, read_jsonl


class PilotTimeout(RuntimeError):
    def __init__(self, kind):
        self.kind = kind
        super().__init__(f"Pilot {kind} deadline reached")


def common_updates(run_dir, world_size=4):
    sets = []
    for rank in range(world_size):
        rows = read_jsonl(Path(run_dir)/'ranks'/f'rank_{rank:05d}'/'update_timings.jsonl', repair_tail=False)
        sets.append({row['global_step'] for row in rows if type(row.get('global_step')) is int})
    return sorted(set.intersection(*sets)) if sets else []


def _terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _run_child(command, *, env, console, deadline, timeout_kind, run_dir=None,
               startup_deadline=None, max_updates=None):
    process = subprocess.Popen(command, env=env, stdout=console, stderr=subprocess.STDOUT,
                               start_new_session=True)
    first_update = False
    last_count = 0
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise PilotTimeout(timeout_kind)
            if run_dir is not None and not first_update:
                steps = common_updates(run_dir)
                first_update = bool(steps)
            if startup_deadline is not None and not first_update and now >= startup_deadline:
                raise PilotTimeout('startup')
            remaining = deadline-now
            if startup_deadline is not None and not first_update:
                remaining = min(remaining, startup_deadline-now)
            try:
                return process.wait(timeout=min(2.0, max(0.001, remaining)))
            except subprocess.TimeoutExpired:
                if run_dir is not None:
                    steps = common_updates(run_dir)
                    first_update = bool(steps)
                    if len(steps) != last_count:
                        last_count = len(steps)
                        print(f'Completed on all four GPUs: {last_count}/{max_updates} updates', flush=True)
    except BaseException:
        _terminate(process)
        raise


def run_pilot(config_path, cfg, *, gpus=4, dry_run=False, hourly_usd=None):
    """Run one fresh bounded pilot, always retaining a partial/final report."""
    if not cfg.get('pilot', {}).get('enabled'):
        raise ValueError('Use the pilot_g_4x5090.yaml configuration for this command')
    if gpus != 4:
        raise ValueError('This pilot requires exactly four GPU workers')
    if hourly_usd is not None and (not math.isfinite(hourly_usd) or hourly_usd < 0):
        raise ValueError('hourly_usd must be finite and nonnegative')
    run_dir = run_directory(cfg)
    preflight = [sys.executable, '-m', 'science_gspo_g.preflight', '--gpus', '4',
                 '--check-judge', '--output', str(run_dir/'preflight.json')]
    training = torchrun_command(config_path, gpus_per_node=4, resume='none', run_name=cfg['run_name'])
    print('Pilot batch:', batch_plan(cfg,4), flush=True)
    print('Preflight:', shlex.join(preflight), flush=True)
    print('Training:', shlex.join(training), flush=True)
    if dry_run:
        return None
    import fcntl
    from .pilot_report import write_pilot_report
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir/'.pilot-launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another pilot launcher owns this run directory') from None
        if any((run_dir/name).exists() for name in ('launcher_metrics.json','ranks','checkpoints','training_completed.json')):
            raise ValueError('Pilot timings require a fresh run; choose a new --run-name')
        atomic_json(run_dir/'launch_config.json',cfg)
        env=os.environ.copy()
        env.update(WANDB_MODE='disabled',WANDB_DISABLED='true')
        env.setdefault('TOKENIZERS_PARALLELISM','false')
        env.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
        env.setdefault('OMP_NUM_THREADS','4')
        started=time.monotonic()
        overall_deadline=started+cfg['launch']['timeout_minutes']*60
        meta={'session_id':uuid.uuid4().hex,'started_at_utc':datetime.now(timezone.utc).isoformat(),
              'status':'starting','phase':'preflight','returncode':None,'hourly_usd':hourly_usd,
              'preflight_seconds':None,'training_process_seconds':None,'timeout_kind':None}
        atomic_json(run_dir/'launcher_metrics.json',meta)
        def interrupted(_signal,_frame):
            raise KeyboardInterrupt
        previous=signal.signal(signal.SIGTERM,interrupted)
        error=None
        phase_started=started
        try:
            print(f'Pilot log: {run_dir / "pilot_console.log"}',flush=True)
            with (run_dir/'pilot_console.log').open('a',buffering=1) as console:
                code=_run_child(preflight,env=env,console=console,
                    deadline=min(overall_deadline,started+cfg['pilot']['preflight_timeout_seconds']),
                    timeout_kind='preflight')
                meta.update(preflight_seconds=time.monotonic()-phase_started,returncode=code)
                if code:
                    meta['status']='preflight_failed'
                    raise subprocess.CalledProcessError(code,preflight)
                meta.update(status='training',phase='training',returncode=None)
                atomic_json(run_dir/'launcher_metrics.json',meta)
                phase_started=time.monotonic()
                code=_run_child(training,env=env,console=console,deadline=overall_deadline,
                    timeout_kind='overall',run_dir=run_dir,
                    startup_deadline=min(overall_deadline,phase_started+cfg['pilot']['startup_timeout_minutes']*60),
                    max_updates=cfg['pilot']['max_updates'])
                meta.update(returncode=code,status='completed' if code==0 else 'training_failed')
                if code:
                    raise subprocess.CalledProcessError(code,training)
        except PilotTimeout as exc:
            meta.update(status='timed_out',timeout_kind=exc.kind)
            error=exc
        except KeyboardInterrupt as exc:
            meta['status']='interrupted'
            error=exc
        except BaseException as exc:
            if meta['status'] not in ('preflight_failed','training_failed'):
                meta['status']='launcher_failed'
            meta['error_type']=type(exc).__name__
            error=exc
        finally:
            elapsed=time.monotonic()-phase_started
            meta['preflight_seconds' if meta['phase']=='preflight' else 'training_process_seconds']=elapsed
            meta.update(wall_seconds=time.monotonic()-started,finished_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(run_dir/'launcher_metrics.json',meta)
            signal.signal(signal.SIGTERM,previous)
            try:
                report=write_pilot_report(run_dir,cfg=cfg,world_size=4)
            except Exception as report_error:
                atomic_json(run_dir/'pilot_report_error.json',{'error_type':type(report_error).__name__})
                if error is None:
                    raise
                report=None
            print(f'Pilot report: {run_dir / "pilot_report.md"}',flush=True)
            print('Stop/delete the Vast instance separately to end compute and storage billing.',flush=True)
        if error is not None:
            raise error
        if not report or not report.get('success',False):
            raise RuntimeError('Pilot exited but did not pass all readiness checks; inspect pilot_report.md')
        return report

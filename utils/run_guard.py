"""Per-run limits for the benchmark: OOT (out of time) and OOM (out of memory).

Every benchmark entry point runs its training through :func:`run_guarded` (or
:func:`guarded_call`). A run is stopped and marked when it hits either limit:

* **OOT** - the run (data loading + training + evaluation) has been going for more
  than ``TIME_LIMIT_HOURS`` (default 4 h). A watchdog thread writes ``OOT.txt`` into
  the run folder and ends the process with exit code 124.
* **OOM** - the run needs more than ``VRAM_LIMIT_GB`` of GPU memory (default 100,
  counted in GiB = 1024**3 bytes, the unit PyTorch reports). The PyTorch allocator is
  capped at that size, so asking for more raises a CUDA out-of-memory error; the
  guard catches it, writes ``OOM.txt`` and exits with code 125. On a GPU with less
  memory than the limit the device itself is the cap, so the run is marked OOM as
  soon as it no longer fits on that GPU (``OOM.txt`` says so).

Each status file starts with a single line ``OOT`` or ``OOM`` followed by details.
No other result files are written for a stopped run, and stale status files are
removed when the same run is started again.

The limits can be changed for testing with environment variables
``GFW_TIME_LIMIT_HOURS`` and ``GFW_VRAM_LIMIT_GB``.
"""
from __future__ import print_function

import os
import sys
import threading
import time
from datetime import datetime

TIME_LIMIT_HOURS = float(os.environ.get('GFW_TIME_LIMIT_HOURS', 4))
VRAM_LIMIT_GB = float(os.environ.get('GFW_VRAM_LIMIT_GB', 100))
OOT_EXIT_CODE = 124
OOM_EXIT_CODE = 125
OOT_FILE = 'OOT.txt'
OOM_FILE = 'OOM.txt'
STATUS_FILES = (OOT_FILE, OOM_FILE)

_GIB = 1024 ** 3
_OOM_MARKERS = ('out of memory', 'cublas_status_alloc_failed', 'cudnn_status_alloc_failed',
                'cudaerrormemoryallocation', 'cuda_error_out_of_memory')


def is_oom_error(exc):
    """True for CUDA out-of-memory errors, including ones wrapped by other exceptions."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        try:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except (ImportError, AttributeError):
            pass
        msg = str(exc).lower()
        if any(marker in msg for marker in _OOM_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def stopped_status(out_dir):
    """'OOT' / 'OOM' if this run folder holds a status file, else ``None``."""
    for name in STATUS_FILES:
        if os.path.isfile(os.path.join(out_dir, name)):
            return name[:3]
    return None


def _gpu_index(gpu):
    if gpu is None:
        return 0
    try:
        idx = int(gpu)
    except (TypeError, ValueError):
        return 0
    return None if idx < 0 else idx


def cap_gpu_memory(gpu=0, limit_gb=None):
    """Cap PyTorch's allocator on ``cuda:<gpu>`` at ``limit_gb`` GiB; return a description."""
    limit_gb = VRAM_LIMIT_GB if limit_gb is None else limit_gb
    idx = _gpu_index(gpu)
    try:
        import torch
    except ImportError:
        return None
    if idx is None or not torch.cuda.is_available() or idx >= torch.cuda.device_count():
        return None
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / _GIB
    info = dict(device=idx, name=props.name, total_gb=total_gb, limit_gb=limit_gb)
    if total_gb > limit_gb:
        torch.cuda.set_per_process_memory_fraction(limit_gb / total_gb, idx)
        info['cap_gb'] = limit_gb
    else:
        info['cap_gb'] = total_gb  # the device is smaller than the limit
    return info


def _peak_vram_gb(gpu_info):
    if not gpu_info:
        return None
    try:
        import torch
        return torch.cuda.max_memory_reserved(gpu_info['device']) / _GIB
    except Exception:  # pragma: no cover
        return None


class _Guard(object):
    def __init__(self, out_dir, run_info, gpu, time_limit_hours, vram_limit_gb):
        self.out_dir = out_dir
        self.run_info = run_info or {}
        self.gpu = gpu
        self.time_limit_hours = TIME_LIMIT_HOURS if time_limit_hours is None else time_limit_hours
        self.vram_limit_gb = VRAM_LIMIT_GB if vram_limit_gb is None else vram_limit_gb
        self.gpu_info = None
        self.started = None
        self._lock = threading.Lock()
        self._done = False
        self._timer = None

    def _write(self, status, headline, extra):
        with self._lock:
            if self._done:
                return None
            self._done = True
        os.makedirs(self.out_dir, exist_ok=True)
        elapsed = time.time() - self.started
        lines = [status, headline, '']
        for key, value in self.run_info.items():
            lines.append('{}: {}'.format(key, value))
        lines.append('time_limit_hours: {:g}'.format(self.time_limit_hours))
        lines.append('vram_limit_gb: {:g}'.format(self.vram_limit_gb))
        lines.append('elapsed_hours: {:.3f}'.format(elapsed / 3600.0))
        if self.gpu_info:
            lines.append('gpu: {} ({:.1f} GiB, allocator cap {:.1f} GiB)'.format(
                self.gpu_info['name'], self.gpu_info['total_gb'], self.gpu_info['cap_gb']))
        else:
            lines.append('gpu: none (CPU run)')
        lines.extend(extra)
        lines.append('started: {}'.format(datetime.fromtimestamp(self.started).isoformat(timespec='seconds')))
        lines.append('stopped: {}'.format(datetime.now().isoformat(timespec='seconds')))
        lines.append('command: {} {}'.format(os.path.basename(sys.executable), ' '.join(sys.argv)))
        path = os.path.join(self.out_dir, status + '.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        return path

    def _on_timeout(self):
        path = self._write(
            'OOT', 'Out of time: the run took longer than {:g} h and was stopped.'.format(
                self.time_limit_hours), [])
        if path is None:
            return
        msg = '\n!!! OOT: run exceeded {:g} h; stopped. Wrote {}\n'.format(self.time_limit_hours, path)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.write(msg)
                stream.flush()
            except Exception:  # pragma: no cover
                pass
        os._exit(OOT_EXIT_CODE)

    def _on_oom(self, exc):
        extra = []
        peak = _peak_vram_gb(self.gpu_info)
        if peak is not None:
            extra.append('peak_reserved_gb_before_failure: {:.2f}'.format(peak))
        if self.gpu_info and self.gpu_info['total_gb'] <= self.vram_limit_gb:
            extra.append('note: this GPU has only {:.1f} GiB (< {:g} GB limit), so the run was '
                         'stopped when it no longer fit on the device. Use a GPU with more than '
                         '{:g} GB for an exact {:g} GB criterion.'.format(
                             self.gpu_info['total_gb'], self.vram_limit_gb,
                             self.vram_limit_gb, self.vram_limit_gb))
        first_line = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        extra.append('error: {}: {}'.format(type(exc).__name__, first_line[:500]))
        path = self._write(
            'OOM', 'Out of memory: the run needed more than {:g} GB of GPU memory and was '
                   'stopped.'.format(min(self.vram_limit_gb, self.gpu_info['cap_gb'])
                                     if self.gpu_info else self.vram_limit_gb), extra)
        if path is not None:
            print('\n!!! OOM: run needed more GPU memory than allowed; stopped. Wrote {}'.format(path),
                  file=sys.stderr)

    def __call__(self, fn):
        for name in STATUS_FILES:  # stale marks from an earlier attempt of the same run
            stale = os.path.join(self.out_dir, name)
            if os.path.isfile(stale):
                os.remove(stale)
        self.gpu_info = cap_gpu_memory(self.gpu, self.vram_limit_gb)
        if self.gpu_info:
            print('Run limits: {:g} h, GPU memory cap {:.1f} GiB on {}'.format(
                self.time_limit_hours, self.gpu_info['cap_gb'], self.gpu_info['name']))
        else:
            print('Run limits: {:g} h (no GPU)'.format(self.time_limit_hours))
        self.started = time.time()
        self._timer = threading.Timer(self.time_limit_hours * 3600.0, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()
        try:
            return fn()
        except Exception as exc:  # pylint: disable=broad-except
            if not is_oom_error(exc):
                raise
            self._timer.cancel()
            self._on_oom(exc)
            sys.exit(OOM_EXIT_CODE)
        finally:
            self._timer.cancel()


def guarded_call(fn, out_dir, run_info=None, gpu=0, time_limit_hours=None, vram_limit_gb=None):
    """Call ``fn()`` under the OOT / OOM limits; status files go to ``out_dir``."""
    return _Guard(out_dir, run_info, gpu, time_limit_hours, vram_limit_gb)(fn)


def run_guarded(train_fn, args, model_name, run_tag=None):
    """Run ``train_fn(args)`` for a benchmark ``*_main.py`` under the OOT / OOM limits."""
    from utils.evaluation import run_dir_from_args
    out_dir = run_dir_from_args(args, model_name, run_tag=run_tag)
    info = dict(model=model_name, dataset=os.path.basename(os.path.dirname(out_dir)),
                seed=getattr(args, 'seed', None), train_ratio=getattr(args, 'train_ratio', None),
                run_dir=out_dir)
    return guarded_call(lambda: train_fn(args), out_dir, info, gpu=getattr(args, 'gpu', 0))

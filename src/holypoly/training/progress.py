"""Stage visibility and opt-in, bounded CUDA-aware batch timing."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

import torch


def _synchronize(device: torch.device) -> None:
    """Finish queued CUDA work only at explicit timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed_stage(
    label: str, device: torch.device, durations: dict[str, float]
) -> Iterator[None]:
    """Print stage boundaries immediately and record elapsed wall-clock seconds."""
    _synchronize(device)
    print(f"[{label}] started", file=sys.stderr, flush=True)
    started = perf_counter()
    try:
        yield
    except BaseException:
        print(f"[{label}] failed", file=sys.stderr, flush=True)
        raise
    else:
        _synchronize(device)
        durations[label] = perf_counter() - started
        print(
            f"[{label}] finished in {durations[label]:.1f}s",
            file=sys.stderr,
            flush=True,
        )


class BatchProfiler:
    """Time a bounded number of batches after warmup; normal training never syncs.

    Opt-in profiling synchronizes each measured section, so these wall-clock
    timings are diagnostic and include synchronization overhead. They must not
    be interpreted as uninstrumented throughput or summed with nested timings.
    """

    def __init__(
        self, device: torch.device, batches: int = 0, warmup_batches: int = 5
    ) -> None:
        if batches < 0 or warmup_batches < 0:
            raise ValueError("profile batches and warmup must be nonnegative")
        self.device = device
        self.limit = batches
        self.warmup = warmup_batches
        self.index = -1
        self.measured = 0
        self.seconds: dict[str, float] = {}

    @property
    def active(self) -> bool:
        """Whether the current batch belongs to the requested timing window."""
        return self.warmup <= self.index < self.warmup + self.limit

    def start_batch(self) -> None:
        """Advance the window once for each actual batch."""
        self.index += 1
        if self.active:
            self.measured += 1

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Record one section, with CUDA boundaries only inside the window."""
        if not self.active:
            yield
            return
        _synchronize(self.device)
        started = perf_counter()
        try:
            yield
        finally:
            _synchronize(self.device)
            self.seconds[name] = self.seconds.get(name, 0.0) + perf_counter() - started

    def report(self) -> dict[str, object]:
        """Return and print per-batch means, explicitly identifying timing method."""
        result: dict[str, object] = {
            "measured_batches": self.measured,
            "warmup_batches": self.warmup,
            "method": "synchronized_wall_clock",
            "seconds_per_batch": {
                name: seconds / self.measured
                for name, seconds in self.seconds.items()
                if self.measured
            },
        }
        if self.limit:
            print("Batch profile: " + json.dumps(result), file=sys.stderr, flush=True)
        return result

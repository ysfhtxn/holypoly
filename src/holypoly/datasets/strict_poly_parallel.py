"""Bounded CPU processes for strict Poly QC, with ordered, quota-safe read loading.

Only the coordinator reads POD5 or writes results. Workers receive one decoded
signal at a time. Reserving remaining per-stratum slots prevents speculative
signal reads beyond the serial early-stop boundary.
"""

from __future__ import annotations

import multiprocessing
import os
from collections import Counter, deque
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

SourceRow = dict[str, str]
Stratum = tuple[int, str, str]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class ImmediateSignalResult:
    """A metadata-only decision that needs neither decoded signal nor a worker."""

    value: Any


@contextmanager
def ordered_signal_jobs(
    sources: Sequence[SourceRow],
    prepare: Callable[[SourceRow], tuple[Any, ...] | ImmediateSignalResult],
    analyze: Callable[..., Any],
    remaining_quota: Callable[[SourceRow], int | None],
    workers: int,
) -> Iterator[Iterator[tuple[SourceRow, Future[Any]]]]:
    """Yield jobs in source order, with at most workers prepared/in-flight reads.

    Preparation runs in the coordinator. Exceptions remain attached to their
    source row for ordinary per-read auditing. Outstanding rows reserve quota
    even when preparation fails, until the consumer commits them in order.
    A full stratum yields a sentinel future without preparing any signal.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    pool = None
    pending: deque[tuple[SourceRow, Future[Any], Stratum, bool]] = deque()
    reserved: Counter[Stratum] = Counter()

    def results() -> Generator[tuple[SourceRow, Future[Any]], None, None]:
        cursor = 0
        while cursor < len(sources) or pending:
            while cursor < len(sources) and len(pending) < workers:
                source = sources[cursor]
                key = (
                    int(float(source["nominal_length_nt"])),
                    source["orientation"],
                    source["split"],
                )
                remaining = remaining_quota(source)
                if remaining is not None and remaining < 0:
                    raise ValueError("remaining quota cannot be negative")
                if remaining and reserved[key] >= remaining:
                    break
                reserve = remaining is not None and remaining > 0
                if reserve:
                    reserved[key] += 1
                future: Future[Any] = Future()
                if remaining == 0:
                    future.set_result(None)
                else:
                    try:
                        arguments = prepare(source)
                    except Exception as error:
                        future.set_exception(error)
                    else:
                        if isinstance(arguments, ImmediateSignalResult):
                            future.set_result(arguments.value)
                        elif pool is None:
                            try:
                                future.set_result(analyze(*arguments))
                            except Exception as error:
                                future.set_exception(error)
                        else:
                            future = pool.submit(analyze, *arguments)
                pending.append((source, future, key, reserve))
                cursor += 1
            if not pending:
                raise RuntimeError("quota reservations blocked an empty job queue")
            source, future, key, reserve = pending.popleft()
            if reserve:
                reserved[key] -= 1
            yield source, future

    iterator = results()
    try:
        if workers > 1:
            # Keep these settings until shutdown: spawn starts processes lazily,
            # and NumPy must see them before its import in every new child.
            for name in THREAD_VARIABLES:
                os.environ[name] = "1"
            pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
        yield iterator
    finally:
        try:
            iterator.close()
            for _, future, _, _ in pending:
                future.cancel()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            if workers > 1:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

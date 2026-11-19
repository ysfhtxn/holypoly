"""Real spawned CPU processes; synthetic jobs only, without POD5 or Torch."""

from __future__ import annotations

import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures.process import BrokenProcessPool

import pytest

from holypoly.datasets.strict_poly_parallel import (
    ImmediateSignalResult,
    ordered_signal_jobs,
)


def _analyze(index, delay, fail=False):
    """Finish with varying delays to exercise ordered result consumption."""
    time.sleep(delay)
    if fail:
        raise ValueError("synthetic worker failure")
    return index, os.getpid(), os.environ.get("OMP_NUM_THREADS")


def _sources(count):
    return [
        {
            "read_id": str(index),
            "nominal_length_nt": str(20 if index % 2 == 0 else 40),
            "orientation": "forward",
            "split": "train",
        }
        for index in range(count)
    ]


def _crash_worker():
    """Terminate only this synthetic test's child to check hard failure handling."""
    os._exit(17)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_results_are_ordered_reads_stay_in_parent_and_threads_restore(
    workers, monkeypatch
):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    parent = os.getpid()
    prepared = []

    def prepare(source):
        assert os.getpid() == parent
        index = int(source["read_id"])
        prepared.append(index)
        return index, (8 - index) * 0.002

    with ordered_signal_jobs(
        _sources(8), prepare, _analyze, lambda _: None, workers
    ) as jobs:
        results = [future.result() for _, future in jobs]
    assert prepared == list(range(8))
    assert [result[0] for result in results] == list(range(8))
    assert all((result[1] == parent) == (workers == 1) for result in results)
    assert all(result[2] == ("7" if workers == 1 else "1") for result in results)
    assert os.environ["OMP_NUM_THREADS"] == "7"


def _run_with_quota(workers):
    accepted = Counter()
    prepared, decisions = [], []

    def remaining(source):
        return 2 - accepted[source["nominal_length_nt"]]

    def prepare(source):
        index = int(source["read_id"])
        prepared.append(index)
        if index == 2:
            raise ValueError("synthetic preparation failure")
        return index, 0.002 * (8 - index), index == 4

    with ordered_signal_jobs(
        _sources(20), prepare, _analyze, remaining, workers
    ) as jobs:
        for source, future in jobs:
            if not remaining(source):
                decisions.append((source["read_id"], "skipped"))
                assert future.result() is None
                continue
            try:
                index, _, _ = future.result()
                status = "accepted" if index != 0 else "rejected"
            except ValueError:
                status = "error"
            decisions.append((source["read_id"], status))
            if status == "accepted":
                accepted[source["nominal_length_nt"]] += 1
            if all(accepted[key] == 2 for key in ["20", "40"]):
                break
    return prepared, decisions, dict(accepted)


@pytest.mark.parametrize("workers", [2, 4])
def test_quota_reservations_preserve_serial_reads_and_decisions_with_failures(workers):
    serial = _run_with_quota(1)
    parallel = _run_with_quota(workers)
    assert parallel == serial
    assert parallel[0] == [0, 1, 2, 3, 4, 6, 8]
    assert parallel[2] == {"40": 2, "20": 2}


def test_exception_exit_joins_children_and_restores_thread_settings(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    children = {child.pid for child in multiprocessing.active_children()}
    with (
        pytest.raises(ValueError, match="worker failure"),
        ordered_signal_jobs(
            _sources(8), lambda _: (0, 0.002, True), _analyze, lambda _: None, 2
        ) as jobs,
    ):
        for _, future in jobs:
            future.result()
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert {child.pid for child in multiprocessing.active_children()} <= children


def test_nonpositive_worker_count_is_rejected():
    with (
        pytest.raises(ValueError, match="workers"),
        ordered_signal_jobs([], lambda _: (), _analyze, lambda _: None, 0),
    ):
        pass


def test_metadata_only_results_do_not_run_analysis_or_start_children():
    children = {child.pid for child in multiprocessing.active_children()}
    with ordered_signal_jobs(
        _sources(4),
        lambda source: ImmediateSignalResult(source["read_id"]),
        _crash_worker,
        lambda _: None,
        2,
    ) as jobs:
        values = [future.result() for _, future in jobs]
    assert values == ["0", "1", "2", "3"]
    assert {child.pid for child in multiprocessing.active_children()} <= children


def test_crashed_worker_propagates_instead_of_becoming_a_negative_label(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    children = {child.pid for child in multiprocessing.active_children()}
    with (
        pytest.raises(BrokenProcessPool),
        ordered_signal_jobs(
            _sources(4), lambda _: (), _crash_worker, lambda _: None, 2
        ) as jobs,
    ):
        for _, future in jobs:
            future.result()
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert {child.pid for child in multiprocessing.active_children()} <= children

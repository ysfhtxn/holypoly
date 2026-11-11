"""Ordered prefetch, bounded queues, and deterministic producer cleanup."""

from collections.abc import Iterator
from threading import Event, current_thread
from threading import enumerate as threads

import pytest

from holypoly.training.prefetch import prefetch_batches


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_order_tail_none_and_repeated_iteration(depth: int) -> None:
    source = [0, None, 2, 3, 4]
    with prefetch_batches(source, depth) as loader:
        assert len(loader) == 5
        assert list(loader) == source
        assert list(loader) == source
    with prefetch_batches([], depth) as loader:
        assert list(loader) == []


def test_prefetch_is_bounded_and_releases_partial_iterator() -> None:
    prepared, closed = Event(), Event()
    seen = []

    def source() -> Iterator[int]:
        try:
            for index in range(100):
                seen.append(index)
                if index == 2:
                    prepared.set()
                yield index
        finally:
            closed.set()

    before = {item.ident for item in threads()}
    with prefetch_batches(source(), depth=2) as loader:
        iterator = iter(loader)
        assert prepared.wait(5), "producer did not prepare its bounded queue"
        assert seen == [0, 1, 2]  # Two queued values and one in flight.
        assert next(iterator) == 0
    assert closed.is_set()
    assert {item.ident for item in threads()} == before


def test_next_batch_preparation_overlaps_consumer_work() -> None:
    ready = Event()
    producer_threads = []

    def source() -> Iterator[int]:
        producer_threads.append(current_thread().ident)
        yield 0
        ready.set()
        yield 1

    with prefetch_batches(source(), depth=1) as loader:
        iterator = iter(loader)
        assert next(iterator) == 0
        assert ready.wait(5), "next batch was not prepared while consumer held batch 0"
        assert len(producer_threads) == 1
        assert producer_threads[0] != current_thread().ident
        assert next(iterator) == 1


@pytest.mark.parametrize("depth", [0, 2])
def test_reader_error_preserves_type_order_and_cleanup(depth: int) -> None:
    closed = Event()

    def source() -> Iterator[int]:
        try:
            yield 0
            yield 1
            raise OSError("synthetic POD5 failure")
        finally:
            closed.set()

    before = {item.ident for item in threads()}
    with (
        pytest.raises(OSError, match="synthetic POD5 failure"),
        prefetch_batches(source(), depth) as loader,
    ):
        iterator = iter(loader)
        assert next(iterator) == 0
        assert next(iterator) == 1
        next(iterator)
    assert closed.is_set()
    assert {item.ident for item in threads()} == before


def test_consumer_error_closes_producer_and_sync_mode_stays_on_main_thread() -> None:
    closed = Event()

    def source() -> Iterator[int]:
        try:
            yield from range(100)
        finally:
            closed.set()

    before = {item.ident for item in threads()}
    with (
        pytest.raises(RuntimeError, match="synthetic model failure"),
        prefetch_batches(source(), depth=1) as loader,
    ):
        assert next(iter(loader)) == 0
        raise RuntimeError("synthetic model failure")
    assert closed.is_set()
    assert {item.ident for item in threads()} == before
    seen = []

    def synchronous() -> Iterator[int]:
        seen.append(current_thread().ident)
        yield 0

    with prefetch_batches(synchronous(), depth=0) as loader:
        assert list(loader) == [0]
    assert seen == [current_thread().ident]


def test_negative_depth_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"), prefetch_batches([], -1):
        pass

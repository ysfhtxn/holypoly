"""Bounded, ordered CPU prefetch with explicit iterator and thread cleanup."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from contextlib import contextmanager
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Generic, TypeVar, cast

Item = TypeVar("Item")


class _PrefetchIterator(Iterator[Item]):
    def __init__(self, source: Iterator[Item], depth: int) -> None:
        self._source = source
        self._queue: Queue[tuple[str, object]] = Queue(maxsize=depth)
        self._cancel = Event()
        self._thread = Thread(
            target=self._produce, name="holypoly-cpu-prefetch", daemon=True
        )
        self._thread.start()

    def _put(self, kind: str, value: object = None) -> None:
        while not self._cancel.is_set():
            try:
                self._queue.put((kind, value), timeout=0.05)
                return
            except Full:
                continue

    def _produce(self) -> None:
        try:
            while not self._cancel.is_set():
                try:
                    value = next(self._source)
                except StopIteration:
                    break
                self._put("value", value)
        except BaseException as error:
            self._put("error", error)
        finally:
            try:
                close = getattr(self._source, "close", None)
                if close is not None:
                    close()
            except BaseException as error:
                self._put("error", error)
            self._put("done")

    def __next__(self) -> Item:
        if self._cancel.is_set():
            raise StopIteration
        kind, value = self._queue.get()
        if kind == "value":
            return cast(Item, value)
        self.close()
        if kind == "error":
            raise cast(BaseException, value)
        raise StopIteration

    def close(self) -> None:
        """Cancel queued work and finish the current read before releasing buffers."""
        self._cancel.set()
        self._thread.join()
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break


class _PrefetchLoader(Iterable[Item], Generic[Item]):
    def __init__(self, source: Iterable[Item], depth: int) -> None:
        self._source = source
        self._depth = depth
        self._active: _PrefetchIterator[Item] | None = None

    def __len__(self) -> int:
        return len(cast(Sized, self._source))

    def __iter__(self) -> Iterator[Item]:
        self.close()
        # DataLoader iterator initialization stays in the main thread, preserving
        # the original order of base-seed consumption. Only next() runs off-thread.
        self._active = _PrefetchIterator(iter(self._source), self._depth)
        return self._active

    def close(self) -> None:
        """Join the producer even when sequence batches are only partly consumed."""
        if self._active is not None:
            self._active.close()
            self._active = None


@contextmanager
def prefetch_batches(
    source: Iterable[Item], depth: int = 2
) -> Iterator[Iterable[Item]]:
    """Overlap ordered CPU preparation with consumption; depth=0 stays synchronous.

    At most depth batches are queued, plus one being prepared and the consumer's
    current batch. No CUDA work, sampling policy, or tensor conversion is added.
    Exit joins the producer, so callers can safely close POD5 readers afterward.
    """
    if depth < 0:
        raise ValueError("prefetch depth must be nonnegative")
    if depth == 0:
        yield source
        return
    loader = _PrefetchLoader(source, depth)
    try:
        yield loader
    finally:
        loader.close()

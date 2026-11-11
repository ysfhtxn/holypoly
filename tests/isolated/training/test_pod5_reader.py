"""Synthetic tests for bounded POD5 reader reuse without biological data."""

import inspect
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import numpy as np
import pod5
import pytest
import torch

from holypoly.training.data import Pod5SignalReader


@pytest.fixture
def reader_factory(tmp_path: Path, monkeypatch: Any) -> Any:
    """Track file lifetimes and emulate read selection and calibrated samples."""
    opened: list[Any] = []

    class Reader:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.closed = False
            opened.append(self)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

        def close(self) -> None:
            assert not self.closed
            self.closed = True

        def reads(self, **kwargs: Any) -> list[Any]:
            assert not self.closed
            assert kwargs["preload"] == {"samples"}
            assert kwargs["missing_ok"] is True
            read_id = kwargs["selection"][0]
            if read_id == "io-error":
                raise OSError("synthetic read error")
            read = SimpleNamespace(
                signal=np.array([-4, 0, 8], dtype=np.int16),
                calibration=SimpleNamespace(scale=0.5, offset=2),
            )
            return (
                []
                if read_id == "missing"
                else [read] * (2 if read_id == "duplicate" else 1)
            )

    monkeypatch.setattr(pod5, "Reader", Reader)
    paths = [tmp_path / f"synthetic-{index}.pod5" for index in range(3)]
    for path in paths:
        path.touch()
    return opened, paths


@pytest.mark.parametrize("cache_size", [0, 1, 2])
def test_reader_reuse_preserves_normalization_and_closes_files(
    reader_factory: Any, cache_size: int
) -> None:
    """Caching changes open counts, while normalized float32 samples stay exact."""
    opened, paths = reader_factory
    expected = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32)
    with Pod5SignalReader(
        signal_shift=1, signal_scale=2, max_open_files=cache_size
    ) as reader:
        for _ in range(3):
            assert torch.equal(reader(paths[0], "valid"), expected)
        assert len(opened) == (3 if cache_size == 0 else 1)
        assert sum(not item.closed for item in opened) == (0 if cache_size == 0 else 1)
    assert all(item.closed for item in opened)
    reader.close()  # Explicit cleanup is idempotent.


def test_reader_cache_evicts_least_recently_used_file(reader_factory: Any) -> None:
    """Revisited readers stay open; evicted readers reopen within the hard limit."""
    opened, paths = reader_factory
    with Pod5SignalReader(signal_shift=0, signal_scale=1, max_open_files=2) as reader:
        for index in (0, 1, 0, 2):
            reader(paths[index], "valid")
            assert sum(not item.closed for item in opened) <= 2
        assert len(opened) == 3
        assert not opened[0].closed and opened[1].closed
        reader(paths[1], "valid")
        assert len(opened) == 4
        assert opened[0].closed
        assert sum(not item.closed for item in opened) == 2
    assert all(item.closed for item in opened)


@pytest.mark.parametrize("cache_size", [0, 2])
@pytest.mark.parametrize(
    ("read_id", "error", "message"),
    [
        ("missing", KeyError, "not found"),
        ("duplicate", ValueError, "occurred 2 times"),
        ("io-error", OSError, "synthetic read error"),
    ],
)
def test_read_failure_retains_error_and_closes_cache(
    reader_factory: Any,
    cache_size: int,
    read_id: str,
    error: type[Exception],
    message: str,
) -> None:
    """Missing, duplicate, and I/O failures keep their semantics and release handles."""
    opened, paths = reader_factory
    with (
        pytest.raises(error, match=message),
        Pod5SignalReader(
            signal_shift=0, signal_scale=1, max_open_files=cache_size
        ) as reader,
    ):
        reader(paths[0], "valid")
        reader(paths[1], read_id)
    assert all(item.closed for item in opened)


def test_cache_cleanup_after_consumer_failure(reader_factory: Any) -> None:
    """A model-side exception must also close handles held by the reader context."""
    opened, paths = reader_factory
    with (
        pytest.raises(RuntimeError, match="synthetic model failure"),
        Pod5SignalReader(signal_shift=0, signal_scale=1, max_open_files=2) as reader,
    ):
        reader(paths[0], "valid")
        raise RuntimeError("synthetic model failure")
    assert all(item.closed for item in opened)


def test_reader_rejects_negative_cache_and_missing_file(tmp_path: Path) -> None:
    """Invalid cache size and nonexistent files fail explicitly."""
    with pytest.raises(ValueError, match="max_open_files"):
        Pod5SignalReader(signal_shift=0, signal_scale=1, max_open_files=-1)
    with (
        Pod5SignalReader(signal_shift=0, signal_scale=1, max_open_files=2) as reader,
        pytest.raises(FileNotFoundError, match="does not exist"),
    ):
        reader(tmp_path / "missing.pod5", "unused")


def test_real_pod5_api_with_synthetic_signals(tmp_path: Path) -> None:
    """Repeated reads via actual POD5 handles match uncached signals byte for byte."""
    path = tmp_path / "synthetic.pod5"
    metadata = {
        name: "synthetic" for name in inspect.signature(pod5.RunInfo).parameters
    }
    metadata.update(
        acquisition_start_time=datetime(2026, 1, 1, tzinfo=UTC),
        protocol_start_time=datetime(2026, 1, 1, tzinfo=UTC),
        adc_max=8191,
        adc_min=-8192,
        context_tags={},
        tracking_id={},
        sample_rate=4000,
    )
    run_info = pod5.RunInfo(**metadata)
    signals = [np.arange(30 + index, dtype=np.int16) - 10 for index in range(3)]
    with pod5.Writer(path) as writer:
        writer.add_reads(
            [
                pod5.Read(
                    read_id=UUID(int=index + 1),
                    pore=pod5.Pore(channel=1, well=1, pore_type="synthetic"),
                    calibration=pod5.Calibration(offset=-5, scale=0.25),
                    read_number=index,
                    start_sample=index * 1000,
                    median_before=float("nan"),
                    end_reason=pod5.EndReason(pod5.EndReasonEnum.UNKNOWN, False),
                    run_info=run_info,
                    signal=signal,
                )
                for index, signal in enumerate(signals)
            ]
        )
    before = path.read_bytes()
    with (
        Pod5SignalReader(signal_shift=2, signal_scale=3) as uncached,
        Pod5SignalReader(signal_shift=2, signal_scale=3, max_open_files=2) as cached,
    ):
        for index in (2, 0, 1, 2):
            read_id = str(UUID(int=index + 1))
            result = cached(path, read_id)
            expected = (0.25 * (torch.tensor(signals[index]).float() - 5) - 2) / 3
            assert torch.equal(result, expected)
            assert torch.equal(result, uncached(path, read_id))
    assert path.read_bytes() == before

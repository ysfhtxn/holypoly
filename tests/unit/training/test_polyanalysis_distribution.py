"""Check unlabeled PolyAnalysis adaptation with synthetic reads and a fake model."""

import argparse
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/polyanalysis_distribution.py"
FUNCTIONS = runpy.run_path(str(SCRIPT))


@pytest.mark.parametrize(
    ("result", "status", "length"),
    [
        (None, "no_candidate", ""),
        (("synthetic", 0, 512, "polyA"), "polyA_proxy_available", 17),
        (("synthetic", 256, 1024, "polyA"), "polyA_proxy_available", 26),
        (("synthetic", 0, 512, "polyT"), "other_region_polyT", ""),
    ],
)
def test_adapter_keeps_missing_calls_and_other_directions_separate(
    result: tuple | None, status: str, length: int | str
) -> None:
    signal = object()
    record = SimpleNamespace(
        read_id="synthetic",
        num_samples=1024,
        signal=signal,
        run_info=SimpleNamespace(sample_rate=5000),
    )
    original_reader = object()
    native = SimpleNamespace(extract_signal_from_fast5=original_reader)

    def process(path, model, device, window, stride, batch):
        assert native.extract_signal_from_fast5(path) is signal
        assert (window, stride, batch) == (512, 256, 128)
        return result

    native.process_fast5_file = process
    row = FUNCTIONS["classify_record"](record, native, None, "cpu")
    assert row["status"] == status
    assert row["length_proxy_nt"] == length
    assert row["label_source"] == "unlabeled"
    assert native.extract_signal_from_fast5 is original_reader
    if result:
        assert row["interval_ms"] == (result[2] - result[1]) / 5


@pytest.mark.parametrize("start,end", [(-1, 512), (512, 512), (0, 1025)])
def test_adapter_rejects_invalid_native_coordinates(start: int, end: int) -> None:
    record = SimpleNamespace(read_id="synthetic", num_samples=1024, signal=object())
    native = SimpleNamespace(
        extract_signal_from_fast5=None,
        process_fast5_file=lambda *args: ("synthetic", start, end, "polyA"),
    )
    with pytest.raises(ValueError, match="out-of-bounds"):
        FUNCTIONS["classify_record"](record, native, None, "cpu")


def test_output_preflight_does_not_create_or_overwrite_files(tmp_path: Path) -> None:
    data, work = tmp_path / "raw", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    pod5 = data / "synthetic.pod5"
    pod5.touch()
    output = work / "analysis"
    args = argparse.Namespace(
        input=data, work_root=work, data_root=data, output_dir=output, max_reads=1000
    )
    assert FUNCTIONS["prepare_paths"](args) == ([pod5], output)
    assert not output.exists()
    for forbidden in (data / "analysis", work, pod5):
        args.output_dir = forbidden
        with pytest.raises(ValueError, match="new external work-root"):
            FUNCTIONS["prepare_paths"](args)
    output.mkdir()
    args.output_dir = output
    with pytest.raises(ValueError, match="new external work-root"):
        FUNCTIONS["prepare_paths"](args)

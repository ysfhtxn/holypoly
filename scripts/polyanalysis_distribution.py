"""Stream unlabeled POD5 reads through the installed PolyAnalysis cDNA detector.

Run with the polyanalysis environment, not the parent project environment.
Lengths reproduce the installed infer CLI's uncalibrated interval/30 rule;
this command does not generate or run the unused sequence inference output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from unittest.mock import patch


def utc_now() -> str:
    """Return a UTC timestamp under the PolyAnalysis Python 3.10 runtime."""
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def prepare_paths(args: argparse.Namespace) -> tuple[list[Path], Path]:
    """Validate a new external output path before importing model dependencies."""
    source = args.input.resolve(strict=True)
    work = args.work_root.resolve(strict=True)
    data = args.data_root.resolve(strict=True)
    output = args.output_dir.absolute()
    project = Path(__file__).resolve().parents[1]
    if args.max_reads < 0:
        raise ValueError("max-reads must be >= 0 (0 explicitly selects all reads)")
    if (
        output.is_symlink()
        or output.exists()
        or not output.resolve().is_relative_to(work)
        or output.resolve() == work
        or output.resolve().is_relative_to(data)
        or output.resolve().is_relative_to(project)
        or source == output.resolve()
        or (source.is_dir() and output.resolve().is_relative_to(source))
    ):
        raise ValueError("output must be a new external work-root path outside inputs")
    files = (
        [source]
        if source.is_file() and source.suffix.lower() == ".pod5"
        else sorted(source.glob("*.pod5"))
        if source.is_dir()
        else []
    )
    if not files:
        raise ValueError(
            "No POD5 files at input (only the immediate directory is used)"
        )
    return files, output.resolve()


def classify_record(record, native, model, device: str) -> dict:
    """Adapt only signal I/O; preserve native cDNA windowing and region selection.

    Coordinates are int raw samples, relative to the read, 0-based half-open.
    No candidate remains undetermined rather than becoming a zero-nt call.
    """
    row = {
        "read_id": str(record.read_id),
        "raw_signal_length": record.num_samples,
        "start_sample": "",
        "end_sample": "",
        "interval_samples": "",
        "interval_ms": "",
        "detector_region_type": "",
        "length_proxy_nt": "",
        "confidence": "",
        "status": "no_candidate",
        "failure_reason": "native_detector_no_polyA_or_polyT_candidate",
        "label_source": "unlabeled",
    }
    with patch.object(native, "extract_signal_from_fast5", return_value=record.signal):
        result = native.process_fast5_file(
            row["read_id"] + ".fast5", model, device, 512, 256, 128
        )
    if result is None:
        return row
    _, start, end, kind = result
    if not (0 <= start < end <= record.num_samples):
        raise ValueError("Native detector returned an out-of-bounds interval")
    if kind not in {"polyA", "polyT"}:
        raise ValueError("Unknown native region type")
    row.update(
        start_sample=int(start),
        end_sample=int(end),
        interval_samples=int(end - start),
        interval_ms=(end - start) * 1000 / record.run_info.sample_rate,
        detector_region_type=kind,
        status="polyA_proxy_available" if kind == "polyA" else "other_region_polyT",
        failure_reason="" if kind == "polyA" else "native_detector_selected_polyT",
    )
    if kind == "polyA":
        row["length_proxy_nt"] = max(1, round((end - start) / 30.0))
    return row


def export_summary(output: Path, counts: Counter, lengths: Counter) -> None:
    """Export a candidate-conditioned histogram/CDF and full status denominators."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total, available = sum(counts.values()), sum(lengths.values())
    with (output / "length_proxy_distribution.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["length_proxy_nt", "read_count", "fraction_of_polyA_calls"])
        for value, number in sorted(lengths.items()):
            writer.writerow([value, number, number / available])
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    if lengths:
        xs = sorted(lengths)
        axes[0].bar(xs, [lengths[x] for x in xs], width=5, color="#517FA2")
        cumulative, ys = 0, []
        for x in xs:
            cumulative += lengths[x]
            ys.append(cumulative / available)
        axes[1].step([0, *xs], [0, *ys], where="post", color="#517FA2")
    else:
        for axis in axes:
            axis.text(
                0.5, 0.5, "No poly(A) candidates", ha="center", transform=axis.transAxes
            )
    for axis in axes:
        axis.set(xlabel="Native length proxy (nt; fixed 30 samples/nt)", xlim=(0, None))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Candidate reads")
    axes[1].set(ylabel="Cumulative fraction of poly(A) calls", ylim=(0, 1.02))
    figure.suptitle(f"PolyAnalysis poly(A) candidates: {available:,}/{total:,} reads")
    figure.supxlabel(
        "Uncalibrated cDNA output. No candidate is not 0 nt; no ground-truth accuracy.",
        fontsize=9,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"polyA_distribution.{suffix}", dpi=200)
    plt.close(figure)


def main() -> None:
    """Plan without loading signals, or explicitly run a bounded/full analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-reads", type=int, default=1000)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    files, output = prepare_paths(args)
    import pod5

    inputs = []
    for path in files:
        with pod5.Reader(path) as reader:
            inputs.append(
                {
                    "path": str(path),
                    "reads": reader.num_reads,
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            )
    total = sum(item["reads"] for item in inputs)
    selected = min(total, args.max_reads) if args.max_reads else total
    plan = {
        "inputs": inputs,
        "available_reads": total,
        "selected_reads": selected,
        "input_bytes": sum(item["bytes"] for item in inputs),
        "output": str(output),
        "max_reads": args.max_reads,
        "selection": "first reads in sorted file/physical read order; not random",
        "length_method": "max(1, round(interval_samples/30)); uncalibrated",
        "coordinates": "read-relative raw samples; 0-based half-open",
        "notes": [
            "No candidate is not zero nt",
            "No reference labels used",
            "Native cDNA detector; chemistry applicability unvalidated",
            "Sequence inference skipped; its output does not set CLI length",
        ],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib-cache")
    os.environ["CUDA_CACHE_PATH"] = str(output / "cuda-cache")
    os.environ["XDG_CACHE_HOME"] = str(output / "cache")
    import numpy as np
    import polyanalysis
    import torch
    from polyanalysis.cnn_lstm_model import cDNA_infer_polya_polyt_from_fast5 as native

    if args.device.startswith("cuda") and (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one available GPU")
    random.seed(20261103)
    np.random.seed(20261103)
    torch.manual_seed(20261103)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_file = Path(polyanalysis.__file__).parent / "models/DNA_signal_model.pt"
    model = native.MultiClassModel(input_len=512, feature_dim=8).to(args.device)
    model.load_state_dict(
        torch.load(model_file, map_location=args.device, weights_only=True)
    )
    model.eval()
    plan.update(
        started_utc=utc_now(),
        seed=20261103,
        command=sys.argv,
        model=str(model_file),
        model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
        native_source_sha256=hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest(),
        python=sys.version,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[1]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        device=args.device,
        gpu=torch.cuda.get_device_name(0) if args.device == "cuda:0" else None,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "polyanalysis",
                "torch",
                "pod5",
                "numpy",
                "scipy",
                "matplotlib",
            )
        },
        window_samples=512,
        stride_samples=256,
        batch_size=128,
    )
    (output / "run_metadata.json").write_text(json.dumps(plan, indent=2) + "\n")
    counts, lengths, seen = Counter(), Counter(), set()
    started = time.monotonic()
    with (output / "per_read_results.tsv").open("w", newline="") as stream:
        fields = [
            "read_id",
            "source_pod5",
            "raw_signal_length",
            "start_sample",
            "end_sample",
            "interval_samples",
            "interval_ms",
            "detector_region_type",
            "length_proxy_nt",
            "confidence",
            "status",
            "failure_reason",
            "label_source",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for path in files:
            with pod5.Reader(path) as reader:
                remaining = selected - sum(counts.values())
                for record in islice(reader.reads(), remaining):
                    read_id = str(record.read_id)
                    if read_id in seen:
                        raise ValueError("Duplicate read_id across selected POD5 reads")
                    seen.add(read_id)
                    try:
                        row = classify_record(record, native, model, args.device)
                    except torch.cuda.OutOfMemoryError:
                        raise
                    except Exception as exc:
                        row = {
                            "read_id": read_id,
                            "status": "failure",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "label_source": "unlabeled",
                        }
                    row["source_pod5"] = str(path)
                    writer.writerow(row)
                    counts[row["status"]] += 1
                    if row["status"] == "polyA_proxy_available":
                        lengths[row["length_proxy_nt"]] += 1
                    if sum(counts.values()) % 100 == 0:
                        stream.flush()
                        print(
                            json.dumps(
                                {
                                    "processed": sum(counts.values()),
                                    "selected": selected,
                                    "status": dict(counts),
                                    "elapsed_seconds": time.monotonic() - started,
                                }
                            ),
                            flush=True,
                        )
            if sum(counts.values()) == selected:
                break
    if sum(counts.values()) != selected:
        raise RuntimeError("POD5 metadata/read count mismatch")
    summary = {
        "processed_reads": sum(counts.values()),
        "status_counts": dict(counts),
        "elapsed_seconds": time.monotonic() - started,
        "finished_utc": utc_now(),
        "input_size_mtime_unchanged": all(
            Path(item["path"]).stat().st_size == item["bytes"]
            and Path(item["path"]).stat().st_mtime_ns == item["mtime_ns"]
            for item in inputs
        ),
        "length_method": plan["length_method"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_summary(output, counts, lengths)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

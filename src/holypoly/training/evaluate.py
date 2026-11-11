"""Export held-out predictions in the Bonito environment, without training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import toml
import torch
from tqdm import tqdm

from holypoly.core.config import load_config
from holypoly.models.model import (
    HolypolyModel,
    head_config_from_checkpoint,
    load_bonito_v5_model,
)
from holypoly.models.poly_heads import ReadStatus, decode_location_intervals
from holypoly.training.data import (
    ManifestLabelStatus,
    Pod5SignalReader,
    PolyTrainingDataset,
    PolyTrainingExample,
    PolyTrainingRecord,
    collate_poly_training_examples,
    load_training_manifest,
)
from holypoly.training.engine import require_single_cuda_device, set_random_seed
from holypoly.training.launcher import resolve_training_paths
from holypoly.training.timing_reference import load_timing_references


def build_parser() -> argparse.ArgumentParser:
    """Expose explicit inputs and a default held-out test split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--timing-reference-manifest",
        type=Path,
        help="Matching timing table, required only for a timing-enabled checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory: results/training_evaluation/<run>/<split>-<tag>",
    )
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-reads", type=int, default=0, help="0 evaluates all reads"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_read_ids(records: Sequence[PolyTrainingRecord]) -> None:
    """Reject repeated UUIDs, including distinct chunks or different splits."""
    seen = set()
    for record in records:
        if record.read_id in seen:
            raise ValueError(f"read_id occurs more than once: {record.read_id}")
        seen.add(record.read_id)


def _evaluation_device() -> torch.device:
    """Require one explicitly selected physical GPU for real evaluation."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("select exactly one GPU with CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    require_single_cuda_device(device)
    return device


def _empty_prediction(record: PolyTrainingRecord) -> dict[str, Any]:
    """Retain label sources and absolute, half-open raw-sample coordinates."""
    return {
        "read_id": record.read_id,
        "split": record.split.value,
        "nominal_length_nt": record.nominal_length_nt,
        "training_length_nt": record.training_length_nt,
        "observed_sequence_length_nt": record.observed_sequence_length_nt,
        "poly_type": record.poly_type.value,
        "orientation": record.orientation,
        "label_status": record.label_status.value,
        "supervision_mode": record.supervision_mode.value,
        "label_source": record.label_source,
        "flank_source": record.flank_source,
        "chunk_start_sample": record.chunk_start_sample,
        "chunk_end_sample": record.chunk_end_sample,
        "model_input_samples": record.input_length_samples,
        "head_excluded": record.head_excluded,
        "head_boundary_sample": record.head_boundary_sample,
        "regression_length_nt": None,
        "decoded_length_nt": None,
        "length_head_status": "",
        "shared_status": "",
        "shared_status_confidence": None,
        "shared_poly_present_probability": None,
        "shared_zero_nt_probability": None,
        "shared_failure_probability": None,
        "predicted_start_sample": None,
        "predicted_end_sample": None,
        "prediction_status": "input_failure",
        "failure_reason": "",
    }


def predict_reads(
    model: HolypolyModel,
    dataset: PolyTrainingDataset,
    *,
    device: torch.device,
    batch_size: int,
    show_progress: bool = False,
    timing_references: dict[str, tuple[float, float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Run eval/inference mode on [batch, 1, samples] float32 signal chunks.

    Never update parameters. Keep input failures in their original row order;
    inference errors such as CUDA OOM stop the run instead of hiding failures.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    rows = [_empty_prediction(record) for record in dataset.records]
    pending: list[tuple[int, PolyTrainingExample]] = []

    def flush() -> None:
        """Predict one batch and map chunk-relative intervals to raw samples."""
        batch = collate_poly_training_examples(
            [example for _, example in pending], timing_references=timing_references
        )
        output = model(
            batch.signal.to(device),
            batch.raw_padding_mask.to(device),
            **(
                {"timing_reference": batch.timing_reference.to(device)}
                if batch.timing_reference is not None
                else {}
            ),
        )
        shared_logits = (
            output.location.status_logits + output.length.status_logits
        ) / 2
        probabilities = shared_logits.softmax(-1).cpu()
        intervals = decode_location_intervals(
            output.location, shared_logits, batch.raw_lengths_samples.to(device)
        )
        lengths = output.length.length_nt.cpu().tolist()
        decoded = output.length.predicted_length_nt.cpu().tolist()
        length_statuses = output.length.status_logits.argmax(-1).cpu().tolist()
        for position, (index, example) in enumerate(pending):
            row = rows[index]
            if (
                not math.isfinite(lengths[position])
                or not torch.isfinite(probabilities[position]).all()
            ):
                row.update(
                    prediction_status="nonfinite_prediction",
                    failure_reason="nonfinite_regression_or_shared_status",
                )
                continue
            status = int(probabilities[position].argmax())
            row.update(
                regression_length_nt=lengths[position],
                decoded_length_nt=(
                    decoded[position] if math.isfinite(decoded[position]) else None
                ),
                length_head_status=ReadStatus(length_statuses[position]).name.lower(),
                shared_status=ReadStatus(status).name.lower(),
                shared_status_confidence=float(probabilities[position, status]),
                shared_poly_present_probability=float(
                    probabilities[position, int(ReadStatus.POLY_PRESENT)]
                ),
                shared_zero_nt_probability=float(
                    probabilities[position, int(ReadStatus.ZERO_NT)]
                ),
                shared_failure_probability=float(
                    probabilities[position, int(ReadStatus.FAILURE)]
                ),
                prediction_status="ok",
            )
            interval = intervals[position]
            if interval is not None:
                row["predicted_start_sample"] = (
                    example.record.chunk_start_sample + interval.raw_start_sample
                )
                row["predicted_end_sample"] = (
                    example.record.chunk_start_sample + interval.raw_end_sample
                )
        pending.clear()

    with torch.inference_mode():
        for index in tqdm(
            range(len(dataset)),
            desc="Predict reads",
            unit="read",
            disable=not show_progress,
            dynamic_ncols=True,
        ):
            try:
                pending.append((index, dataset[index]))
            except (OSError, KeyError, ValueError, RuntimeError) as error:
                rows[index]["failure_reason"] = f"{type(error).__name__}: {error}"
            if len(pending) == batch_size:
                flush()
        if pending:
            flush()
    return rows


def _ranks(values: list[float]) -> list[float]:
    """Return average ranks so ties are handled correctly for Spearman."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            ranks[index] = (start + end - 1) / 2
        start = end
    return ranks


def agreement_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Describe raw regression agreement with finite, non-failure targets.

    Fractions use successfully predicted, labeled reads as the denominator;
    report failed inputs separately. These are weak-label agreement metrics,
    not formal per-read accuracy estimates.
    """
    valid = [
        row
        for row in rows
        if row["prediction_status"] == "ok"
        and row["label_status"] != ManifestLabelStatus.FAILURE.value
        and math.isfinite(row["training_length_nt"])
    ]
    result: dict[str, Any] = {
        "reads": len(rows),
        "evaluable_reads": len(valid),
        "prediction_failures": sum(row["prediction_status"] != "ok" for row in rows),
        "shared_status_failure_reads": sum(
            row["shared_status"] == "failure" for row in rows
        ),
    }
    names = [
        "mae_nt",
        "median_ae_nt",
        "rmse_nt",
        "bias_nt",
        "pearson",
        "spearman",
        "within_1nt_fraction",
        "within_5nt_fraction",
        "within_10nt_fraction",
    ]
    result.update(dict.fromkeys(names))
    if not valid:
        return result
    targets = [float(row["training_length_nt"]) for row in valid]
    predictions = [float(row["regression_length_nt"]) for row in valid]
    errors = [
        prediction - target
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    absolute = [abs(error) for error in errors]
    result.update(
        mae_nt=statistics.mean(absolute),
        median_ae_nt=statistics.median(absolute),
        rmse_nt=math.sqrt(statistics.mean(error * error for error in errors)),
        bias_nt=statistics.mean(errors),
    )
    for tolerance in (1, 5, 10):
        result[f"within_{tolerance}nt_fraction"] = sum(
            error <= tolerance for error in absolute
        ) / len(valid)
    if len(set(targets)) > 1 and len(set(predictions)) > 1:
        result["pearson"] = statistics.correlation(targets, predictions)
        result["spearman"] = statistics.correlation(
            _ranks(targets), _ranks(predictions)
        )
    return result


def status_agreement_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare shared argmax status with weak labels; input failures stay separate.

    Failure predictions on labeled positives/zeros are reported separately from
    false positives/negatives. No gating is applied to raw length metrics.
    """
    classes = [status.name.lower() for status in ReadStatus]
    confusion = {target: dict.fromkeys(classes, 0) for target in classes}
    for row in rows:
        if row["prediction_status"] == "ok" and row["label_status"] in confusion:
            confusion[row["label_status"]][row["shared_status"]] += 1
    zeros = sum(confusion["zero_nt"].values())
    positives = sum(confusion["poly_present"].values())
    total = sum(sum(counts.values()) for counts in confusion.values())
    return {
        "label_source": "manifest_label_status_weak_supervision",
        "decision_rule": "argmax_of_averaged_head_logits; no_length_gating",
        "confusion_true_rows_predicted_columns": confusion,
        "evaluable_reads": total,
        "input_or_prediction_failures": sum(
            row["prediction_status"] != "ok" for row in rows
        ),
        "accuracy": sum(confusion[c][c] for c in classes) / total if total else None,
        "zero_nt_reads": zeros,
        "zero_nt_false_positive_fraction": (
            confusion["zero_nt"]["poly_present"] / zeros if zeros else None
        ),
        "zero_nt_predicted_failure_fraction": (
            confusion["zero_nt"]["failure"] / zeros if zeros else None
        ),
        "poly_present_reads": positives,
        "poly_present_false_zero_fraction": (
            confusion["poly_present"]["zero_nt"] / positives if positives else None
        ),
        "poly_present_recall": (
            confusion["poly_present"]["poly_present"] / positives if positives else None
        ),
        "poly_present_predicted_failure_fraction": (
            confusion["poly_present"]["failure"] / positives if positives else None
        ),
    }


def _write_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write TSV, preserving unknown/nonfinite values as empty cells."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(
            {
                key: ""
                if isinstance(value, float) and not math.isfinite(value)
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate paths, load frozen checkpoint weights, and export held-out results."""
    if args.batch_size < 1 or args.max_reads < 0:
        raise ValueError("batch-size must be positive; max-reads must be nonnegative")
    config = load_config(
        args.config, cli_data_root=args.data_root, cli_work_root=args.work_root
    )
    config = replace(
        config,
        training=replace(
            config.training,
            manifest_path=args.manifest or config.training.manifest_path,
            bonito_model=args.model_dir or config.training.bonito_model,
            checkpoint_path=args.checkpoint or config.training.checkpoint_path,
        ),
    )
    paths = resolve_training_paths(config)
    if not paths.checkpoint_path.is_file():
        raise FileNotFoundError(paths.checkpoint_path)
    output_dir = (paths.work_root / args.output_dir).resolve()
    project_root = Path(__file__).resolve().parents[3]
    if (
        output_dir == paths.work_root
        or not output_dir.is_relative_to(paths.work_root)
        or output_dir.is_relative_to(paths.data_root)
        or output_dir.is_relative_to(project_root)
    ):
        raise ValueError(
            "output-dir must be an external work-root child outside raw data"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"output already exists; choose a new directory: {output_dir}"
        )
    records = load_training_manifest(paths.manifest_path)
    _validate_read_ids(records)
    selected = [record for record in records if record.split.value == args.split]
    split_total = len(selected)
    if args.max_reads:
        selected = selected[: args.max_reads]
    if not selected:
        raise ValueError(f"manifest contains no {args.split} reads")
    for record in selected:
        if not record.pod5_path.resolve().is_relative_to(paths.data_root):
            raise ValueError(f"POD5 path is outside data_root: {record.pod5_path}")
    summary: dict[str, Any] = {
        "split": args.split,
        "split_total_reads": split_total,
        "selected_reads": len(selected),
        "partial_evaluation": len(selected) < split_total,
        "batch_size": args.batch_size,
        "pod5_files": len({r.pod5_path for r in selected}),
        "raw_samples_to_read": sum(r.raw_signal_length for r in selected),
        "approx_float32_gib": sum(r.raw_signal_length * 4 for r in selected) / 1024**3,
        "checkpoint_bytes": paths.checkpoint_path.stat().st_size,
        "manifest": str(paths.manifest_path),
        "checkpoint": str(paths.checkpoint_path),
        "model_dir": str(paths.bonito_model),
        "output_dir": str(output_dir),
        "limitations": (
            "Training-target agreement only; construct nominal lengths and provisional "
            "boundaries are not independent per-read truth. Same-batch held-out reads "
            "do not establish cross-batch generalization. length_only labels do not "
            "supervise location; status is ignored unless joint_config explicitly "
            "enables weak status supervision. No failure-detection ability is "
            "established without labeled failure examples. "
            "Raw regression drives length metrics; "
            "decoded_length_nt uses the length head status, while shared_status and "
            "interval decoding use averaged head logits."
        ),
    }
    print(json.dumps(summary, sort_keys=True))
    if args.dry_run:
        return summary
    device = _evaluation_device()
    for name, directory in {
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TRITON_CACHE_DIR": "triton",
        "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "CUDA_CACHE_PATH": "cuda",
    }.items():
        cache = Path(
            os.environ.get(name, str(paths.work_root / "cache" / directory))
        ).resolve()
        if not cache.is_relative_to(paths.work_root) or cache.is_relative_to(
            paths.data_root
        ):
            raise ValueError(f"{name} must be inside work_root and outside data_root")
        os.environ[name] = str(cache)
    summary["started_utc"] = datetime.now(UTC).isoformat()
    payload = torch.load(paths.checkpoint_path, map_location="cpu", weights_only=False)
    model_config = toml.load(paths.bonito_model / "config.toml")
    if model_config != payload["model_config"]:
        raise ValueError("Bonito model config differs from the checkpoint")
    if {record.manifest_version for record in records} != {
        payload["manifest_schema_version"]
    }:
        raise ValueError("manifest schema differs from the checkpoint")
    set_random_seed(int(payload["random_seed"]))
    state = payload["model_state_dict"]
    head_config = head_config_from_checkpoint(payload)
    timing_references = None
    timing_path = getattr(args, "timing_reference_manifest", None)
    if bool(head_config.get("use_timing_reference", False)) != (
        timing_path is not None
    ):
        raise ValueError(
            "timing-enabled checkpoint requires --timing-reference-manifest; "
            "ordinary checkpoints must omit it"
        )
    if timing_path is not None:
        timing_path = (paths.work_root / timing_path).resolve()
        if not timing_path.is_relative_to(
            paths.work_root
        ) or timing_path.is_relative_to(paths.data_root):
            raise ValueError("timing table must be inside work_root, outside raw data")
        timing_references = load_timing_references(
            timing_path, records, source_manifest=paths.manifest_path
        )
        saved_timing = (payload.get("joint_config") or {}).get(
            "timing_reference_manifest_sha256"
        )
        from holypoly.datasets.prepare_sequence import sha256

        if saved_timing is not None and sha256(timing_path) != saved_timing:
            raise ValueError("evaluation timing table differs from checkpoint")
        summary["timing_reference_manifest"] = str(timing_path)
    model = HolypolyModel(
        load_bonito_v5_model(paths.bonito_model, device=str(device)),
        **head_config,
    ).to(device)
    model.load_state_dict(state, strict=True)
    reader = Pod5SignalReader(
        signal_shift=float(model_config["standardisation"]["mean"]),
        signal_scale=float(model_config["standardisation"]["stdev"]),
    )
    dataset = PolyTrainingDataset(selected, reader)
    rows = predict_reads(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        show_progress=True,
        timing_references=timing_references,
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["nominal_length_nt"], row["poly_type"], row["supervision_mode"])
        ].append(row)
    group_metrics = [
        {
            "nominal_length_nt": key[0],
            "poly_type": key[1],
            "supervision_mode": key[2],
            **agreement_metrics(group),
        }
        for key, group in sorted(grouped.items())
    ]
    summary.update(
        metrics=agreement_metrics(rows),
        label_sources=dict(Counter(r.label_source for r in selected)),
        finished_utc=datetime.now(UTC).isoformat(),
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        random_seed=int(payload["random_seed"]),
        git_commit=payload["git_commit"],
        bonito_commit=payload["bonito_commit"],
        model_config=model_config,
        loss_config=payload["loss_config"],
        head_config=head_config,
        joint_config=payload.get("joint_config"),
        data_order_seed=payload.get("data_order_seed"),
        manifest_schema_version=payload["manifest_schema_version"],
        python_version=sys.version,
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        command=sys.argv,
    )
    if (payload.get("joint_config") or {}).get("status_supervision"):
        summary["status_metrics"] = status_agreement_metrics(rows)
        summary["status_metrics_by_nominal_length"] = {
            str(length): status_agreement_metrics(
                [row for row in rows if row["nominal_length_nt"] == length]
            )
            for length in sorted({row["nominal_length_nt"] for row in rows})
        }
    output_dir.mkdir(parents=True)
    _write_table(output_dir / "predictions.tsv", rows)
    _write_table(output_dir / "metrics_by_group.tsv", group_metrics)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Report a clear failure without changing the input checkpoint or manifest."""
    try:
        summary = run(build_parser().parse_args(argv))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary.get("metrics", {"dry_run": True}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

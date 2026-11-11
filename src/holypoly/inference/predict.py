"""Run unchanged Poly weights on unlabeled, sequence-assisted cDNA candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pysam
import toml
import torch
from pod5 import Reader
from tqdm import tqdm

from holypoly.core.config import load_config
from holypoly.inference.candidates import candidate_signal_bounds, terminal_candidates
from holypoly.inference.timing_reference import (
    timing_reference_from_record,
    timing_vector,
)
from holypoly.models.model import (
    HolypolyModel,
    head_config_from_checkpoint,
    load_bonito_v5_model,
)
from holypoly.models.poly_heads import ReadStatus
from holypoly.training.data import Pod5SignalReader
from holypoly.training.engine import require_single_cuda_device, set_random_seed

LIMITATIONS = (
    "Exploratory transfer from construction-trained weights. Terminal sequence "
    "candidates and move boundaries do not establish biological tails or their "
    "true boundaries. Internal A-rich regions, trimmed/missing tails and split "
    "reads can be missed or misidentified. No ground-truth accuracy is computed. "
    "No candidate does not mean zero nt. Completeness is unknown. Status scores "
    "are uncalibrated and never gate the exported regression lengths. "
    "Distribution denominator is selected unique primary BAM records, not all "
    "POD5 reads; basecalling failures absent from BAM are not counted."
)


def build_parser() -> argparse.ArgumentParser:
    """Expose external inputs; keep the default run bounded to 1000 BAM reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--pod5-dir", type=Path, required=True)
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory: results/cdna_prediction/<experiment>/prediction",
    )
    parser.add_argument(
        "--max-reads", type=int, default=1000, help="0 means all BAM reads"
    )
    parser.add_argument("--chunk-size-samples", type=int, default=12000)
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="Candidate reads per model forward; independent of basecalling batch size",
    )
    parser.add_argument(
        "--pod5-reader-cache-size",
        type=int,
        default=8,
        help="Maximum open POD5 readers reused during prediction; 0 disables caching",
    )
    parser.add_argument(
        "--poly-types", nargs="+", choices=("polyA", "polyT"), default=["polyA"]
    )
    parser.add_argument("--min-target-bases", type=int, default=10)
    parser.add_argument("--max-interruption-bases", type=int, default=2)
    parser.add_argument("--min-target-fraction", type=float, default=0.8)
    parser.add_argument("--end-margin-bases", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--use-timing-reference",
        action="store_true",
        help="Ordinary-window extraction for a timing-enabled checkpoint (default off)",
    )
    return parser


def candidate_rows(bam_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Keep every selected primary record, including missing or ambiguous evidence."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for record in bam.fetch(until_eof=True):
            if record.is_secondary or record.is_supplementary:
                continue
            read_id = str(record.query_name)
            if read_id in seen:
                raise ValueError(
                    "duplicate primary BAM read_id; use one basecall per read"
                )
            seen.add(read_id)
            row: dict[str, Any] = {
                "read_id": read_id,
                "poly_type": "",
                "candidate_count": 0,
                "start_base": "",
                "end_base": "",
                "observed_sequence_span_nt": "",
                "observed_target_bases": "",
                "raw_signal_length": "",
                "start_sample": "",
                "end_sample": "",
                "pod5_path": "",
                "predicted_length_nt": "",
                "status": "no_candidate",
                "failure_reason": "no_qualifying_terminal_sequence_candidate",
                "poly_is_complete": "unknown",
                "candidate_source": "terminal_sequence_moves_provisional",
                "model_status": "",
                "model_status_confidence": "",
                "model_poly_present_probability": "",
                "model_zero_nt_probability": "",
            }
            rows.append(row)
            if getattr(args, "use_timing_reference", False):
                timing, _ = timing_reference_from_record(record)
                row.update(timing)
            sequence = record.get_forward_sequence() or ""
            candidates = terminal_candidates(
                sequence,
                min_target_bases=args.min_target_bases,
                max_interruption_bases=args.max_interruption_bases,
                min_target_fraction=args.min_target_fraction,
                end_margin_bases=args.end_margin_bases,
            )
            row["candidate_count"] = len(candidates)
            if (
                record.has_tag("pi")
                or (record.has_tag("dx") and record.get_tag("dx") != 0)
                or any(op == 5 for op, _ in (record.cigartuples or []))
            ):
                row.update(
                    status="unsupported_read",
                    failure_reason="split_duplex_or_hardclipped_read",
                )
            elif len(candidates) > 1:
                row.update(
                    status="ambiguous_candidate",
                    failure_reason="multiple_terminal_candidates",
                )
            elif candidates:
                candidate = candidates[0]
                row.update(
                    poly_type=candidate.poly_type,
                    start_base=candidate.start_base,
                    end_base=candidate.end_base,
                    observed_sequence_span_nt=candidate.span_bases,
                    observed_target_bases=candidate.target_bases,
                )
                if candidate.poly_type not in args.poly_types:
                    row.update(
                        status="type_not_selected",
                        failure_reason="candidate_type_not_requested",
                    )
                else:
                    try:
                        if not all(record.has_tag(tag) for tag in ("mv", "ts", "ns")):
                            raise ValueError("missing_move_tags")
                        raw_length = int(record.get_tag("ns"))
                        start, end = candidate_signal_bounds(
                            record.get_tag("mv"),
                            trim_samples=int(record.get_tag("ts")),
                            raw_samples=raw_length,
                            sequence_length=len(sequence),
                            start_base=candidate.start_base,
                            end_base=candidate.end_base,
                        )
                        row.update(
                            raw_signal_length=raw_length,
                            start_sample=start,
                            end_sample=end,
                        )
                        if end - start > args.chunk_size_samples:
                            raise ValueError(
                                "candidate_exceeds_model_chunk; not_clipped"
                            )
                        row.update(status="pending", failure_reason="")
                    except ValueError as error:
                        row.update(
                            status="candidate_mapping_failure",
                            failure_reason=str(error),
                        )
            if args.max_reads and len(rows) >= args.max_reads:
                break
    if not rows:
        raise ValueError("BAM contains no selected primary records")
    return rows


def locate_signals(rows: list[dict[str, Any]], pod5_files: Sequence[Path]) -> None:
    """Match requested UUIDs using POD5 ID metadata, without loading raw samples."""
    pending = {row["read_id"]: row for row in rows if row["status"] == "pending"}
    if not pending:
        return
    found: dict[str, Path] = {}
    for path in pod5_files:
        with Reader(path) as reader:
            for read_id in set(reader.read_ids) & pending.keys():
                if read_id in found:
                    raise ValueError("selected read_id occurs in multiple POD5 files")
                found[read_id] = path
    for read_id, row in pending.items():
        if read_id in found:
            row["pod5_path"] = str(found[read_id])
        else:
            row.update(
                status="input_failure", failure_reason="read_id_not_found_in_pod5"
            )


def predict_candidates(
    model: HolypolyModel,
    rows: list[dict[str, Any]],
    signal_reader: Callable[[Path, str], torch.Tensor],
    *,
    device: torch.device,
    chunk_size_samples: int,
    inference_batch_size: int = 1,
    use_timing_reference: bool = False,
) -> None:
    """Infer bounded candidate batches, retaining row order and input failures.

    Signal is normalized float32 [B,1,S]; padding mask is bool [B,S]. No
    PolyTrainingRecord, fake labels, or untrained location output is used.
    Model/inference errors propagate instead of being mislabeled input failures.
    """
    if inference_batch_size < 1:
        raise ValueError("inference-batch-size must be positive")
    model.eval()
    batch_rows: list[dict[str, Any]] = []
    batch_signals: list[torch.Tensor] = []
    batch_lengths: list[int] = []

    def flush_batch() -> None:
        """Forward valid signals and assign each result to its original read."""
        if not batch_rows:
            return
        signal_batch = torch.stack(batch_signals)[:, None].to(device)
        lengths = torch.tensor(batch_lengths, device=device)
        mask = torch.arange(chunk_size_samples, device=device)[None] >= lengths[:, None]
        extra = (
            {
                "timing_reference": torch.tensor(
                    [timing_vector(row) for row in batch_rows],
                    dtype=torch.float32,
                    device=device,
                )
            }
            if use_timing_reference
            else {}
        )
        output = model(signal_batch, mask, **extra)
        values = output.length.length_nt.detach().cpu().tolist()
        logits = (output.location.status_logits + output.length.status_logits) / 2
        batch_probabilities = logits.softmax(-1).detach().cpu()
        for row, value, probabilities in zip(
            batch_rows, values, batch_probabilities, strict=True
        ):
            if not math.isfinite(value) or not torch.isfinite(probabilities).all():
                row.update(
                    status="prediction_failure", failure_reason="nonfinite_model_output"
                )
                continue
            status = int(probabilities.argmax())
            row.update(
                predicted_length_nt=value,
                status="candidate_predicted",
                failure_reason="",
                model_status=ReadStatus(status).name.lower(),
                model_status_confidence=float(probabilities[status]),
                model_poly_present_probability=float(
                    probabilities[int(ReadStatus.POLY_PRESENT)]
                ),
                model_zero_nt_probability=float(probabilities[int(ReadStatus.ZERO_NT)]),
            )
        batch_rows.clear()
        batch_signals.clear()
        batch_lengths.clear()

    with torch.inference_mode():
        for row in tqdm(rows, desc="cDNA candidates", unit="read"):
            if row["status"] != "pending":
                continue
            try:
                signal = signal_reader(Path(row["pod5_path"]), row["read_id"])
                if signal.ndim != 1 or len(signal) != row["raw_signal_length"]:
                    raise ValueError("pod5_bam_raw_signal_length_mismatch")
                signal = signal[row["start_sample"] : row["end_sample"]]
                length = len(signal)
                if (
                    not 0 < length <= chunk_size_samples
                    or not torch.isfinite(signal).all()
                ):
                    raise ValueError("invalid_candidate_signal")
            except (OSError, KeyError, ValueError) as error:
                row.update(status="input_failure", failure_reason=str(error))
                continue
            padded = torch.nn.functional.pad(
                signal.float(), (0, chunk_size_samples - length)
            )
            batch_rows.append(row)
            batch_signals.append(padded)
            batch_lengths.append(length)
            if len(batch_rows) == inference_batch_size:
                flush_batch()
        flush_batch()


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Describe raw predicted distributions, never computing label accuracy."""
    distributions = {}
    for poly_type in ("polyA", "polyT"):
        values = [
            float(row["predicted_length_nt"])
            for row in rows
            if row["status"] == "candidate_predicted" and row["poly_type"] == poly_type
        ]
        if values:
            distributions[poly_type] = {
                "reads": len(values),
                "mean_nt": statistics.mean(values),
                "median_nt": statistics.median(values),
                "min_nt": min(values),
                "max_nt": max(values),
            }
    return {
        "selected_primary_bam_reads": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "distributions_by_type": distributions,
        "limitations": LIMITATIONS,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate paths, report scope, and predict without touching training outputs."""
    if (
        args.max_reads < 0
        or args.inference_batch_size < 1
        or args.pod5_reader_cache_size < 0
        or args.chunk_size_samples < 12
        or args.chunk_size_samples % 12
    ):
        raise ValueError(
            "invalid max-reads, inference-batch-size, pod5-reader-cache-size "
            "or model input alignment"
        )
    terminal_candidates(
        "",
        min_target_bases=args.min_target_bases,
        max_interruption_bases=args.max_interruption_bases,
        min_target_fraction=args.min_target_fraction,
        end_margin_bases=args.end_margin_bases,
    )
    config = load_config(
        args.config, cli_data_root=args.data_root, cli_work_root=args.work_root
    )
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("data_root and work_root are required")
    data, work = config.paths.data_root.resolve(), config.paths.work_root.resolve()
    project = Path(__file__).resolve().parents[3]
    pod5_dir, bam, checkpoint = (
        (root / value).resolve()
        for root, value in (
            (data, args.pod5_dir),
            (work, args.bam),
            (work, args.checkpoint),
        )
    )
    model_value = args.model_dir or config.training.bonito_model
    if model_value is None:
        raise ValueError("provide --model-dir or training.bonito_model")
    model_dir = (work / model_value).resolve()
    output = (work / args.output_dir).resolve()
    if not pod5_dir.is_dir() or not pod5_dir.is_relative_to(data):
        raise ValueError("pod5-dir must be an existing directory inside data_root")
    for path in (bam, checkpoint, model_dir / "config.toml"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        output == work
        or not output.is_relative_to(work)
        or output.is_relative_to(data)
        or output.is_relative_to(project)
        or output.exists()
        or output.is_symlink()
        or any(
            path.is_relative_to(output)
            for path in (bam, checkpoint, model_dir, pod5_dir)
        )
    ):
        raise ValueError("output must be a new external work-root child outside inputs")
    files = sorted(pod5_dir.rglob("*.pod5"))
    if not files:
        raise ValueError("no POD5 files found")
    summary: dict[str, Any] = {
        "command": sys.argv,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "bam": str(bam),
        "checkpoint": str(checkpoint),
        "model_dir": str(model_dir),
        "pod5_files": len(files),
        "bam_bytes": bam.stat().st_size,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "max_primary_bam_reads": args.max_reads,
        "scope": (
            "POD5 UUID metadata; full signals only for reads with candidates; "
            "single explicit GPU"
        ),
        "python_version": sys.version,
        "limitations": LIMITATIONS,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        return summary
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("select exactly one GPU using CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    require_single_cuda_device(device)
    summary["started_utc"] = datetime.now(UTC).isoformat()
    rows = candidate_rows(bam, args)
    locate_signals(rows, files)
    if any(row["status"] == "pending" for row in rows):
        for name in (
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "CUDA_CACHE_PATH",
        ):
            path = Path(
                os.environ.get(name, str(work / "cache/cdna-inference" / name))
            ).resolve()
            if not path.is_relative_to(work) or path.is_relative_to(data):
                raise ValueError(
                    f"{name} must be inside work_root and outside data_root"
                )
            os.environ[name] = str(path)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_config = toml.load(model_dir / "config.toml")
        if model_config != payload["model_config"]:
            raise ValueError("Bonito model configuration differs from checkpoint")
        expected_size = (payload.get("joint_config") or {}).get("chunk_size_samples")
        if expected_size is not None and expected_size != args.chunk_size_samples:
            raise ValueError("chunk-size-samples differs from checkpoint")
        set_random_seed(int(payload["random_seed"]))
        if (
            bool(
                head_config_from_checkpoint(payload).get("use_timing_reference", False)
            )
            != args.use_timing_reference
        ):
            raise ValueError(
                "timing-reference option must match checkpoint architecture; "
                "use --use-timing-reference for a trained timing-enabled checkpoint"
            )
        model = HolypolyModel(
            load_bonito_v5_model(model_dir, device=str(device)),
            **head_config_from_checkpoint(payload),
        ).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        standard = model_config["standardisation"]
        with Pod5SignalReader(
            signal_shift=float(standard["mean"]),
            signal_scale=float(standard["stdev"]),
            max_open_files=args.pod5_reader_cache_size,
        ) as reader:
            predict_candidates(
                model,
                rows,
                reader,
                device=device,
                chunk_size_samples=args.chunk_size_samples,
                inference_batch_size=args.inference_batch_size,
                use_timing_reference=args.use_timing_reference,
            )
        summary.update(
            epoch=int(payload["epoch"]),
            head_config=head_config_from_checkpoint(payload),
            loss_config=payload["loss_config"],
            joint_config=payload.get("joint_config"),
            checkpoint_git_commit=payload.get("git_commit"),
            checkpoint_bonito_commit=payload.get("bonito_commit"),
            gpu=torch.cuda.get_device_name(0),
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
        with checkpoint.open("rb") as stream:
            summary["checkpoint_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    for key, directory in (
        ("git_commit", project),
        ("bonito_commit", project / "src/bonito"),
    ):
        completed = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        summary[key] = (
            completed.stdout.strip() if completed.returncode == 0 else "unknown"
        )
    summary.update(summarize(rows), finished_utc=datetime.now(UTC).isoformat())
    if args.use_timing_reference:
        summary["timing_reference"] = {
            "available_reads": sum(row["reference_available"] == 1 for row in rows),
            "status_counts": dict(
                Counter(str(row["reference_status"]) for row in rows)
            ),
        }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.tsv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summarize(rows)), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone unlabeled inference command."""
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"cDNA inference failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Length/status/reference-head joint training in the isolated Bonito environment."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import toml
import torch
from torch.utils.data import DataLoader

from holypoly.core.config import load_config
from holypoly.core.result_paths import training_result_dir
from holypoly.datasets.prepare_sequence import sha256
from holypoly.models.joint_model import JointHolypolyModel, create_joint_optimizer
from holypoly.models.length_head_config import (
    DEFAULT_TCN_LENGTH_FEATURES,
    DEFAULT_TCN_LENGTH_SUM_SCALE,
    TCN_LENGTH_FEATURES,
    validate_tcn_length_features,
    validate_tcn_length_sum_scale,
)
from holypoly.models.model import head_config_from_checkpoint, load_bonito_v5_model
from holypoly.training.data import (
    ManifestSplit,
    Pod5SignalReader,
    PolyTrainingDataset,
    collate_poly_training_examples,
    load_training_manifest,
)
from holypoly.training.engine import (
    CheckpointContext,
    epoch_checkpoint_path,
    load_checkpoint,
    require_single_cuda_device,
    save_checkpoint,
    set_random_seed,
    validate_joint_config,
)
from holypoly.training.joint_branch import materialize_joint_branch, plan_joint_branch
from holypoly.training.joint_engine import (
    joint_log_record,
    train_joint_epoch,
    validate_joint_epoch,
    validate_sequences,
)
from holypoly.training.loss_config import normalize_loss_config
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
from holypoly.training.progress import timed_stage
from holypoly.training.run_validation import (
    git_commit,
    manifest_summary,
    single_manifest_version,
    validate_epoch_checkpoint_paths,
    validate_manifest_sources,
    validate_output_state,
    validate_training_paths,
)
from holypoly.training.sequence_data import (
    SequenceDataset,
    collate_sequence_examples,
    load_sequence_manifest,
)
from holypoly.training.timing_reference import load_timing_references


def _validate_resume_metadata(
    payload: dict[str, Any],
    *,
    log: Path,
    result_dir: Path,
    joint_config: dict[str, Any],
    target_epochs: int,
    allow_batch_size_change: bool,
    recorded_utc: str,
    allow_learning_rate_change: bool = True,
    historical_branch: bool = False,
) -> list[dict[str, Any]]:
    """Validate mixed-batch epoch history without rewriting any completed records."""
    saved_config = payload.get("joint_config")
    validate_joint_config(
        saved_config,
        joint_config,
        allow_batch_size_change=allow_batch_size_change,
        allow_learning_rate_change=allow_learning_rate_change,
    )
    epoch, step = payload["epoch"], payload["global_step"]
    if target_epochs <= epoch:
        raise ValueError("resume target epochs must exceed checkpoint epoch")
    if not log.is_file():
        raise ValueError("resume requires the existing complete training log")
    history_lines = log.read_text().splitlines()
    if historical_branch:
        history_lines = history_lines[: 2 * epoch]
    history = [json.loads(line) for line in history_lines]
    expected = [
        (index, split)
        for index in range(1, epoch + 1)
        for split in ("train", "validation")
    ]
    if (
        [(row["epoch"], row["split"]) for row in history] != expected
        or history[-1]["global_step"] != step
        or history[-1].get("joint_config") != saved_config
    ):
        raise ValueError("training log and resume checkpoint are inconsistent")
    for index in range(0, len(history), 2):
        train_config = history[index].get("joint_config")
        if train_config != history[index + 1].get("joint_config"):
            raise ValueError("training/validation joint configurations differ")
        # Historical settings remain valid even on a later strict resume.
        validate_joint_config(
            train_config,
            saved_config,
            allow_batch_size_change=True,
            allow_learning_rate_change=True,
        )
    if not (result_dir / "basecalling_validation_baseline.json").is_file():
        raise ValueError("resume requires the original basecalling validation baseline")
    summary_path = result_dir / "run_summary.json"
    if not summary_path.is_file():
        raise ValueError("resume requires the original run summary")
    original_summary = json.loads(summary_path.read_text())
    # Summary may predate the latest saved epoch if a continuation was interrupted.
    validate_joint_config(
        original_summary.get("joint_config"),
        saved_config,
        allow_batch_size_change=True,
        allow_learning_rate_change=True,
    )
    changes = list(payload.get("length_batch_size_changes", []))
    old_size = saved_config["length_batch_size"]
    new_size = joint_config["length_batch_size"]
    if old_size != new_size:
        changes.append(
            {
                "after_epoch": epoch,
                "global_step": step,
                "from_batch_size": old_size,
                "to_batch_size": new_size,
                "recorded_utc": recorded_utc,
            }
        )
    return changes


def _resume_learning_rate_changes(
    payload: dict[str, Any], joint_config: dict[str, Any], recorded_utc: str
) -> list[dict[str, Any]]:
    """Append explicit rate provenance without modifying checkpoint metadata."""
    changes = list(payload.get("learning_rate_changes", []))
    saved = payload["joint_config"]
    keys = ("head_learning_rate", "encoder_learning_rate")
    if any(saved[key] != joint_config[key] for key in keys):
        changes.append(
            {
                "after_epoch": payload["epoch"],
                "global_step": payload["global_step"],
                "recorded_utc": recorded_utc,
                **{f"from_{key}": saved[key] for key in keys},
                **{f"to_{key}": joint_config[key] for key in keys},
            }
        )
    return changes


def build_parser() -> argparse.ArgumentParser:
    """Expose the standard joint workflow and explicit legacy resume switches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--run-id",
        default="reference-joint-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Compatible frozen sum/Huber checkpoint; optimizer starts fresh",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Archived joint checkpoint to branch from; requires a new run-id",
    )
    parser.add_argument(
        "--allow-learning-rate-change",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply requested head/encoder learning rates after optimizer restore "
            "on resume (default); --no-allow-learning-rate-change requires matching"
        ),
    )
    parser.add_argument(
        "--allow-batch-size-change",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow length --batch-size changes on resume (default); "
            "use --no-allow-batch-size-change for strict matching"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-batch-size", type=int, default=1)
    parser.add_argument(
        "--frozen-sequence-monitor",
        choices=("auto", "baseline", "per-step"),
        default="auto",
        help=(
            "Frozen encoder only: auto uses baseline-only monitoring for new runs "
            "and preserves the saved policy on resume (old checkpoints: per-step)"
        ),
    )
    parser.add_argument(
        "--profile-batches",
        type=int,
        default=0,
        help=(
            "Time N train batches after 5 warmup batches in the first new/resumed "
            "epoch; 0 disables synchronized profiling"
        ),
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="CPU batches queued per length/sequence stream; 0 disables prefetch",
    )
    parser.add_argument(
        "--pod5-reader-cache-size",
        type=int,
        default=8,
        help="Maximum open POD5 readers per stream; 0 disables handle reuse",
    )
    parser.add_argument("--chunk-size-samples", type=int, default=12000)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--head-type", choices=("legacy", "tcn"), default="legacy")
    parser.add_argument(
        "--tcn-length-features",
        choices=TCN_LENGTH_FEATURES,
        default=DEFAULT_TCN_LENGTH_FEATURES,
        help="TCN length inputs: scaled sum + mean + log1p(valid frames) by default; "
        "sum restores the historical MLP",
    )
    parser.add_argument(
        "--tcn-length-sum-scale",
        type=float,
        default=DEFAULT_TCN_LENGTH_SUM_SCALE,
        help="Fixed multiplier before the TCN length MLP (default: 0.001); "
        "independent of each read's valid frame count",
    )
    parser.add_argument(
        "--use-timing-reference",
        action="store_true",
        help="Ordinary-window timing fusion (default off); requires an audited table",
    )
    parser.add_argument("--timing-reference-manifest", type=Path)
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--encoder-learning-rate", type=float, default=0.00001)
    parser.add_argument("--lambda-basecall", type=float, default=0.1)
    parser.add_argument("--lambda-length", type=float)
    parser.add_argument("--lambda-status", type=float)
    parser.add_argument(
        "--supervise-status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use construct weak present/zero labels (default); location stays ignored. "
            "--no-supervise-status is for explicit ablations/legacy two-loss resumes."
        ),
    )
    parser.add_argument(
        "--unfreeze-last-layers",
        type=int,
        default=1,
        help="Final Transformer blocks to train; 0 freezes the entire encoder",
    )
    parser.add_argument("--huber-delta-nt", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=20260923)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run joint training while releasing prefetched readers on success or failure."""
    with ExitStack() as resources:
        return _run(args, resources)


def _run(args: argparse.Namespace, resources: ExitStack) -> dict[str, Any]:
    """Validate both manifests before reading model weights, then train one GPU."""
    validate_tcn_length_sum_scale(args.tcn_length_sum_scale)
    validate_tcn_length_features(args.tcn_length_features)
    if args.head_type == "tcn" and (
        args.hidden_features != 128 or args.use_timing_reference
    ):
        raise ValueError("tcn requires hidden-features=128 and no timing reference")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError("run-id must be one directory name")
    if (
        args.prefetch_batches < 0
        or args.pod5_reader_cache_size < 0
        or args.profile_batches < 0
    ):
        raise ValueError(
            "prefetch-batches, pod5-reader-cache-size and profile-batches "
            "must be nonnegative"
        )
    if args.unfreeze_last_layers < 0:
        raise ValueError("unfreeze_last_layers must be nonnegative")
    for key in (
        "epochs",
        "batch_size",
        "sequence_batch_size",
        "chunk_size_samples",
        "hidden_features",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    for key in (
        "head_learning_rate",
        "encoder_learning_rate",
        "lambda_basecall",
        "gradient_clip_norm",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if (
        args.chunk_size_samples % 12
        or args.random_seed < 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
        or (args.resume and args.initialize_from is not None)
        or (args.resume_from is not None and (args.resume or args.initialize_from))
    ):
        raise ValueError(
            "invalid chunk size, seed, weight decay or resume/initialization"
        )
    config = load_config(args.config)
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("Configure data_root/work_root")
    data, work = (
        p.expanduser().resolve()
        for p in (config.paths.data_root, config.paths.work_root)
    )
    project = Path(__file__).resolve().parents[3]
    if work.is_relative_to(data) or work.is_relative_to(project):
        raise ValueError("work_root must be external and outside data_root")
    model_value = args.model_dir or config.training.bonito_model
    if model_value is None:
        raise ValueError("configure training.bonito_model or pass --model-dir")
    model_dir = (work / model_value).resolve()
    manifest, sequence_path = (
        (work / p).resolve() for p in (args.manifest, args.sequence_manifest)
    )
    checkpoint = work / "models" / args.run_id / "heads.pt"
    log = work / "logs" / args.run_id / "train.jsonl"
    result_dir = training_result_dir(work, args.run_id, resume=args.resume)
    for path in (manifest, sequence_path):
        if (
            not path.is_file()
            or not path.is_relative_to(work)
            or path.is_relative_to(data)
        ):
            raise ValueError(
                "both manifests must exist inside work_root and outside raw data"
            )
    for path in (checkpoint.resolve(), log.resolve(), result_dir):
        if (
            not path.is_relative_to(work)
            or path.is_relative_to(data)
            or path.is_relative_to(project)
            or path in (manifest, sequence_path)
        ):
            raise ValueError("unsafe joint output path")
    validate_training_paths(
        data_root=data,
        work_root=work,
        manifest_path=manifest,
        model_directory=model_dir,
        checkpoint_path=checkpoint,
        loss_log_path=log,
    )
    validate_output_state(
        checkpoint_path=checkpoint,
        loss_log_path=log,
        resume=args.resume,
        overwrite=False,
    )
    if not args.resume:
        for path in (checkpoint.parent, log.parent, result_dir):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"choose a new run-id; output exists: {path}")
    if args.initialize_from is not None:
        args.initialize_from = (work / args.initialize_from).resolve()
        if (
            not args.initialize_from.is_relative_to(work)
            or not args.initialize_from.is_file()
        ):
            raise ValueError("initialization checkpoint must exist inside work_root")
    records = load_training_manifest(manifest)
    validate_manifest_sources(records, data_root=data)
    if any(
        r.supervision_mode.value != "length_only"
        or r.manifest_version != "construct-nominal-head-poly-v4"
        for r in records
    ):
        raise ValueError(
            "reference-head joint training requires unchanged "
            "head-poly length_only labels"
        )
    input_sizes = {record.input_length_samples for record in records}
    if input_sizes != {args.chunk_size_samples}:
        raise ValueError(
            f"manifest model_input_samples must match --chunk-size-samples "
            f"{args.chunk_size_samples}; found {sorted(input_sizes)}"
        )
    sequence_records = load_sequence_manifest(
        sequence_path, records, source_manifest_sha256=sha256(manifest), data_root=data
    )
    timing_references = None
    timing_path = None
    if args.use_timing_reference != (args.timing_reference_manifest is not None):
        raise ValueError(
            "use --use-timing-reference together with --timing-reference-manifest"
        )
    if args.use_timing_reference:
        timing_path = (work / args.timing_reference_manifest).resolve()
        if not timing_path.is_relative_to(work) or timing_path.is_relative_to(data):
            raise ValueError("timing table must be inside work_root, outside raw data")
        timing_references = load_timing_references(
            timing_path, records, source_manifest=manifest
        )
    sequence_counts = Counter(r.split.value for r in sequence_records)
    if not sequence_counts["train"] or not sequence_counts["validation"]:
        raise ValueError(
            "need sequence train and validation reads; inspect preparation audit"
        )
    model_config = toml.load(model_dir / "config.toml")
    depth = (
        model_config.get("model", {})
        .get("encoder", {})
        .get("transformer_encoder", {})
        .get("depth")
    )
    if not isinstance(depth, int) or not 0 <= args.unfreeze_last_layers <= depth:
        raise ValueError("model config must specify a supported v5 Transformer depth")
    state_len = model_config["model"]["seqdist"]["state_len"]
    if any(len(row.sequence_target) < state_len for row in sequence_records):
        raise ValueError("sequence targets shorter than native CRF state context")
    weights = PolyLossWeights(
        **{
            **vars(config.auxiliary_loss),
            "length_loss": "huber",
            "huber_delta_nt": args.huber_delta_nt,
            **({"lambda_location": 0.0} if args.supervise_status else {}),
            **(
                {"lambda_length": args.lambda_length}
                if args.lambda_length is not None
                else {}
            ),
            **(
                {"lambda_status": args.lambda_status}
                if args.lambda_status is not None
                else {}
            ),
        }
    )
    if args.supervise_status and (
        weights.lambda_status <= 0 or weights.lambda_length <= 0
    ):
        raise ValueError(
            "status joint training requires positive length/status weights"
        )
    joint_config: dict[str, Any] = {
        "schema": "reference-head-joint-v1",
        "length_manifest": str(manifest),
        "length_manifest_sha256": sha256(manifest),
        "sequence_manifest": str(sequence_path),
        "sequence_manifest_sha256": sha256(sequence_path),
        "reference_sha256": sequence_records[0].reference_sha256,
        "model_dir": str(model_dir),
        "model_config_sha256": sha256(model_dir / "config.toml"),
        "unfreeze_last_layers": args.unfreeze_last_layers,
        "head_learning_rate": args.head_learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "lambda_basecall": args.lambda_basecall,
        "sequence_batch_size": args.sequence_batch_size,
        "length_batch_size": args.batch_size,
        "chunk_size_samples": args.chunk_size_samples,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "random_seed": args.random_seed,
        "basecall_loss": "bonito_native_ctc_crf",
        "sequence_label_source": "reference_head_exact_match",
        "boundary_source": "dorado_moves_provisional",
        "crf_frozen": True,
    }
    if args.head_type != "legacy":
        joint_config["head_type"] = args.head_type
        # Missing in old runs means an unscaled sum; retain their resume config
        # when the user explicitly requests the matching multiplier of 1.
        if args.tcn_length_sum_scale != 1.0:
            joint_config["tcn_length_sum_scale"] = args.tcn_length_sum_scale
        if args.tcn_length_features != "sum":
            joint_config["tcn_length_features"] = args.tcn_length_features
    if args.supervise_status:
        # Absence preserves exact configuration equality for old joint resumes.
        joint_config.update(
            status_supervision="construct_nominal_weak_supervision",
            status_target_source="v4_manifest_label_status",
            status_classes=["poly_present", "zero_nt", "failure"],
            status_class_balance="none",
            status_train_counts=dict(
                Counter(
                    r.label_status.value
                    for r in records
                    if r.split == ManifestSplit.TRAIN
                )
            ),
            location_supervision=False,
        )
    if timing_path is not None:
        joint_config.update(
            use_timing_reference=True,
            timing_reference_manifest=str(timing_path),
            timing_reference_manifest_sha256=sha256(timing_path),
            timing_reference_summary_sha256=sha256(timing_path.parent / "summary.json"),
        )
    summary = {
        "started_utc": datetime.now(UTC).isoformat(),
        "command": [
            sys.executable,
            "-m",
            "holypoly.training.joint_command",
            *sys.argv[1:],
        ],
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **manifest_summary(records, epochs=args.epochs),
        "joint_config": joint_config,
        "input_pipeline": {
            "prefetch_batches": args.prefetch_batches,
            "pod5_reader_cache_size": args.pod5_reader_cache_size,
            "backend": "ordered_cpu_threads",
        },
        "loss_config": asdict(weights),
        "sequence_split_counts": dict(sequence_counts),
        "status_supervision_enabled": args.supervise_status,
        "run_id": args.run_id,
        "result_dir": str(result_dir),
        "dry_run": args.dry_run,
        "initialize_from": str(args.initialize_from) if args.initialize_from else None,
        "limitations": (
            "Construct weak-length and selected reference-head sequence agreement; "
            "not independent read truth or cross-batch generalization."
        ),
    }
    if timing_references is not None:
        summary["timing_reference_coverage"] = {
            split.value: {
                "reads": sum(r.split == split for r in records),
                "available": sum(
                    timing_references[r.read_id][2] == 1
                    for r in records
                    if r.split == split
                ),
            }
            for split in ManifestSplit
        }
    resume_position = None
    frozen_monitor = "baseline"
    batch_size_changes: list[dict[str, Any]] = []
    learning_rate_changes: list[dict[str, Any]] = []
    branch = None
    restore_checkpoint = checkpoint
    restore_log, restore_result = log, result_dir
    if args.resume_from is not None:
        branch = plan_joint_branch(
            args.resume_from,
            output_checkpoint=checkpoint,
            output_log=log,
            output_result=result_dir,
            work_root=work,
            data_root=data,
            project_root=project,
        )
        restore_checkpoint = branch.checkpoint
        restore_log, restore_result = branch.parent_log, branch.parent_result
    if args.resume or branch is not None:
        # mmap touches metadata only here; no GPU allocation or POD5 signal reads.
        payload = torch.load(
            restore_checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
        if branch is not None and payload["epoch"] != branch.epoch:
            raise ValueError("resume-from filename and checkpoint epoch differ")
        batch_size_changes = _validate_resume_metadata(
            payload,
            log=restore_log,
            result_dir=restore_result,
            joint_config=joint_config,
            target_epochs=args.epochs,
            allow_batch_size_change=args.allow_batch_size_change,
            allow_learning_rate_change=args.allow_learning_rate_change,
            recorded_utc=summary["started_utc"],
            historical_branch=branch is not None,
        )
        original_summary = json.loads((restore_result / "run_summary.json").read_text())
        if branch is not None:
            summary["branch_origin"] = branch.provenance(payload["global_step"])
            summary["initialize_from"] = original_summary.get("initialize_from")
            summary["initialization"] = original_summary.get("initialization")
        elif "branch_origin" in original_summary:
            summary["branch_origin"] = original_summary["branch_origin"]
        resume_position = (payload["epoch"], payload["global_step"])
        learning_rate_changes = _resume_learning_rate_changes(
            payload, joint_config, summary["started_utc"]
        )
        frozen_monitor = payload.get("input_pipeline", {}).get(
            "frozen_sequence_monitor", "per-step"
        )
        del payload
        summary["length_batch_size_changes"] = batch_size_changes
        summary["learning_rate_changes"] = learning_rate_changes
        if batch_size_changes:
            print(
                "Length batch size changes are recorded. One sequence batch per "
                "length update is unchanged; sequence updates per epoch therefore "
                "depend on length batch size. Learning rates are not rescaled.",
                file=sys.stderr,
                flush=True,
            )
    if args.frozen_sequence_monitor != "auto":
        frozen_monitor = args.frozen_sequence_monitor
    baseline_only = args.unfreeze_last_layers == 0 and frozen_monitor == "baseline"
    summary["input_pipeline"].update(
        poly_only_forward=True,
        frozen_sequence_monitor=frozen_monitor,
        profile_batches=args.profile_batches,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print(
        (
            "Single selected GPU; encoder fully frozen. "
            if args.unfreeze_last_layers == 0
            else "Single selected GPU; partial encoder backward increases memory use. "
        )
        + f"{len(sequence_records)} head chunks plus original length reads; "
        + (
            "Head sequences are baseline monitoring only. "
            if baseline_only
            else "One head batch per length update. "
        )
        + f"Snapshots roughly {(args.epochs + 1) * 0.4:.1f} GiB; "
        "runtime may be hours. "
        "Start with --dry-run, then 1-5 epochs on a small manifest.",
        file=sys.stderr,
        flush=True,
    )
    if args.dry_run:
        return summary
    device = torch.device("cuda:0")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("select exactly one physical GPU with CUDA_VISIBLE_DEVICES")
    require_single_cuda_device(device)
    # Keep framework-generated caches outside source and raw-data directories.
    for name in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
    ):
        cache = Path(
            os.environ.get(name, str(work / "cache" / args.run_id / name.lower()))
        ).resolve()
        if not cache.is_relative_to(work) or cache.is_relative_to(data):
            raise ValueError(f"{name} must remain inside work_root")
        os.environ[name] = str(cache)
    set_random_seed(args.random_seed)
    standard = model_config["standardisation"]
    # Independent readers avoid concurrent access to the same cached POD5 handle.
    reader_options = {
        "signal_shift": float(standard["mean"]),
        "signal_scale": float(standard["stdev"]),
        "max_open_files": args.pod5_reader_cache_size,
    }
    reader = resources.enter_context(Pod5SignalReader(**reader_options))
    sequence_reader = resources.enter_context(Pod5SignalReader(**reader_options))
    length_generators = torch.Generator(), torch.Generator()
    loaders = {}
    for split in (ManifestSplit.TRAIN, ManifestSplit.VALIDATION):
        loaders[split] = DataLoader(
            PolyTrainingDataset(
                records, reader, split=split, chunk_size_samples=args.chunk_size_samples
            ),
            batch_size=args.batch_size,
            shuffle=split == ManifestSplit.TRAIN,
            generator=length_generators[0] if split == ManifestSplit.TRAIN else None,
            collate_fn=partial(
                collate_poly_training_examples,
                supervise_status=args.supervise_status,
                timing_references=timing_references,
            ),
            num_workers=0,
        )
    sequence_loaders = {}
    for split in (ManifestSplit.TRAIN, ManifestSplit.VALIDATION, ManifestSplit.TEST):
        if sequence_counts[split.value]:
            sequence_loaders[split] = DataLoader(
                SequenceDataset(sequence_records, sequence_reader, split=split),
                batch_size=args.sequence_batch_size,
                shuffle=split == ManifestSplit.TRAIN,
                generator=length_generators[1]
                if split == ManifestSplit.TRAIN
                else None,
                collate_fn=collate_sequence_examples,
                num_workers=0,
            )
    model = JointHolypolyModel(
        load_bonito_v5_model(model_dir, device=str(device)),
        unfreeze_last_layers=args.unfreeze_last_layers,
        length_head_mode="sum",
        hidden_features=args.hidden_features,
        dropout=0.0,
        use_timing_reference=args.use_timing_reference,
        head_type=args.head_type,
        tcn_length_sum_scale=args.tcn_length_sum_scale,
        tcn_length_features=args.tcn_length_features,
    ).to(device)
    optimizer = create_joint_optimizer(
        model,
        head_learning_rate=args.head_learning_rate,
        encoder_learning_rate=args.encoder_learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch, step = 0, 0
    if args.initialize_from is not None:
        payload = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        # A new experiment may change objective weights, but must retain the
        # same regression definition and architecture. Resume remains strict.
        if (
            head_config_from_checkpoint(payload) != model.head_config
            or any(
                normalize_loss_config(payload["loss_config"])[key]
                != normalize_loss_config(asdict(weights))[key]
                for key in ("length_loss", "huber_delta_nt")
            )
            or payload["model_config"] != model_config
            or payload.get("joint_config") is not None
            or payload["manifest_schema_version"] != single_manifest_version(records)
        ):
            raise ValueError(
                "initialization must match frozen sum/Huber model configuration"
            )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        summary["initialization"] = {
            "checkpoint": str(args.initialize_from),
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "git_commit": payload.get("git_commit"),
            "loss_config": payload["loss_config"],
        }
        del payload
    if args.resume or branch is not None:
        state = load_checkpoint(
            restore_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
            data_order_seed=args.random_seed,
            loss_weights=weights,
            joint_config=joint_config,
            allow_batch_size_change=args.allow_batch_size_change,
            allow_learning_rate_change=args.allow_learning_rate_change,
        )
        start_epoch, step = state.epoch, state.global_step
        if (start_epoch, step) != resume_position:
            raise ValueError("resume checkpoint changed after metadata preflight")
    validate_epoch_checkpoint_paths(
        checkpoint,
        start_epoch=start_epoch,
        epochs=args.epochs,
        data_root=data,
        work_root=work,
        protected_paths=(manifest, sequence_path, log),
    )
    if branch is not None:
        materialize_joint_branch(branch, summary)
    result_dir.mkdir(parents=True, exist_ok=args.resume or branch is not None)
    log.parent.mkdir(parents=True, exist_ok=args.resume or branch is not None)
    baseline_path = result_dir / "basecalling_validation_baseline.json"
    summary["parameter_counts"] = model.parameter_counts()
    timings: dict[str, float] = {}
    summary["stage_seconds"] = timings
    baseline: dict[str, Any] | None = None
    if not args.resume and branch is None:
        with timed_stage("Initial head sequence validation", device, timings):
            baseline = validate_sequences(
                model,
                sequence_loaders[ManifestSplit.VALIDATION],
                device=device,
                prefetch_batches=args.prefetch_batches,
                progress_desc="Initial head sequence validation",
                total_examples=sequence_counts[ManifestSplit.VALIDATION.value],
            )
        baseline_path.write_text(
            json.dumps(
                {**baseline, "initialize_from": summary["initialize_from"]}, indent=2
            )
            + "\n"
        )
        (result_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    elif baseline_only:
        baseline = json.loads(baseline_path.read_text())
    if baseline_only:
        if (
            baseline is None
            or baseline.get("sequence_examples")
            != sequence_counts[ManifestSplit.VALIDATION.value]
            or not all(
                isinstance(baseline.get(key), (int, float))
                and math.isfinite(baseline[key])
                for key in ("basecalling_loss", "reference_edit_rate")
            )
        ):
            raise ValueError("invalid frozen sequence validation baseline")
        baseline = {
            key: baseline[key]
            for key in (
                "basecalling_loss",
                "sequence_examples",
                "reference_edit_rate",
                "reference_bases",
            )
        }
    context = CheckpointContext(
        random_seed=args.random_seed,
        bonito_commit=git_commit(project / "src" / "bonito"),
        git_commit=git_commit(project),
        model_config=model_config,
        manifest_schema_version=single_manifest_version(records),
        loss_weights=weights,
        data_order_seed=args.random_seed,
        joint_config=joint_config,
        length_batch_size_changes=tuple(batch_size_changes),
        input_pipeline=summary["input_pipeline"],
        learning_rate_changes=tuple(learning_rate_changes),
        branch_origin=summary.get("branch_origin"),
    )
    criterion = PolyAuxiliaryLoss(weights)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        length_generators[0].manual_seed(args.random_seed + epoch)
        length_generators[1].manual_seed(args.random_seed + epoch + 1_000_000)
        profile: dict[str, object] = {}
        with timed_stage(f"Epoch {epoch}/{args.epochs} train", device, timings):
            training, train_sequence = train_joint_epoch(
                model,
                loaders[ManifestSplit.TRAIN],
                sequence_loaders[ManifestSplit.TRAIN],
                criterion,
                optimizer,
                lambda_basecall=args.lambda_basecall,
                device=device,
                gradient_clip_norm=args.gradient_clip_norm,
                global_step=step,
                progress_desc=f"Epoch {epoch}/{args.epochs} train",
                prefetch_batches=args.prefetch_batches,
                monitor_frozen_sequences=not baseline_only,
                profile_batches=args.profile_batches if epoch == start_epoch + 1 else 0,
                profile_result=profile,
            )
        step = training.global_step
        validation, val_sequence = validate_joint_epoch(
            model,
            loaders[ManifestSplit.VALIDATION],
            sequence_loaders[ManifestSplit.VALIDATION],
            criterion,
            device=device,
            global_step=step,
            prefetch_batches=args.prefetch_batches,
            progress_desc=f"Epoch {epoch}/{args.epochs} val",
            sequence_examples=sequence_counts[ManifestSplit.VALIDATION.value],
            cached_sequence=baseline if baseline_only else None,
            timings=timings,
        )
        with log.open("a") as stream:
            for split_name, metrics, seq in (
                ("train", training, train_sequence),
                ("validation", validation, val_sequence),
            ):
                row = joint_log_record(
                    metrics,
                    seq,
                    epoch=epoch,
                    split=split_name,
                    lambda_basecall=args.lambda_basecall,
                    loss_config=asdict(weights),
                    joint_config=joint_config,
                )
                row["stage_seconds"] = {
                    key: seconds
                    for key, seconds in timings.items()
                    if key.startswith(f"Epoch {epoch}/")
                }
                if learning_rate_changes:
                    row["learning_rate_changes"] = learning_rate_changes
                if split_name == "train" and profile:
                    row["batch_profile"] = profile
                stream.write(json.dumps(row) + "\n")
        with timed_stage(
            f"Epoch {epoch}/{args.epochs} save checkpoint", device, timings
        ):
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=step,
                context=context,
                data_root=data,
                work_root=work,
                archive_epoch=True,
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "validation_length_mae_nt": validation.length_mae_nt,
                    **val_sequence,
                }
            ),
            flush=True,
        )
    history = [json.loads(line) for line in log.read_text().splitlines()]
    validation_rows = [row for row in history if row["split"] == "validation"]
    best = min(validation_rows, key=lambda row: (row["length_mae_nt"], row["epoch"]))
    best_path = epoch_checkpoint_path(checkpoint, best["epoch"])
    (result_dir / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "minimum validation length MAE, earliest epoch on ties"
                ),
                "epoch": best["epoch"],
                "validation_mae_nt": best["length_mae_nt"],
                "checkpoint": str(best_path),
                "joint_config": best["joint_config"],
                "loss_config": asdict(weights),
            },
            indent=2,
        )
        + "\n"
    )
    if ManifestSplit.TEST in sequence_loaders:
        test_path = result_dir / f"sequence_test_epoch_{best['epoch']:04d}.json"
        if not test_path.exists():
            payload = torch.load(best_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            del payload
            with timed_stage("Best checkpoint head sequence test", device, timings):
                test = validate_sequences(
                    model,
                    sequence_loaders[ManifestSplit.TEST],
                    device=device,
                    prefetch_batches=args.prefetch_batches,
                    progress_desc="Best checkpoint head sequence test",
                    total_examples=sequence_counts[ManifestSplit.TEST.value],
                )
            with test_path.open("x") as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": best["epoch"],
                            "split": "test",
                            **test,
                            "limitations": summary["limitations"],
                        },
                        indent=2,
                    )
                    + "\n"
                )
    summary["finished_utc"] = datetime.now(UTC).isoformat()
    summary["stage_seconds"] = timings
    # Preserve the original run's initialization provenance across continuation.
    summary_path = result_dir / "run_summary.json"
    if args.resume:
        original_summary = json.loads(summary_path.read_text())
        summary["initialize_from"] = original_summary.get("initialize_from")
        summary["initialization"] = original_summary.get("initialization")
        summary["started_utc"] = original_summary["started_utc"]
        summary["stage_seconds"] = {
            **original_summary.get("stage_seconds", {}),
            **timings,
        }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Report configuration or explicitly run an isolated joint experiment."""
    try:
        run(build_parser().parse_args(argv))
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        print(f"Joint training failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

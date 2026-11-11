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
from holypoly.training.joint_engine import (
    joint_log_record,
    train_joint_epoch,
    validate_joint_epoch,
    validate_sequences,
)
from holypoly.training.loss_config import normalize_loss_config
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
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
) -> list[dict[str, Any]]:
    """Validate mixed-batch epoch history without rewriting any completed records."""
    saved_config = payload.get("joint_config")
    validate_joint_config(
        saved_config, joint_config, allow_batch_size_change=allow_batch_size_change
    )
    epoch, step = payload["epoch"], payload["global_step"]
    if target_epochs <= epoch:
        raise ValueError("resume target epochs must exceed checkpoint epoch")
    if not log.is_file():
        raise ValueError("resume requires the existing complete training log")
    history = [json.loads(line) for line in log.read_text().splitlines()]
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
        # Older epochs retain their actual batch size, even on a later strict resume.
        validate_joint_config(train_config, saved_config, allow_batch_size_change=True)
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
            "Use v4 weak present/zero labels (default); location stays ignored. "
            "--no-supervise-status is for explicit ablations/legacy two-loss resumes."
        ),
    )
    parser.add_argument("--unfreeze-last-layers", type=int, default=1)
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
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError("run-id must be one directory name")
    if args.prefetch_batches < 0 or args.pod5_reader_cache_size < 0:
        raise ValueError(
            "prefetch-batches and pod5-reader-cache-size must be nonnegative"
        )
    for key in (
        "epochs",
        "batch_size",
        "sequence_batch_size",
        "chunk_size_samples",
        "hidden_features",
        "unfreeze_last_layers",
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
            "reference-head joint training requires unchanged v4 length_only labels"
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
    if not isinstance(depth, int) or not 1 <= args.unfreeze_last_layers <= depth:
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
    batch_size_changes: list[dict[str, Any]] = []
    if args.resume:
        # mmap touches metadata only here; no GPU allocation or POD5 signal reads.
        payload = torch.load(
            checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
        batch_size_changes = _validate_resume_metadata(
            payload,
            log=log,
            result_dir=result_dir,
            joint_config=joint_config,
            target_epochs=args.epochs,
            allow_batch_size_change=args.allow_batch_size_change,
            recorded_utc=summary["started_utc"],
        )
        resume_position = (payload["epoch"], payload["global_step"])
        del payload
        summary["length_batch_size_changes"] = batch_size_changes
        if batch_size_changes:
            print(
                "Length batch size changes are recorded. One sequence batch per "
                "length update is unchanged; sequence updates per epoch therefore "
                "depend on length batch size. Learning rates are not rescaled.",
                file=sys.stderr,
                flush=True,
            )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print(
        "Single selected GPU; partial encoder backward increases memory use. "
        f"{len(sequence_records)} head chunks plus original length reads; "
        "one head batch per length update. "
        f"Snapshots roughly {(args.epochs + 1) * 0.4:.1f} GiB; "
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
    if args.resume:
        state = load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
            data_order_seed=args.random_seed,
            loss_weights=weights,
            joint_config=joint_config,
            allow_batch_size_change=args.allow_batch_size_change,
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
    result_dir.mkdir(parents=True, exist_ok=args.resume)
    log.parent.mkdir(parents=True, exist_ok=args.resume)
    baseline_path = result_dir / "basecalling_validation_baseline.json"
    summary["parameter_counts"] = model.parameter_counts()
    if not args.resume:
        baseline = validate_sequences(
            model,
            sequence_loaders[ManifestSplit.VALIDATION],
            device=device,
            prefetch_batches=args.prefetch_batches,
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
    )
    criterion = PolyAuxiliaryLoss(weights)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        length_generators[0].manual_seed(args.random_seed + epoch)
        length_generators[1].manual_seed(args.random_seed + epoch + 1_000_000)
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
            progress_desc=f"Joint epoch {epoch}/{args.epochs}",
            prefetch_batches=args.prefetch_batches,
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
        )
        with log.open("a") as stream:
            for split_name, metrics, seq in (
                ("train", training, train_sequence),
                ("validation", validation, val_sequence),
            ):
                stream.write(
                    json.dumps(
                        joint_log_record(
                            metrics,
                            seq,
                            epoch=epoch,
                            split=split_name,
                            lambda_basecall=args.lambda_basecall,
                            loss_config=asdict(weights),
                            joint_config=joint_config,
                        )
                    )
                    + "\n"
                )
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
            test = validate_sequences(
                model,
                sequence_loaders[ManifestSplit.TEST],
                device=device,
                prefetch_batches=args.prefetch_batches,
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
    # Preserve the original run's initialization provenance across continuation.
    summary_path = result_dir / "run_summary.json"
    if args.resume:
        original_summary = json.loads(summary_path.read_text())
        summary["initialize_from"] = original_summary.get("initialize_from")
        summary["initialization"] = original_summary.get("initialization")
        summary["started_utc"] = original_summary["started_utc"]
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

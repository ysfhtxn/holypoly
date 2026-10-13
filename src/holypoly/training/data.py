"""Manifest, dataset, and collation utilities for auxiliary-head training."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

from holypoly.models.poly_heads import (
    BONITO_V5_STRIDE,
    PolyClass,
    ReadStatus,
    make_location_targets,
)

BONITO_V5_INPUT_ALIGNMENT = 12
LOCATION_IGNORE_INDEX = -100


class ManifestSplit(StrEnum):
    """Supported dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ManifestPolyType(StrEnum):
    """Poly types stored in the training manifest."""

    POLY_A = "polyA"
    POLY_T = "polyT"
    NONE = "none"


class ManifestLabelStatus(StrEnum):
    """Read-level label states stored in the training manifest."""

    POLY_PRESENT = "poly_present"
    ZERO_NT = "zero_nt"
    FAILURE = "failure"


class ManifestSupervisionMode(StrEnum):
    """Length-only weak labels do not supervise location or read status."""

    FULL = "full"
    LENGTH_ONLY = "length_only"


@dataclass(frozen=True)
class PolyTrainingRecord:
    """Validated manifest row using absolute raw-signal sample coordinates."""

    read_id: str
    pod5_path: Path
    pod5_file_id: str
    split: ManifestSplit
    orientation: str
    raw_signal_length: int
    chunk_start_sample: int
    chunk_end_sample: int
    poly_start_sample: int | None
    poly_end_sample: int | None
    poly_type: ManifestPolyType
    true_length_nt: float
    label_status: ManifestLabelStatus
    poly_is_complete: bool | None
    failure_reason: str
    manifest_version: str = "synthetic-v1"
    bam_path: Path | None = None
    nominal_length_nt: float = math.nan
    observed_sequence_length_nt: float = math.nan
    label_source: str = "synthetic"
    flank_source: str = "not_applicable"
    flank_status: str = "not_applicable"
    selection_reason: str = "synthetic_fixture"
    supervision_mode: ManifestSupervisionMode = ManifestSupervisionMode.FULL
    model_input_samples: int | None = None
    head_excluded: bool = False
    head_boundary_sample: int | None = None

    @property
    def chunk_length_samples(self) -> int:
        """Return the real retained interval length in raw samples, before padding."""
        return self.chunk_end_sample - self.chunk_start_sample

    @property
    def training_length_nt(self) -> float:
        """Return the explicit length target used by the training loss."""
        return self.true_length_nt

    @property
    def input_length_samples(self) -> int:
        """Return padded model input size; legacy rows use their full chunk size."""
        return self.model_input_samples or self.chunk_length_samples


@dataclass(frozen=True)
class PolyTrainingExample:
    """One model-ready fixed-size signal chunk and its validated metadata."""

    signal: Tensor
    record: PolyTrainingRecord


@dataclass(frozen=True)
class PolyTrainingBatch:
    """Batch consumed by ``BonitoV5MultiTaskModel`` and ``auxiliary_loss``."""

    signal: Tensor
    raw_padding_mask: Tensor
    raw_lengths_samples: Tensor
    location_targets: Tensor
    true_length_nt: Tensor
    status_targets: Tensor
    read_ids: tuple[str, ...]


class SignalReader(Protocol):
    """Read one complete, normalized signal without modifying its source."""

    def __call__(self, pod5_path: Path, read_id: str) -> Tensor: ...


class Pod5SignalReader:
    """Read calibrated POD5 signal with caller-provided model standardisation.

    Coordinates remain relative to the original untrimmed POD5 signal. The
    ``signal_shift`` and ``signal_scale`` values must come from the selected
    Bonito model configuration; they are deliberately not hard-coded here.
    """

    def __init__(self, *, signal_shift: float, signal_scale: float) -> None:
        if not math.isfinite(signal_shift):
            raise ValueError("signal_shift must be finite")
        if not math.isfinite(signal_scale) or signal_scale <= 0:
            raise ValueError("signal_scale must be finite and greater than zero")
        self.signal_shift = signal_shift
        self.signal_scale = signal_scale

    def __call__(self, pod5_path: Path, read_id: str) -> Tensor:
        """Return one normalized float32 signal and never write to the POD5."""
        if not pod5_path.is_file():
            raise FileNotFoundError(f"POD5 file does not exist: {pod5_path}")

        from pod5 import Reader

        with Reader(pod5_path) as reader:
            matches = list(
                reader.reads(
                    selection=[read_id],
                    missing_ok=True,
                    preload={"samples"},
                )
            )
        if not matches:
            raise KeyError(f"read_id {read_id!r} was not found in {pod5_path}")
        if len(matches) != 1:
            raise ValueError(
                f"read_id {read_id!r} occurred {len(matches)} times in {pod5_path}"
            )

        read = matches[0]
        raw_signal = torch.from_numpy(read.signal.astype("float32"))
        calibrated_signal = read.calibration.scale * (
            raw_signal + read.calibration.offset
        )
        return (calibrated_signal - self.signal_shift) / self.signal_scale


class PolyTrainingDataset(Dataset[PolyTrainingExample]):
    """Load real raw intervals, optionally right-padding to fixed model inputs."""

    def __init__(
        self,
        records: Sequence[PolyTrainingRecord],
        signal_reader: SignalReader,
        *,
        split: ManifestSplit | None = None,
        chunk_size_samples: int | None = None,
    ) -> None:
        selected = tuple(
            record for record in records if split is None or record.split is split
        )
        if not selected:
            raise ValueError("training dataset must contain at least one record")

        inferred_chunk_size = selected[0].input_length_samples
        expected_chunk_size = (
            inferred_chunk_size if chunk_size_samples is None else chunk_size_samples
        )
        if expected_chunk_size < BONITO_V5_INPUT_ALIGNMENT:
            raise ValueError(
                f"chunk_size_samples must be at least {BONITO_V5_INPUT_ALIGNMENT}"
            )
        if expected_chunk_size % BONITO_V5_INPUT_ALIGNMENT != 0:
            raise ValueError(
                "chunk_size_samples must be divisible by "
                f"{BONITO_V5_INPUT_ALIGNMENT} for exact Bonito v5 alignment"
            )
        for record in selected:
            if (
                record.input_length_samples != expected_chunk_size
                or not 0 < record.chunk_length_samples <= expected_chunk_size
            ):
                raise ValueError(
                    "all manifest chunks must have the same length; "
                    f"read {record.read_id!r} has input size "
                    f"{record.input_length_samples}, "
                    f"expected {expected_chunk_size}"
                )

        self.records = selected
        self.signal_reader = signal_reader
        self.chunk_size_samples = expected_chunk_size

    def __len__(self) -> int:
        """Return the number of selected manifest records."""
        return len(self.records)

    def __getitem__(self, index: int) -> PolyTrainingExample:
        """Read and slice one signal without changing the external POD5."""
        record = self.records[index]
        full_signal = self.signal_reader(record.pod5_path, record.read_id)
        if full_signal.ndim != 1:
            raise ValueError(
                f"signal for read {record.read_id!r} must be one-dimensional, "
                f"received shape {tuple(full_signal.shape)}"
            )
        if not torch.is_floating_point(full_signal):
            raise ValueError(
                f"signal for read {record.read_id!r} must be floating point"
            )
        if full_signal.shape[0] != record.raw_signal_length:
            raise ValueError(
                f"signal length mismatch for read {record.read_id!r}: manifest "
                f"declares {record.raw_signal_length}, reader returned "
                f"{full_signal.shape[0]}"
            )
        signal = full_signal[record.chunk_start_sample : record.chunk_end_sample].to(
            dtype=torch.float32
        )
        if signal.shape != (record.chunk_length_samples,):
            raise ValueError(
                f"chunk for read {record.read_id!r} has shape "
                f"{tuple(signal.shape)}, expected {(record.chunk_length_samples,)}"
            )
        if not torch.all(torch.isfinite(signal)):
            raise ValueError(f"signal for read {record.read_id!r} contains NaN or Inf")
        signal = torch.nn.functional.pad(
            signal, (0, self.chunk_size_samples - signal.shape[0]), value=0.0
        )
        return PolyTrainingExample(signal=signal.contiguous(), record=record)


def load_training_manifest(path: Path) -> tuple[PolyTrainingRecord, ...]:
    """Load and validate a CSV or TSV auxiliary-head training manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {path}")
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"training manifest has no header: {path}")
        missing_columns = sorted(_REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                f"training manifest {path} is missing required columns: "
                + ", ".join(missing_columns)
            )
        records = tuple(
            _parse_manifest_row(row, row_number, path)
            for row_number, row in enumerate(reader, start=2)
        )
    if not records:
        raise ValueError(f"training manifest contains no records: {path}")

    identities: set[tuple[str, int, int]] = set()
    for record in records:
        identity = (
            record.read_id,
            record.chunk_start_sample,
            record.chunk_end_sample,
        )
        if identity in identities:
            raise ValueError(
                f"training manifest contains duplicate read/chunk identity: {identity}"
            )
        identities.add(identity)
    return records


def collate_poly_training_examples(
    examples: Sequence[PolyTrainingExample],
) -> PolyTrainingBatch:
    """Collate fixed inputs with suffix padding and raw-coordinate Poly labels."""
    if not examples:
        raise ValueError("cannot collate an empty training batch")
    chunk_size_samples = examples[0].signal.shape[0]
    if chunk_size_samples % BONITO_V5_INPUT_ALIGNMENT != 0:
        raise ValueError(
            "signal chunk length must be divisible by "
            f"{BONITO_V5_INPUT_ALIGNMENT} for exact Bonito v5 alignment"
        )
    if any(example.signal.shape != (chunk_size_samples,) for example in examples):
        raise ValueError("all signal chunks in a batch must have the same shape")

    feature_steps = chunk_size_samples // BONITO_V5_STRIDE
    raw_lengths = torch.tensor(
        [example.record.chunk_length_samples for example in examples], dtype=torch.long
    )
    if torch.any(raw_lengths <= 0) or torch.any(raw_lengths > chunk_size_samples):
        raise ValueError("real chunk lengths must fit inside the padded inputs")
    padding_mask = torch.arange(chunk_size_samples)[None, :] >= raw_lengths[:, None]
    location_targets = torch.stack(
        [
            _location_targets_for_record(
                example.record,
                feature_steps=feature_steps,
            )
            for example in examples
        ]
    )
    feature_lengths = torch.div(
        raw_lengths + BONITO_V5_STRIDE - 1, BONITO_V5_STRIDE, rounding_mode="floor"
    )
    location_targets.masked_fill_(
        torch.arange(feature_steps)[None, :] >= feature_lengths[:, None],
        LOCATION_IGNORE_INDEX,
    )
    signal = torch.stack([example.signal for example in examples]).unsqueeze(1)
    return PolyTrainingBatch(
        signal=signal,
        raw_padding_mask=padding_mask,
        raw_lengths_samples=raw_lengths,
        location_targets=location_targets,
        true_length_nt=torch.tensor(
            [example.record.true_length_nt for example in examples],
            dtype=torch.float32,
        ),
        status_targets=torch.tensor(
            [_read_status(example.record) for example in examples],
            dtype=torch.long,
        ),
        read_ids=tuple(example.record.read_id for example in examples),
    )


def _location_targets_for_record(
    record: PolyTrainingRecord, *, feature_steps: int
) -> Tensor:
    if (
        record.label_status is ManifestLabelStatus.FAILURE
        or record.supervision_mode is ManifestSupervisionMode.LENGTH_ONLY
    ):
        return torch.full((feature_steps,), LOCATION_IGNORE_INDEX, dtype=torch.long)
    if record.label_status is ManifestLabelStatus.ZERO_NT:
        return torch.full((feature_steps,), int(PolyClass.BACKGROUND), dtype=torch.long)

    if record.poly_start_sample is None or record.poly_end_sample is None:
        raise ValueError(f"Poly interval is missing for read {record.read_id!r}")
    local_start = record.poly_start_sample - record.chunk_start_sample
    local_end = record.poly_end_sample - record.chunk_start_sample
    poly_class = (
        PolyClass.POLY_A
        if record.poly_type is ManifestPolyType.POLY_A
        else PolyClass.POLY_T
    )
    return make_location_targets(
        local_start,
        local_end,
        raw_length_samples=record.chunk_length_samples,
        feature_steps=feature_steps,
        poly_class=poly_class,
    )


def _read_status(record: PolyTrainingRecord) -> int:
    if record.supervision_mode is ManifestSupervisionMode.LENGTH_ONLY:
        return LOCATION_IGNORE_INDEX
    return {
        ManifestLabelStatus.POLY_PRESENT: int(ReadStatus.POLY_PRESENT),
        ManifestLabelStatus.ZERO_NT: int(ReadStatus.ZERO_NT),
        ManifestLabelStatus.FAILURE: int(ReadStatus.FAILURE),
    }[record.label_status]


def _parse_manifest_row(
    row: dict[str, str], row_number: int, manifest_path: Path
) -> PolyTrainingRecord:
    context = f"{manifest_path}:{row_number}"
    read_id = _required_text(row, "read_id", context)
    pod5_value = _required_text(row, "pod5_path", context)
    pod5_path = Path(pod5_value).expanduser()
    if not pod5_path.is_absolute():
        pod5_path = (manifest_path.parent / pod5_path).resolve()

    try:
        record = PolyTrainingRecord(
            read_id=read_id,
            pod5_path=pod5_path,
            pod5_file_id=_required_text(row, "pod5_file_id", context),
            split=ManifestSplit(_required_text(row, "split", context)),
            orientation=_required_text(row, "orientation", context),
            raw_signal_length=_required_int(row, "raw_signal_length", context),
            chunk_start_sample=_required_int(row, "chunk_start_sample", context),
            chunk_end_sample=_required_int(row, "chunk_end_sample", context),
            poly_start_sample=_optional_int(row, "poly_start_sample", context),
            poly_end_sample=_optional_int(row, "poly_end_sample", context),
            poly_type=ManifestPolyType(_required_text(row, "poly_type", context)),
            true_length_nt=_optional_float(row, "training_length_nt", context),
            label_status=ManifestLabelStatus(
                _required_text(row, "label_status", context)
            ),
            poly_is_complete=_optional_bool(row, "poly_is_complete", context),
            failure_reason=(row.get("failure_reason") or "").strip(),
            manifest_version=_required_text(row, "manifest_version", context),
            bam_path=_optional_resolved_path(
                row, "bam_path", context, manifest_path.parent
            ),
            nominal_length_nt=_optional_float(row, "nominal_length_nt", context),
            observed_sequence_length_nt=_optional_float(
                row, "observed_sequence_length_nt", context
            ),
            label_source=_required_text(row, "label_source", context),
            flank_source=_required_text(row, "flank_source", context),
            flank_status=_required_text(row, "flank_status", context),
            selection_reason=_required_text(row, "selection_reason", context),
            supervision_mode=ManifestSupervisionMode(
                row.get("supervision_mode") or "full"
            ),
            model_input_samples=_optional_int(row, "model_input_samples", context),
            head_excluded=_optional_bool(row, "head_excluded", context) is True,
            head_boundary_sample=_optional_int(row, "head_boundary_sample", context),
        )
    except ValueError as error:
        raise ValueError(
            f"invalid training manifest value at {context}: {error}"
        ) from error
    _validate_record(record, context)
    return record


def _validate_record(record: PolyTrainingRecord, context: str) -> None:
    if record.raw_signal_length < 1:
        raise ValueError(f"{context}: raw_signal_length must be positive")
    if not (
        0
        <= record.chunk_start_sample
        < record.chunk_end_sample
        <= record.raw_signal_length
    ):
        raise ValueError(
            f"{context}: chunk interval must satisfy 0 <= start < end <= "
            "raw_signal_length"
        )

    if record.model_input_samples is not None and (
        record.model_input_samples < BONITO_V5_INPUT_ALIGNMENT
        or record.model_input_samples % BONITO_V5_INPUT_ALIGNMENT != 0
        or record.chunk_length_samples > record.model_input_samples
    ):
        raise ValueError(f"{context}: invalid model_input_samples for real chunk")
    if (
        record.manifest_version == "construct-nominal-head-excluded-v3"
        and not record.head_excluded
    ):
        raise ValueError(f"{context}: v3 requires head_excluded=true")
    if record.head_excluded:
        boundary = record.head_boundary_sample
        if (
            record.model_input_samples is None
            or boundary is None
            or not 0 <= boundary <= record.raw_signal_length
            or record.orientation not in {"forward", "reverse"}
        ):
            raise ValueError(
                f"{context}: head exclusion requires a boundary and input size"
            )
        if (
            record.orientation == "forward"
            and record.chunk_start_sample < boundary
            or record.orientation == "reverse"
            and record.chunk_end_sample > boundary
        ):
            raise ValueError(f"{context}: chunk includes signal on the head side")

    if record.label_status is ManifestLabelStatus.POLY_PRESENT:
        if record.poly_type is ManifestPolyType.NONE:
            raise ValueError(f"{context}: poly_present requires polyA or polyT")
        if record.supervision_mode is ManifestSupervisionMode.LENGTH_ONLY:
            if (
                record.poly_start_sample is not None
                or record.poly_end_sample is not None
                or record.poly_is_complete is not None
                or not math.isnan(record.observed_sequence_length_nt)
            ):
                raise ValueError(
                    f"{context}: length_only must leave Poly bounds, completeness "
                    "and observed_sequence_length_nt unknown"
                )
            if (
                record.label_source != "construct_nominal_weak_supervision"
                or not math.isfinite(record.true_length_nt)
                or record.true_length_nt <= 0
                or record.true_length_nt != record.nominal_length_nt
                or record.failure_reason
            ):
                raise ValueError(
                    f"{context}: length_only requires a positive nominal weak target"
                )
            return
        if record.poly_start_sample is None or record.poly_end_sample is None:
            raise ValueError(f"{context}: poly_present requires a Poly interval")
        if not (
            record.chunk_start_sample
            <= record.poly_start_sample
            < record.poly_end_sample
            <= record.chunk_end_sample
        ):
            raise ValueError(
                f"{context}: the complete Poly interval must be contained in its chunk"
            )
        if record.poly_is_complete is not True:
            raise ValueError(
                f"{context}: poly_present training records must have "
                "poly_is_complete=true"
            )
        if not math.isfinite(record.true_length_nt) or record.true_length_nt <= 0:
            raise ValueError(
                f"{context}: poly_present requires a positive finite true_length_nt"
            )
        if not math.isfinite(record.nominal_length_nt) or record.nominal_length_nt < 0:
            raise ValueError(
                f"{context}: poly_present requires a non-negative nominal_length_nt"
            )
        if (
            not math.isfinite(record.observed_sequence_length_nt)
            or record.observed_sequence_length_nt <= 0
        ):
            raise ValueError(
                f"{context}: poly_present requires a positive "
                "observed_sequence_length_nt"
            )
        if record.failure_reason:
            raise ValueError(f"{context}: poly_present must not define failure_reason")
        return

    if record.supervision_mode is ManifestSupervisionMode.LENGTH_ONLY:
        raise ValueError(f"{context}: length_only requires a positive construct")
    if record.poly_type is not ManifestPolyType.NONE:
        raise ValueError(f"{context}: non-present records require poly_type=none")
    if record.poly_start_sample is not None or record.poly_end_sample is not None:
        raise ValueError(f"{context}: non-present records must not define Poly bounds")

    if record.label_status is ManifestLabelStatus.ZERO_NT:
        if record.true_length_nt != 0.0:
            raise ValueError(f"{context}: zero_nt requires true_length_nt=0")
        if record.nominal_length_nt != 0.0:
            raise ValueError(f"{context}: zero_nt requires nominal_length_nt=0")
        if record.observed_sequence_length_nt != 0.0:
            raise ValueError(
                f"{context}: zero_nt requires observed_sequence_length_nt=0"
            )
        if record.failure_reason:
            raise ValueError(f"{context}: zero_nt must not define failure_reason")
    else:
        if math.isfinite(record.true_length_nt):
            raise ValueError(
                f"{context}: failure records must leave true_length_nt empty"
            )
        if not record.failure_reason:
            raise ValueError(f"{context}: failure records require failure_reason")


def _required_text(row: dict[str, str], name: str, context: str) -> str:
    value = (row.get(name) or "").strip()
    if not value:
        raise ValueError(f"{context}: {name} must not be empty")
    return value


def _required_int(row: dict[str, str], name: str, context: str) -> int:
    return int(_required_text(row, name, context))


def _optional_int(row: dict[str, str], name: str, context: str) -> int | None:
    value = (row.get(name) or "").strip()
    return None if not value else int(value)


def _optional_float(row: dict[str, str], name: str, context: str) -> float:
    value = (row.get(name) or "").strip()
    return math.nan if not value else float(value)


def _optional_bool(row: dict[str, str], name: str, context: str) -> bool | None:
    value = (row.get(name) or "").strip().lower()
    if not value:
        return None
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"{context}: {name} must be true, false, or empty")


def _optional_resolved_path(
    row: dict[str, str], name: str, context: str, base_path: Path
) -> Path | None:
    value = (row.get(name) or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_path / path).resolve()


_REQUIRED_COLUMNS = {
    "manifest_version",
    "read_id",
    "pod5_path",
    "bam_path",
    "pod5_file_id",
    "split",
    "orientation",
    "raw_signal_length",
    "chunk_start_sample",
    "chunk_end_sample",
    "poly_start_sample",
    "poly_end_sample",
    "poly_type",
    "training_length_nt",
    "nominal_length_nt",
    "observed_sequence_length_nt",
    "label_status",
    "poly_is_complete",
    "failure_reason",
    "label_source",
    "flank_source",
    "flank_status",
    "selection_reason",
}

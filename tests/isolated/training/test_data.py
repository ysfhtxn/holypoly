from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import toml
import torch
from bonito.transformer import Model as TransformerModel

from holypoly.models.poly_heads import (
    BONITO_V5_STRIDE,
    BonitoV5MultiTaskModel,
    ReadStatus,
    encoder_interval_to_raw_interval,
    raw_interval_to_encoder_interval,
)
from holypoly.training.data import (
    BONITO_V5_INPUT_ALIGNMENT,
    LOCATION_IGNORE_INDEX,
    ManifestLabelStatus,
    ManifestPolyType,
    ManifestSplit,
    ManifestSupervisionMode,
    PolyTrainingDataset,
    PolyTrainingRecord,
    collate_poly_training_examples,
    load_training_manifest,
)
from tests.isolated.helpers import tiny_v5_basecaller

FIELDNAMES = [
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
]


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "manifest_version": "milestone3-test-v1",
        "read_id": "read-a",
        "pod5_path": "signals/example.pod5",
        "bam_path": "calls/example.bam",
        "pod5_file_id": "pod5-a",
        "split": "train",
        "orientation": "canonical",
        "raw_signal_length": 240,
        "chunk_start_sample": 60,
        "chunk_end_sample": 180,
        "poly_start_sample": 84,
        "poly_end_sample": 126,
        "poly_type": "polyA",
        "training_length_nt": 20,
        "nominal_length_nt": 20,
        "observed_sequence_length_nt": 18,
        "label_status": "poly_present",
        "poly_is_complete": "true",
        "failure_reason": "",
        "label_source": "construct_nominal",
        "flank_source": "inferred_consensus_v1",
        "flank_status": "provisional",
        "selection_reason": "unit_test",
    }
    row.update(overrides)
    return row


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES
            + [
                "supervision_mode",
                "model_input_samples",
                "head_excluded",
                "head_boundary_sample",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _record(
    read_id: str,
    *,
    status: ManifestLabelStatus,
    poly_type: ManifestPolyType = ManifestPolyType.NONE,
    poly_start_sample: int | None = None,
    poly_end_sample: int | None = None,
    true_length_nt: float = math.nan,
    failure_reason: str = "",
) -> PolyTrainingRecord:
    return PolyTrainingRecord(
        read_id=read_id,
        pod5_path=Path(f"/{read_id}.pod5"),
        pod5_file_id="pod5-a",
        split=ManifestSplit.TRAIN,
        orientation="canonical",
        raw_signal_length=240,
        chunk_start_sample=60,
        chunk_end_sample=180,
        poly_start_sample=poly_start_sample,
        poly_end_sample=poly_end_sample,
        poly_type=poly_type,
        true_length_nt=true_length_nt,
        label_status=status,
        poly_is_complete=True if status is ManifestLabelStatus.POLY_PRESENT else None,
        failure_reason=failure_reason,
    )


def test_manifest_schema_loads_present_zero_and_failure_records(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.tsv"
    _write_manifest(
        manifest,
        [
            _row(),
            _row(
                read_id="read-zero",
                split="validation",
                poly_start_sample="",
                poly_end_sample="",
                poly_type="none",
                training_length_nt=0,
                nominal_length_nt=0,
                observed_sequence_length_nt=0,
                label_status="zero_nt",
                poly_is_complete="",
            ),
            _row(
                read_id="read-failure",
                split="test",
                poly_start_sample="",
                poly_end_sample="",
                poly_type="none",
                training_length_nt="",
                nominal_length_nt="",
                observed_sequence_length_nt="",
                label_status="failure",
                poly_is_complete="false",
                failure_reason="missing signal mapping",
            ),
        ],
    )

    records = load_training_manifest(manifest)

    assert len(records) == 3
    assert records[0].pod5_path == (tmp_path / "signals/example.pod5").resolve()
    assert records[0].poly_type is ManifestPolyType.POLY_A
    assert records[1].label_status is ManifestLabelStatus.ZERO_NT
    assert records[1].true_length_nt == 0.0
    assert records[2].label_status is ManifestLabelStatus.FAILURE
    assert math.isnan(records[2].true_length_nt)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"poly_start_sample": 40}, "contained in its chunk"),
        ({"poly_is_complete": "false"}, "poly_is_complete=true"),
        ({"training_length_nt": 0}, "positive finite"),
        (
            {
                "poly_start_sample": "",
                "poly_end_sample": "",
                "poly_type": "none",
                "training_length_nt": 1,
                "nominal_length_nt": 0,
                "observed_sequence_length_nt": 0,
                "label_status": "zero_nt",
                "poly_is_complete": "",
            },
            "zero_nt requires true_length_nt=0",
        ),
        (
            {
                "poly_start_sample": "",
                "poly_end_sample": "",
                "poly_type": "none",
                "training_length_nt": 12,
                "nominal_length_nt": "",
                "observed_sequence_length_nt": "",
                "label_status": "failure",
                "poly_is_complete": "",
                "failure_reason": "mapping failed",
            },
            "failure records must leave true_length_nt empty",
        ),
        (
            {
                "poly_start_sample": "",
                "poly_end_sample": "",
                "poly_type": "none",
                "training_length_nt": "",
                "nominal_length_nt": "",
                "observed_sequence_length_nt": "",
                "label_status": "failure",
                "poly_is_complete": "",
            },
            "failure records require failure_reason",
        ),
    ],
)
def test_manifest_schema_rejects_unsafe_labels(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    manifest = tmp_path / "invalid.tsv"
    _write_manifest(manifest, [_row(**overrides)])

    with pytest.raises(ValueError, match=message):
        load_training_manifest(manifest)


def test_nominal_zero_does_not_override_observed_positive_label(tmp_path: Path) -> None:
    manifest = tmp_path / "nominal-zero.tsv"
    _write_manifest(manifest, [_row(nominal_length_nt=0)])
    record = load_training_manifest(manifest)[0]
    assert record.nominal_length_nt == 0
    assert record.label_status is ManifestLabelStatus.POLY_PRESENT
    assert record.true_length_nt == 20


@pytest.mark.parametrize("nominal", [-1, "", "nan"])
def test_invalid_nominal_positive_record_still_rejected(
    tmp_path: Path, nominal: object
) -> None:
    manifest = tmp_path / "invalid-nominal.tsv"
    _write_manifest(manifest, [_row(nominal_length_nt=nominal)])
    with pytest.raises(ValueError, match="non-negative nominal_length_nt"):
        load_training_manifest(manifest)


def test_manifest_rejects_duplicate_read_chunk(tmp_path: Path) -> None:
    manifest = tmp_path / "duplicate.tsv"
    _write_manifest(manifest, [_row(), _row()])

    with pytest.raises(ValueError, match="duplicate read/chunk"):
        load_training_manifest(manifest)


def test_length_only_manifest_keeps_unknowns_and_masks_location_status(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weak.tsv"
    weak = _row(
        supervision_mode="length_only",
        poly_start_sample="",
        poly_end_sample="",
        poly_is_complete="",
        observed_sequence_length_nt="",
        label_source="construct_nominal_weak_supervision",
    )
    _write_manifest(path, [weak])
    records = load_training_manifest(path)
    assert records[0].supervision_mode is ManifestSupervisionMode.LENGTH_ONLY
    assert math.isnan(records[0].observed_sequence_length_nt)
    dataset = PolyTrainingDataset(records, lambda path, read_id: torch.zeros(240))
    batch = collate_poly_training_examples([dataset[0]])
    assert torch.all(batch.location_targets == -100)
    assert batch.status_targets.tolist() == [-100]
    assert batch.true_length_nt.tolist() == [20]
    for field, value in [
        ("poly_start_sample", 84),
        ("poly_is_complete", "true"),
        ("observed_sequence_length_nt", 18),
        ("training_length_nt", 19),
    ]:
        _write_manifest(path, [{**weak, field: value}])
        with pytest.raises(ValueError, match="length_only"):
            load_training_manifest(path)


def test_dataset_reads_only_requested_fixed_chunks_and_collates_labels() -> None:
    records = [
        _record(
            "read-a",
            status=ManifestLabelStatus.POLY_PRESENT,
            poly_type=ManifestPolyType.POLY_A,
            poly_start_sample=65,
            poly_end_sample=79,
            true_length_nt=14.0,
        ),
        _record(
            "read-zero",
            status=ManifestLabelStatus.ZERO_NT,
            true_length_nt=0.0,
        ),
        _record(
            "read-failure",
            status=ManifestLabelStatus.FAILURE,
            failure_reason="unusable label",
        ),
    ]
    source_signals = {
        record.read_id: torch.arange(240, dtype=torch.float32) for record in records
    }
    calls: list[tuple[Path, str]] = []

    def fake_reader(path: Path, read_id: str) -> torch.Tensor:
        calls.append((path, read_id))
        return source_signals[read_id]

    dataset = PolyTrainingDataset(records, fake_reader)
    examples = [dataset[index] for index in range(len(dataset))]
    batch = collate_poly_training_examples(examples)

    assert len(calls) == 3
    assert batch.signal.shape == (3, 1, 120)
    assert torch.equal(batch.signal[0, 0], torch.arange(60, 180).float())
    assert batch.raw_padding_mask.shape == (3, 120)
    assert not torch.any(batch.raw_padding_mask)
    assert batch.location_targets.shape == (3, 20)
    assert torch.equal(
        batch.location_targets[0],
        torch.tensor([1, 1, 1, 1] + [0] * 16),
    )
    assert torch.count_nonzero(batch.location_targets[1]) == 0
    assert torch.all(batch.location_targets[2] == LOCATION_IGNORE_INDEX)
    assert torch.equal(
        batch.status_targets,
        torch.tensor([ReadStatus.POLY_PRESENT, ReadStatus.ZERO_NT, ReadStatus.FAILURE]),
    )
    assert torch.equal(batch.true_length_nt[:2], torch.tensor([14.0, 0.0]))
    assert torch.isnan(batch.true_length_nt[2])


def test_dataset_rejects_misaligned_or_variable_chunks() -> None:
    aligned = _record(
        "read-a",
        status=ManifestLabelStatus.ZERO_NT,
        true_length_nt=0.0,
    )
    short = PolyTrainingRecord(
        **{
            **aligned.__dict__,
            "read_id": "read-short",
            "chunk_end_sample": 174,
        }
    )

    with pytest.raises(ValueError, match="same length"):
        PolyTrainingDataset([aligned, short], lambda _path, _read_id: torch.zeros(240))
    with pytest.raises(ValueError, match="divisible by 12"):
        PolyTrainingDataset(
            [short],
            lambda _path, _read_id: torch.zeros(240),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"head_excluded": "false"}, "v3 requires head_excluded"),
        ({"head_boundary_sample": ""}, "requires a boundary"),
        ({"model_input_samples": ""}, "requires a boundary"),
        ({"model_input_samples": 60}, "invalid model_input_samples"),
        ({"model_input_samples": 121}, "invalid model_input_samples"),
        ({"head_boundary_sample": 61}, "includes signal on the head side"),
        (
            {"orientation": "reverse", "head_boundary_sample": 179},
            "includes signal on the head side",
        ),
    ],
)
def test_manifest_rejects_unsafe_head_exclusion(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    manifest = tmp_path / "head-free.tsv"
    row = _row(
        manifest_version="construct-nominal-head-excluded-v3",
        orientation="forward",
        head_excluded="true",
        head_boundary_sample=60,
        model_input_samples=120,
    )
    row.update(overrides)
    _write_manifest(manifest, [row])
    with pytest.raises(ValueError, match=message):
        load_training_manifest(manifest)


def test_head_free_padded_inputs_masks_and_backward(tmp_path: Path) -> None:
    """Both acquisition directions exclude head, including from length features."""
    torch.manual_seed(37)
    manifest = tmp_path / "head-free.tsv"
    common = {
        "manifest_version": "construct-nominal-head-excluded-v3",
        "head_excluded": "true",
        "model_input_samples": 120,
    }
    _write_manifest(
        manifest,
        [
            _row(
                **common,
                orientation="forward",
                head_boundary_sample=60,
                chunk_end_sample=139,
            ),
            _row(
                **common,
                read_id="reverse",
                orientation="reverse",
                head_boundary_sample=180,
                chunk_start_sample=101,
                poly_start_sample=126,
                poly_end_sample=180,
                poly_type="polyT",
            ),
            _row(
                **common,
                read_id="zero",
                orientation="forward",
                head_boundary_sample=60,
                chunk_end_sample=115,
                poly_start_sample="",
                poly_end_sample="",
                poly_type="none",
                label_status="zero_nt",
                poly_is_complete="",
                training_length_nt=0,
                nominal_length_nt=0,
                observed_sequence_length_nt=0,
            ),
            _row(
                **common,
                read_id="weak",
                orientation="forward",
                head_boundary_sample=60,
                chunk_end_sample=97,
                supervision_mode="length_only",
                poly_start_sample="",
                poly_end_sample="",
                poly_is_complete="",
                observed_sequence_length_nt="",
                label_source="construct_nominal_weak_supervision",
            ),
        ],
    )
    records = load_training_manifest(manifest)
    signals = {r.read_id: torch.randn(240) for r in records}
    for record in records:
        if record.orientation == "forward":
            signals[record.read_id][: record.head_boundary_sample] = 999
        else:
            signals[record.read_id][record.head_boundary_sample :] = 999
    dataset = PolyTrainingDataset(records, lambda _, read_id: signals[read_id])
    examples = [dataset[i] for i in range(len(dataset))]
    batch = collate_poly_training_examples(examples)
    assert batch.signal.shape == (4, 1, 120)
    assert batch.raw_lengths_samples.tolist() == [79, 79, 55, 37]
    assert batch.raw_padding_mask.sum(dim=1).tolist() == [41, 41, 65, 83]
    assert not torch.any(batch.signal == 999)
    for i, record in enumerate(records):
        length = record.chunk_length_samples
        assert torch.equal(
            batch.signal[i, 0, :length],
            signals[record.read_id][
                record.chunk_start_sample : record.chunk_end_sample
            ],
        )
        assert torch.all(batch.signal[i, 0, length:] == 0)
    assert (
        batch.location_targets[0].tolist() == [0] * 4 + [1] * 7 + [0] * 3 + [-100] * 6
    )
    assert batch.location_targets[1].tolist() == [0] * 4 + [2] * 10 + [-100] * 6
    assert batch.location_targets[2].tolist() == [0] * 10 + [-100] * 10
    assert torch.all(batch.location_targets[3] == LOCATION_IGNORE_INDEX)
    assert batch.status_targets.tolist() == [0, 0, 1, -100]

    model = BonitoV5MultiTaskModel(tiny_v5_basecaller(), hidden_features=6)
    model.eval()
    for parameter in model.basecaller.parameters():
        parameter.requires_grad_(False)
    output = model(batch.signal, batch.raw_padding_mask)
    assert output.location.padding_mask.sum(dim=1).tolist() == [6, 6, 10, 13]
    losses = model.auxiliary_loss(
        output, batch.location_targets, batch.true_length_nt, batch.status_targets
    )
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses["total_loss"].backward()
    assert all(p.grad is None for p in model.basecaller.parameters())
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.length_head.parameters()
    )
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.location_head.parameters()
    )

    # Alter only excluded head samples: neither input nor regression may change.
    for record in records:
        signal = signals[record.read_id]
        if record.orientation == "forward":
            signal[: record.head_boundary_sample] = -999
        else:
            signal[record.head_boundary_sample :] = -999
    changed = collate_poly_training_examples([dataset[i] for i in range(len(dataset))])
    assert torch.equal(batch.signal, changed.signal)
    with torch.no_grad():
        changed_output = model(changed.signal, changed.raw_padding_mask)
    assert torch.equal(output.length.length_nt, changed_output.length.length_nt)


def test_raw_to_encoder_mapping_is_stable_for_aligned_chunks() -> None:
    chunk_size = 120
    feature_steps = chunk_size // BONITO_V5_STRIDE
    assert chunk_size % BONITO_V5_INPUT_ALIGNMENT == 0

    for raw_start in range(chunk_size + 1):
        for raw_end in range(raw_start, chunk_size + 1):
            encoder_start, encoder_end = raw_interval_to_encoder_interval(
                raw_start,
                raw_end,
                raw_length_samples=chunk_size,
                feature_steps=feature_steps,
            )
            mapped_start, mapped_end = encoder_interval_to_raw_interval(
                encoder_start,
                encoder_end,
                raw_length_samples=chunk_size,
                feature_steps=feature_steps,
            )
            assert 0 <= encoder_start <= encoder_end <= feature_steps
            if raw_start < raw_end:
                assert mapped_start <= raw_start < mapped_start + BONITO_V5_STRIDE
                assert mapped_end >= raw_end
                assert mapped_end - raw_end < BONITO_V5_STRIDE


def _run_batch_forward_backward(device: torch.device) -> None:
    records = [
        _record(
            "read-a",
            status=ManifestLabelStatus.POLY_PRESENT,
            poly_type=ManifestPolyType.POLY_T,
            poly_start_sample=72,
            poly_end_sample=120,
            true_length_nt=24.0,
        ),
        _record(
            "read-zero",
            status=ManifestLabelStatus.ZERO_NT,
            true_length_nt=0.0,
        ),
    ]
    signals = {record.read_id: torch.randn(240) for record in records}
    dataset = PolyTrainingDataset(
        records,
        lambda _path, read_id: signals[read_id],
    )
    batch = collate_poly_training_examples([dataset[0], dataset[1]])
    model = BonitoV5MultiTaskModel(tiny_v5_basecaller(), hidden_features=6).to(device)
    for parameter in model.basecaller.parameters():
        parameter.requires_grad_(False)

    output = model(
        batch.signal.to(device),
        batch.raw_padding_mask.to(device),
    )
    assert output.location.location_logits.shape[:2] == tuple(
        batch.location_targets.shape
    )
    losses = model.auxiliary_loss(
        output,
        batch.location_targets.to(device),
        batch.true_length_nt.to(device),
        batch.status_targets.to(device),
    )
    losses["total_loss"].backward()

    assert torch.isfinite(losses["total_loss"])
    assert all(parameter.grad is None for parameter in model.basecaller.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.location_head.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.length_head.parameters()
    )


def test_collated_batch_matches_v5_adapter_on_cpu() -> None:
    _run_batch_forward_backward(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_collated_batch_matches_v5_adapter_on_gpu() -> None:
    _run_batch_forward_backward(torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_official_v5_encoder_steps_match_collated_targets_on_gpu() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "bonito"
        / "bonito"
        / "models"
        / "configs"
        / "dna_r10.4.1@v5.0.toml"
    )
    record = _record(
        "read-a",
        status=ManifestLabelStatus.POLY_PRESENT,
        poly_type=ManifestPolyType.POLY_A,
        poly_start_sample=72,
        poly_end_sample=120,
        true_length_nt=24.0,
    )
    dataset = PolyTrainingDataset(
        [record],
        lambda _path, _read_id: torch.randn(240),
    )
    batch = collate_poly_training_examples([dataset[0]])
    model = BonitoV5MultiTaskModel(
        TransformerModel(toml.load(config_path)), hidden_features=8
    ).to("cuda")
    model.eval()

    with torch.no_grad():
        output = model(
            batch.signal.to("cuda"),
            batch.raw_padding_mask.to("cuda"),
        )

    assert output.encoder_features.shape == (1, 20, 512)
    assert output.location.location_logits.shape[:2] == tuple(
        batch.location_targets.shape
    )

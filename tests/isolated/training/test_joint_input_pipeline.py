"""Synthetic POD5 and CPU-surrogate regression for cached, prefetched inputs."""

from __future__ import annotations

import inspect
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pod5
import pytest
import torch
from torch.utils.data import DataLoader

from holypoly.models.joint_model import JointHolypolyModel, create_joint_optimizer
from holypoly.training import engine, joint_command, joint_engine
from holypoly.training.data import (
    ManifestSplit,
    Pod5SignalReader,
    PolyTrainingDataset,
    collate_poly_training_examples,
)
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
from holypoly.training.prefetch import prefetch_batches
from holypoly.training.sequence_data import (
    SequenceDataset,
    SequenceRecord,
    collate_sequence_examples,
)
from tests.isolated.training.test_reference_joint import (
    joint_basecaller,
    synthetic_sequence,
    weak_status_examples,
)


def _synthetic_pod5(path: Path) -> None:
    """Write six wholly synthetic raw signals, without real sample identifiers."""
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
                    signal=np.arange(240, dtype=np.int16) - 100 + index,
                )
                for index in range(6)
            ]
        )


def test_cached_prefetched_pod5_matches_synchronous_inputs_and_shuffle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real POD5 API, exact normalized tensors/masks/labels, and fewer file opens."""
    path = tmp_path / "synthetic.pod5"
    _synthetic_pod5(path)
    before = path.read_bytes()
    templates = [example.record for example in weak_status_examples()]
    records = [
        replace(
            templates[index % 2],
            read_id=str(UUID(int=index + 1)),
            pod5_path=path,
            raw_signal_length=240,
            chunk_start_sample=84,
            chunk_end_sample=209,
            head_boundary_sample=84,
        )
        for index in range(6)
    ]
    sequences = [
        SequenceRecord(
            record.read_id,
            path,
            ManifestSplit.TRAIN,
            240,
            0,
            72,
            "ACGTACGT",
            "a" * 64,
            "b" * 64,
        )
        for record in records
    ]
    actual_reader = pod5.Reader
    opens = []

    def counted_reader(*args: Any, **kwargs: Any) -> Any:
        opens.append(args[0])
        return actual_reader(*args, **kwargs)

    monkeypatch.setattr(pod5, "Reader", counted_reader)
    results = []
    open_counts = []
    for cache_size, depth in ((0, 0), (2, 2)):
        opens.clear()
        with (
            Pod5SignalReader(
                signal_shift=2, signal_scale=3, max_open_files=cache_size
            ) as length_reader,
            Pod5SignalReader(
                signal_shift=2, signal_scale=3, max_open_files=cache_size
            ) as sequence_reader,
        ):
            lengths = DataLoader(
                PolyTrainingDataset(records, length_reader),
                batch_size=2,
                shuffle=True,
                generator=torch.Generator().manual_seed(17),
                collate_fn=partial(
                    collate_poly_training_examples, supervise_status=True
                ),
            )
            heads = DataLoader(
                SequenceDataset(sequences, sequence_reader, split=ManifestSplit.TRAIN),
                batch_size=2,
                shuffle=True,
                generator=torch.Generator().manual_seed(23),
                collate_fn=collate_sequence_examples,
            )
            with prefetch_batches(lengths, depth) as prepared:
                length_batches = list(prepared)
            with prefetch_batches(heads, depth) as prepared:
                head_batches = list(prepared)
            results.append((length_batches, head_batches))
        open_counts.append(len(opens))
    assert open_counts == [12, 2]
    baseline, optimized = results
    for original, cached in zip(baseline[0], optimized[0], strict=True):
        assert original.read_ids == cached.read_ids
        for field in (
            "signal",
            "raw_padding_mask",
            "raw_lengths_samples",
            "true_length_nt",
            "status_targets",
            "location_targets",
        ):
            assert torch.equal(getattr(original, field), getattr(cached, field))
    for original_batch, cached_batch in zip(baseline[1], optimized[1], strict=True):
        for original, cached in zip(original_batch, cached_batch, strict=True):
            assert original.record == cached.record
            assert torch.equal(original.signal, cached.signal)
            assert torch.equal(original.targets, cached.targets)
    assert path.read_bytes() == before


def test_prefetch_preserves_joint_updates_and_sequence_loader_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU surrogate verifies identical updates while cycling a short head loader."""
    monkeypatch.setattr(joint_engine, "require_single_cuda_device", lambda _: None)
    monkeypatch.setattr(engine, "require_single_cuda_device", lambda _: None)
    torch.manual_seed(17)
    initial = JointHolypolyModel(
        joint_basecaller(), hidden_features=6, length_head_mode="sum"
    )
    batch = collate_poly_training_examples(
        weak_status_examples(), supervise_status=True
    )
    criterion = PolyAuxiliaryLoss(
        PolyLossWeights(lambda_location=0, length_loss="huber")
    )
    results = []
    validations = []
    for depth in (0, 2):
        model = deepcopy(initial)
        monkeypatch.setattr(
            model.basecaller, "loss", lambda scores, *_: scores.square().mean()
        )
        monkeypatch.setattr(model.basecaller, "decode_batch", lambda _: ["ACGTACGT"])
        optimizer = create_joint_optimizer(
            model,
            head_learning_rate=0.001,
            encoder_learning_rate=1e-5,
            weight_decay=0,
        )
        metrics = joint_engine.train_joint_epoch(
            model,
            [batch] * 3,
            [[synthetic_sequence()]],
            criterion,
            optimizer,
            lambda_basecall=0.1,
            device=torch.device("cpu"),
            gradient_clip_norm=2,
            global_step=7,
            prefetch_batches=depth,
        )
        results.append((metrics, deepcopy(model.state_dict()), optimizer.state_dict()))
        validations.append(
            joint_engine.validate_joint_epoch(
                model,
                [batch] * 3,
                [[synthetic_sequence()]],
                criterion,
                device=torch.device("cpu"),
                global_step=10,
                prefetch_batches=depth,
            )
        )
    baseline, optimized = results
    assert asdict(baseline[0][0]) == asdict(optimized[0][0])
    assert baseline[0][1] == optimized[0][1]
    assert optimized[0][0].global_step == 10
    assert optimized[0][1]["sequence_examples"] == 3
    assert asdict(validations[0][0]) == asdict(validations[1][0])
    assert validations[0][1] == validations[1][1]
    for key in baseline[1]:
        assert torch.equal(baseline[1][key], optimized[1][key])
    assert baseline[2]["param_groups"] == optimized[2]["param_groups"]
    for key, values in baseline[2]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, optimized[2]["state"][key][name])


@pytest.mark.parametrize("option", ["--prefetch-batches", "--pod5-reader-cache-size"])
def test_negative_input_pipeline_settings_fail_before_data_access(option: str) -> None:
    args = joint_command.build_parser().parse_args(
        ["--manifest", "unused", "--sequence-manifest", "unused", option, "-1"]
    )
    with pytest.raises(ValueError, match="must be nonnegative"):
        joint_command.run(args)


def test_joint_run_releases_both_readers_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run-level resource scope closes readers even after a model failure."""
    closed = []

    class Reader:
        def __enter__(self) -> Reader:
            return self

        def __exit__(self, *args: Any) -> None:
            closed.append(self)

    def fail(args: Any, resources: ExitStack) -> None:
        resources.enter_context(Reader())
        resources.enter_context(Reader())
        raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(joint_command, "_run", fail)
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        joint_command.run(None)
    assert len(closed) == 2

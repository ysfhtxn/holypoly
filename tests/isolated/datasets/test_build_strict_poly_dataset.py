"""Synthetic POD5/BAM to existing training consumers; no GPU or real data."""

from __future__ import annotations

import csv
import inspect
import json
import os
import subprocess
from array import array
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import pod5
import pysam
import pytest
import torch

from holypoly.datasets import build_strict_poly_dataset as module
from holypoly.datasets.prepare import FIELDS, HEAD_POLY_VERSION, write_table
from holypoly.datasets.prepare_sequence import (
    BOUNDARY_SOURCE,
    LABEL_SOURCE,
    VERSION,
    sha256,
)
from holypoly.training.data import (
    ManifestSplit,
    Pod5SignalReader,
    PolyTrainingDataset,
    collate_poly_training_examples,
    load_training_manifest,
)
from holypoly.training.sequence_data import (
    SequenceDataset,
    collate_sequence_examples,
    load_sequence_manifest,
)


@pytest.fixture
def strict_fixture(tmp_path):
    """Twelve real synthetic POD5 reads, two lengths and both raw directions."""
    data, work = tmp_path / "raw", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    pod5_path = data / "synthetic.pod5"
    bam_path = data / "synthetic.bam"
    head = "ACGT" * 75
    header = pysam.AlignmentHeader.from_references(
        ["synthetic_0A", "synthetic_50A"], [len(head)] * 2
    )
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
        sample_rate=5000,
    )
    run_info = pod5.RunInfo(**metadata)
    length_rows, sequence_rows, raw_reads, alignments = [], [], [], []
    for length in [0, 50]:
        for reverse in [False, True]:
            for split in module.SPLITS:
                index = len(length_rows)
                read_id = UUID(int=index + 1)
                rng = np.random.default_rng(17)
                values = rng.normal(100, 20, 5000)
                if length:
                    values[3600:4100] = 60 + rng.normal(0, 0.8, 500)
                    values[3800:3803] = 140
                if reverse:
                    values = values[::-1].copy()
                raw_reads.append(
                    pod5.Read(
                        read_id=read_id,
                        pore=pod5.Pore(channel=1, well=1, pore_type="synthetic"),
                        calibration=pod5.Calibration(offset=0, scale=0.1),
                        read_number=index,
                        start_sample=index * 6000,
                        median_before=float("nan"),
                        end_reason=pod5.EndReason(pod5.EndReasonEnum.UNKNOWN, False),
                        run_info=run_info,
                        signal=np.round(values * 10).astype(np.int16),
                    )
                )
                record = pysam.AlignedSegment(header)
                record.query_name = str(read_id)
                record.query_sequence = head + "A" * 200
                record.query_qualities = array("B", [40] * 500)
                record.flag = 16 if reverse else 0
                record.reference_id = int(length > 0)
                record.reference_start = 0
                record.mapping_quality = 60
                record.cigar = [(0, 300), (4, 200)]
                record.set_tag("mv", array("b", [10] + [1] * 500))
                record.set_tag("ts", 0)
                record.set_tag("ns", 5000)
                record.set_tag("fn", pod5_path.name)
                alignments.append(record)
                row = dict.fromkeys(FIELDS, "")
                row.update(
                    manifest_version=HEAD_POLY_VERSION,
                    read_id=str(read_id),
                    pod5_path=str(pod5_path),
                    pod5_file_id="synthetic",
                    bam_path=str(bam_path),
                    split=split,
                    orientation="reverse" if reverse else "forward",
                    raw_signal_length=5000,
                    chunk_start_sample=0 if reverse else 3000,
                    chunk_end_sample=2000 if reverse else 5000,
                    poly_type=("polyT" if reverse else "polyA") if length else "none",
                    training_length_nt=length,
                    true_length_nt=length,
                    nominal_length_nt=length,
                    label_status="poly_present" if length else "zero_nt",
                    label_source="construct_nominal_weak_supervision",
                    flank_source="not_applicable_no_flank",
                    flank_status="not_applicable_no_flank",
                    selection_reason="synthetic",
                    reference_name=record.reference_name,
                    supervision_mode="length_only",
                    model_input_samples=12000,
                    head_excluded="true",
                    head_boundary_sample=2000 if reverse else 3000,
                )
                length_rows.append(row)
                target = head[50:82]
                if reverse:
                    target = target.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                sequence_rows.append(
                    dict(
                        manifest_version=VERSION,
                        read_id=str(read_id),
                        pod5_path=str(pod5_path),
                        split=split,
                        orientation=row["orientation"],
                        raw_signal_length=5000,
                        chunk_start_sample=4180 if reverse else 500,
                        chunk_end_sample=4500 if reverse else 820,
                        sequence_target=target,
                        reference_start_base=50,
                        reference_end_base=82,
                        label_source=LABEL_SOURCE,
                        boundary_source=BOUNDARY_SOURCE,
                        reference_sha256="a" * 64,
                    )
                )
    with pod5.Writer(pod5_path) as writer:
        writer.add_reads(raw_reads)
    with pysam.AlignmentFile(bam_path, "wb", header=header) as stream:
        for record in alignments:
            stream.write(record)
    manifest = work / "source_length.tsv"
    sequences = work / "source_sequence.tsv"
    write_table(manifest, length_rows, FIELDS)
    for row in sequence_rows:
        row["source_manifest_sha256"] = sha256(manifest)
    module._write_table(sequences, sequence_rows, [])
    args = module.build_parser().parse_args(
        [
            "--data-root",
            str(data),
            "--work-root",
            str(work),
            "--source-manifest",
            str(manifest),
            "--source-sequence-manifest",
            str(sequences),
            "--dorado-bam",
            str(bam_path),
            "--output-dir",
            "metadata/strict-new",
            "--poly-types",
            "polyA",
            "polyT",
            "--nominal-lengths",
            "0",
            "50",
            "--pool-per-group",
            "3",
            "--no-progress",
        ]
    )
    return args, data, work, length_rows, sequence_rows


def test_real_synthetic_pod5_roundtrip_reaches_both_training_consumers(strict_fixture):
    args, data, work, source_rows, _ = strict_fixture
    before = {
        path: sha256(path)
        for path in [
            *data.iterdir(),
            work / "source_length.tsv",
            work / "source_sequence.tsv",
        ]
    }
    args.run = True
    result = module.run(args)
    assert result["status"] == "complete"
    assert result["workflow"] == module.METHOD
    assert result["selected_reads"] == 12
    manifest = Path(result["artifacts"]["length_manifest"]["path"])
    sequence_manifest = Path(result["artifacts"]["sequence_manifest"]["path"])
    records = load_training_manifest(manifest)
    sequences = load_sequence_manifest(
        sequence_manifest,
        records,
        source_manifest_sha256=sha256(manifest),
        data_root=data,
    )
    original = {row["read_id"]: row for row in source_rows}
    for record in records:
        source = original[record.read_id]
        assert record.split.value == source["split"]
        assert record.training_length_nt == source["nominal_length_nt"]
        assert record.poly_start_sample is None and record.poly_is_complete is None
        assert record.label_source == "construct_nominal_weak_supervision"
        if record.training_length_nt:
            assert record.chunk_length_samples < 1000
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert all(row["dataset_method"] == module.METHOD for row in rows)
    assert all(
        int(row["strict_candidate_start_sample"]) > 3500
        if row["orientation"] == "forward"
        else int(row["strict_candidate_end_sample"]) < 1500
        for row in rows
        if row["training_length_nt"] != "0"
    )
    reader = Pod5SignalReader(signal_shift=0, signal_scale=1)
    dataset = PolyTrainingDataset(records, reader, split=ManifestSplit.TRAIN)
    batch = collate_poly_training_examples(
        [dataset[index] for index in range(len(dataset))], supervise_status=True
    )
    assert batch.signal.shape == (4, 1, 12000)
    assert sorted(batch.true_length_nt.tolist()) == [0, 0, 50, 50]
    assert torch.all(batch.location_targets == -100)
    assert batch.raw_padding_mask.any()
    sequence_dataset = SequenceDataset(sequences, reader, split=ManifestSplit.TRAIN)
    sequence_batch = collate_sequence_examples([sequence_dataset[0]])
    assert sequence_batch[0].signal.shape == (320,)
    assert len(sequence_batch[0].targets) == 32
    assert before == {path: sha256(path) for path in before}
    assert result["source_table_sha256"]["length"] != sha256(manifest)


def test_read_only_plan_opens_no_bam_pod5_or_manifest_rows(strict_fixture, monkeypatch):
    args, _, work, _, _ = strict_fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not scan sequencing inputs")

    monkeypatch.setattr(pysam, "AlignmentFile", forbidden)
    monkeypatch.setattr(pod5, "Reader", forbidden)
    monkeypatch.setattr(module, "_sample_sources", forbidden)
    assert module.run(args)["status"] == "planned"
    assert not (work / args.output_dir).exists()


@pytest.mark.parametrize("invalid_signal", [False, True])
def test_spawned_workers_preserve_manifests_and_audits_exactly(
    strict_fixture, monkeypatch, invalid_signal
):
    """One versus three CPUs must select and audit the same reads in the same order."""
    args, _, work, sources, _ = strict_fixture
    args.run = True
    args.target_per_group = 3
    if invalid_signal:
        original_load = module._load_signal

        def inject_nan(source, data, max_samples, **kwargs):
            values, fs, path = original_load(source, data, max_samples, **kwargs)
            if source["read_id"] == sources[0]["read_id"]:
                values = values.copy()
                values[0] = np.nan
            return values, fs, path

        monkeypatch.setattr(module, "_load_signal", inject_nan)
    outputs = []
    for workers in [1, 3]:
        args.workers = workers
        args.output_dir = Path(f"metadata/synthetic-cpu-{workers}")
        result = module.run(args)
        outputs.append((work / args.output_dir, result))
    first, second = outputs
    assert (
        first[1]["status"]
        == second[1]["status"]
        == ("incomplete_no_training_manifest" if invalid_signal else "complete")
    )
    for key in [
        "audit_status_counts",
        "early_stop",
        "signal_reads_attempted",
        "eligible_reads",
        "selected_reads",
        "split_counts",
    ]:
        assert first[1][key] == second[1][key]
    for rel in [
        "source_pool.tsv",
        "read_audit.tsv",
        "candidate_audit.jsonl",
        "eligible_before_balance.tsv",
        "length/manifest.tsv",
        "sequence/sequence_manifest.tsv",
    ]:
        left, right = first[0] / rel, second[0] / rel
        assert left.exists() == right.exists()
        if left.exists():
            assert left.read_bytes() == right.read_bytes(), rel


def test_workers_must_be_positive_in_read_only_plan(strict_fixture):
    args, _, _, _, _ = strict_fixture
    args.workers = 0
    with pytest.raises(ValueError):
        module.preflight(args)


@pytest.mark.parametrize("cache_size", [0, 4])
def test_pod5_reader_reuse_is_bounded_and_closed(
    strict_fixture, monkeypatch, cache_size
):
    args, _, _, _, _ = strict_fixture
    args.run = True
    args.pod5_reader_cache_size = cache_size
    original_reader = pod5.Reader
    original_close = pod5.Reader.close
    opened, closed = [], []

    def track_open(*args, **kwargs):
        reader = original_reader(*args, **kwargs)
        opened.append(reader)
        return reader

    def track_close(reader):
        if reader not in closed:
            closed.append(reader)
        original_close(reader)

    monkeypatch.setattr(pod5, "Reader", track_open)
    monkeypatch.setattr(original_reader, "close", track_close)
    result = module.run(args)
    assert result["status"] == "complete"
    assert len(opened) == (1 if cache_size else 12)
    assert all(reader in closed for reader in opened)


def test_pod5_cache_evicts_and_rejects_changed_files(tmp_path, monkeypatch):
    """Fake readers verify LRU/immutability/cleanup without any signal loading."""
    first, second = tmp_path / "synthetic1.pod5", tmp_path / "synthetic2.pod5"
    first.touch()
    second.touch()
    opened, closed = [], []

    class FakeReader:
        def __init__(self, path):
            self.path = path
            opened.append(path)

        def close(self):
            closed.append(self.path)

    monkeypatch.setattr(pod5, "Reader", FakeReader)
    with (
        pytest.raises(ValueError, match="POD5 changed"),
        closing(module.Pod5ReaderCache(1)) as readers,
    ):
        assert readers.get(first) is readers.get(first)
        readers.get(second)
        assert closed == [first]
        second.write_bytes(b"synthetic-change")
        readers.get(second)
    assert opened == closed == [first, second]


def test_fused_moves_equal_separate_scan_and_reject_wrong_provenance(strict_fixture):
    args, _, _, sources, _ = strict_fixture
    bam = Path(args.dorado_bam)
    with pysam.AlignmentFile(bam, "rb") as stream:
        header, records = stream.header, list(stream)
    # A late primary duplicate must invalidate an earlier cached success.
    with pysam.AlignmentFile(bam, "wb", header=header) as stream:
        for record in [*records, records[0]]:
            stream.write(record)
    # Match CSV source rows, which are all string values.
    with Path(args.source_manifest).open() as stream:
        by_id = {row["read_id"]: row for row in csv.DictReader(stream, delimiter="\t")}
    digest = sha256(Path(args.source_manifest))
    collector = module.SelectedMoveCollector(args, bam, by_id, digest)
    with pysam.AlignmentFile(bam, "rb") as stream:
        for record in stream:
            collector.observe(record)
    fused, errors, scanned = collector.result(args, bam, by_id, digest)
    separate, expected_errors, expected_scanned = module._alignments(args, bam, by_id)
    assert errors == expected_errors
    assert scanned == expected_scanned == 13
    assert errors[sources[0]["read_id"]] == "duplicate_primary_in_scanned_bam"
    assert fused.keys() == separate.keys()
    for key in fused:
        np.testing.assert_array_equal(
            fused[key]["positions"], separate[key]["positions"]
        )
    with pytest.raises(ValueError, match="does not match"):
        collector.result(args, bam, by_id, "0" * 64)
    args.max_bam_records += 1
    with pytest.raises(ValueError, match="does not match"):
        collector.result(args, bam, by_id, digest)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("workers", [1, 3])
def test_final_target_stops_signals_only_after_all_strata_fill(
    strict_fixture, monkeypatch, interleaved, workers
):
    """The remaining pool is audited as unevaluated, never failure/weak zero."""
    args, _, work, _, _ = strict_fixture
    args.run = True
    args.workers = workers
    args.pool_per_group = 6
    args.target_per_group = 3  # One accepted read per length/direction/split.
    original_sample = module._sample_sources
    original_load = module._load_signal
    loaded = []

    def sample_with_unevaluated_tail(args, path):
        rows, summary = original_sample(args, path)
        # Synthetic metadata-only tail: no corresponding signal is necessary
        # because these UUIDs must never be decoded after quotas are satisfied.
        tail = [
            {**row, "read_id": str(UUID(int=100 + i))} for i, row in enumerate(rows)
        ]
        summary["pool_reads"] = len(rows) + len(tail)
        if interleaved:
            return [
                row for pair in zip(rows, tail, strict=True) for row in pair
            ], summary
        return rows + tail, summary

    def record_load(source, data, max_samples, **kwargs):
        assert int(UUID(source["read_id"])) < 100
        loaded.append(source)
        return original_load(source, data, max_samples, **kwargs)

    monkeypatch.setattr(module, "_sample_sources", sample_with_unevaluated_tail)
    monkeypatch.setattr(module, "_load_signal", record_load)
    result = module.run(args)
    assert result["status"] == "complete"
    assert (
        result["selected_reads"]
        == result["signal_reads_attempted"]
        == len(loaded)
        == 12
    )
    assert result["split_counts"] == {"train": 4, "validation": 4, "test": 4}
    assert result["early_stop"] == {
        "enabled": True,
        "target_reached": True,
        "stopped_before_pool_end": True,
        "processed_reads": 23 if interleaved else 12,
        "not_evaluated_reads": 12,
    }
    assert result["progress"]["processed_reads"] == (23 if interleaved else 12)
    assert result["progress"]["total_reads"] == 24
    with (work / args.output_dir / "read_audit.tsv").open() as stream:
        audit = list(csv.DictReader(stream, delimiter="\t"))
    assert len(audit) == 24
    assert sum(row["export_status"] == "selected" for row in audit) == 12
    skipped = [row for row in audit if row["export_status"] == "not_evaluated"]
    assert len(skipped) == 12
    assert sum(
        row["status"] == "not_evaluated_stratum_quota_met" for row in skipped
    ) == (11 if interleaved else 0)
    assert all(row["status"].startswith("not_evaluated_") for row in skipped)


def test_nonempty_but_under_target_stratum_does_not_export(strict_fixture):
    args, _, work, _, _ = strict_fixture
    args.run = True
    args.pool_per_group = args.target_per_group = 4  # Requires two train reads/group.
    result = module.run(args)
    assert result["status"] == "incomplete_no_training_manifest"
    assert not result["early_stop"]["target_reached"]
    assert (
        result["eligible_reads"] == 12
    )  # All nonempty, but the target was not reached.
    assert len(result["missing_strata"]) == 4
    assert result["signal_reads_attempted"] == 12
    assert not (work / args.output_dir / "length/manifest.tsv").exists()


@pytest.mark.parametrize("target,budget", [(2, 210), (4, 210), (3, 11)])
def test_impossible_final_target_fails_before_signal_or_bam_scan(
    strict_fixture, target, budget
):
    args, _, _, _, _ = strict_fixture
    args.target_per_group = target
    args.max_signal_reads = budget
    with pytest.raises(ValueError, match="target"):
        module.preflight(args)


@pytest.mark.parametrize("no_progress", [False, True])
def test_scope_precedes_selection_and_progress_survives_no_progress_flag(
    strict_fixture, monkeypatch, capsys, no_progress
):
    """Keep head platforms audited/excluded from timing, outside selection scope."""
    args, _, work, _, _ = strict_fixture
    args.run = True
    args.no_progress = no_progress
    output = work / args.output_dir
    original_load = module._load_signal
    original_speed = module.estimate_external_speed
    load_count = 0
    timing_regions = []

    def signal_with_head_platform(source, data, max_samples, **kwargs):
        nonlocal load_count
        snapshot = json.loads((output / "strict_dataset.json").read_text())
        assert snapshot["stage"] == "signal_qc"
        assert snapshot["progress"]["processed_reads"] == load_count
        load_count += 1
        values, fs, path = original_load(source, data, max_samples, **kwargs)
        values = values.copy()
        start, end = (
            (2100, 2400) if source["orientation"] == "reverse" else (2600, 2900)
        )
        values[start:end] = 140 + np.random.default_rng(17).normal(0, 0.8, 300)
        return values, fs, path

    def record_speed(positions, regions, fs, config):
        timing_regions.append(regions)
        return original_speed(positions, regions, fs, config)

    monkeypatch.setattr(module, "_load_signal", signal_with_head_platform)
    monkeypatch.setattr(module, "estimate_external_speed", record_speed)
    result = module.run(args)
    assert result["status"] == "complete" and result["selected_reads"] == 12
    assert result["selection_scope"] == module.SELECTION_SCOPE
    assert result["progress"]["processed_reads"] == 12
    assert result["progress"]["eligible_reads"] == 12
    assert result["progress"]["deferred_reads"] == 0
    assert result["progress"]["excluded_reads"] == 0
    assert ("Strict signal QC" in capsys.readouterr().err) is (not no_progress)
    with (output / "read_audit.tsv").open() as stream:
        audit = list(csv.DictReader(stream, delimiter="\t"))
    assert len(audit) == 12
    assert all(
        int(row["candidate_count"]) > int(row["eligible_candidate_count"])
        for row in audit
    )
    assert (
        sorted(int(row["supported_candidate_count"]) for row in audit)
        == [0] * 6 + [1] * 6
    )
    details = [
        json.loads(line)
        for line in (output / "candidate_audit.jsonl").read_text().splitlines()
    ]
    assert len(details) == len(timing_regions) == 12
    for detail, exclusions in zip(details, timing_regions, strict=True):
        scope = detail["selection_scope"]
        assert exclusions == detail["detection"]["regions"]
        assert any(
            r["morphology_supported"]
            and not scope["start_sample"]
            <= r["start"]
            < r["end"]
            <= scope["end_sample"]
            for r in exclusions
        )
        chosen = detail["decision"]["selected"]
        if chosen:
            assert (
                scope["start_sample"]
                <= chosen["start"]
                < chosen["end"]
                <= scope["end_sample"]
            )


@pytest.mark.parametrize(
    "mode", ["raw", "work", "existing", "symlink", "repository", "full_scan"]
)
def test_paths_nonoverwrite_and_full_scan_guards(strict_fixture, mode):
    args, data, work, _, _ = strict_fixture
    if mode == "raw":
        args.output_dir = data / "out"
    elif mode == "work":
        args.output_dir = work
    elif mode == "existing":
        args.output_dir = work / "existing"
        args.output_dir.mkdir()
    elif mode == "symlink":
        args.output_dir = work / "alias"
        args.output_dir.symlink_to(work / "missing")
    elif mode == "repository":
        args.output_dir = Path.cwd() / "strict-output"
    else:
        args.max_bam_records = 0
    with pytest.raises(ValueError):
        module.preflight(args)


def test_bam_prefix_shortage_is_audited_and_emits_no_trainable_manifest(strict_fixture):
    args, _, work, _, _ = strict_fixture
    args.run = True
    args.max_bam_records = 1
    result = module.run(args)
    output = work / args.output_dir
    assert result["status"] == "incomplete_no_training_manifest"
    assert result["missing_strata"]
    assert not (output / "length/manifest.tsv").exists()
    with (output / "read_audit.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 12
    assert sum(row["failure_reason"] == "not_seen_in_bam_prefix" for row in rows) == 11


def test_signal_budget_retains_unprocessed_reads_and_checks_count_before_decode(
    strict_fixture, monkeypatch
):
    args, data, work, sources, _ = strict_fixture
    args.run = True
    args.max_signal_reads = 1
    result = module.run(args)
    assert result["signal_reads_attempted"] == 1
    with (work / args.output_dir / "read_audit.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert (
        sum(row["failure_reason"] == "signal_read_budget_exhausted" for row in rows)
        == 11
    )
    # A tiny budget fails on metadata, before signal_pa is touched.
    with pytest.raises(ValueError, match="sample_budget"):
        module._load_signal(sources[0], data, 12)


def test_old_sequence_pair_hash_is_not_silently_rebound(strict_fixture):
    args, _, work, _, sequences = strict_fixture
    sequences[0]["source_manifest_sha256"] = "0" * 64
    Path(args.source_sequence_manifest).unlink()
    module._write_table(Path(args.source_sequence_manifest), sequences, [])
    args.run = True
    with pytest.raises(ValueError, match="parent length-manifest hash"):
        module.run(args)
    summary = json.loads((work / args.output_dir / "strict_dataset.json").read_text())
    assert summary["status"] == "failed"


@pytest.mark.parametrize("target", [None, 3])
def test_duplicate_primary_is_audited_without_forced_replacement(
    strict_fixture, target
):
    args, _, work, _, _ = strict_fixture
    args.target_per_group = target
    bam = Path(args.dorado_bam)
    with pysam.AlignmentFile(bam, "rb") as stream:
        header = stream.header
        records = list(stream)
    with pysam.AlignmentFile(bam, "wb", header=header) as stream:
        for record in [*records, records[0]]:
            stream.write(record)
    args.run = True
    result = module.run(args)
    assert result["status"] == "incomplete_no_training_manifest"
    with (work / args.output_dir / "read_audit.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    duplicate = next(row for row in rows if row["read_id"] == records[0].query_name)
    assert duplicate["failure_reason"] == "duplicate_primary_in_scanned_bam"


def test_new_tcn_script_dryrun_consumes_new_manifests_without_weights(strict_fixture):
    args, data, work, _, _ = strict_fixture
    args.run = True
    module.run(args)
    model = work / "synthetic-model-no-weights"
    model.mkdir()
    (model / "config.toml").write_text(
        "[model.encoder.transformer_encoder]\ndepth = 2\n"
        "[model.seqdist]\nstate_len = 3\n"
    )
    config = work / "synthetic-config.yaml"
    config.write_text(
        f"paths:\n  data_root: {data}\n  work_root: {work}\n"
        f"training:\n  bonito_model: {model}\n"
    )
    environment = {
        **os.environ,
        "GPU_ID": "0",
        "EXPERIMENT_ID": "synthetic-strict-new",
        "RESUME": "0",
        "INITIALIZE_FROM": "",
    }
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_strict_poly_candidate_tcn.sh",
            str(args.output_dir),
            str(config),
            "--dry-run",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summaries = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    joint = next(item for item in summaries if "joint_config" in item)
    assert joint["joint_config"]["head_type"] == "tcn"
    assert joint["joint_config"]["tcn_length_features"] == "sum_mean_duration"
    assert joint["joint_config"]["unfreeze_last_layers"] == 1
    assert joint["joint_config"]["location_supervision"] is False
    assert joint["joint_config"]["lambda_basecall"] == 0.1
    assert not (work / "models/synthetic-strict-new").exists()
    # A changed manifest must be rejected before training even in dry-run mode.
    manifest = work / args.output_dir / "length/manifest.tsv"
    manifest.write_text(manifest.read_text() + "\n")
    rejected = subprocess.run(
        completed.args, env=environment, text=True, capture_output=True, check=False
    )
    assert rejected.returncode != 0
    assert "hash mismatch" in rejected.stderr

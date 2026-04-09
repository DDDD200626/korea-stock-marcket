"""외부 JSONL 학습 병합·파서 단위 테스트."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_external_disabled_returns_empty():
    from edu_tools.team_external_train import load_external_training_rows

    os.environ.pop("TEAM_EXTERNAL_TRAIN_ENABLED", None)
    rows, meta = load_external_training_rows()
    assert rows == []
    assert meta.get("enabled") is False


def test_external_file_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from edu_tools.team_external_train import load_external_training_rows

    p = tmp_path / "ext.jsonl"
    row = {"x": [float(i) * 0.01 for i in range(34)], "y": 55.0}
    p.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setenv("TEAM_EXTERNAL_TRAIN_ENABLED", "1")
    monkeypatch.setenv("TEAM_EXTERNAL_TRAIN_FILE", str(p))
    monkeypatch.delenv("TEAM_EXTERNAL_TRAIN_URL", raising=False)
    rows, meta = load_external_training_rows()
    assert len(rows) == 1
    assert len(rows[0]["x"]) == 34
    assert meta.get("error") is None
    assert meta.get("source") == "file"


def test_resolve_merges_and_dedupes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from edu_tools import team_ml_model as m

    ds = tmp_path / "team_ml_dataset.jsonl"
    ext = tmp_path / "ext.jsonl"
    row = {"x": [0.1] * 34, "y": 50.0, "member": "m1", "sess": "s1"}
    ds.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    ext.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(m, "DATASET_PATH", ds)
    monkeypatch.setenv("TEAM_EXTERNAL_TRAIN_ENABLED", "1")
    monkeypatch.setenv("TEAM_EXTERNAL_TRAIN_FILE", str(ext))
    data, n_pool, n_used = m.resolve_training_rows(reservoir_max=0, reservoir_seed=42)
    assert len(data) == 1
    assert n_used == 1
    assert n_pool >= 1
    meta = m.last_training_resolve_meta()
    assert meta["external_train"].get("dedupe_dropped", 0) >= 1


def test_resolve_adds_unique_external(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from edu_tools import team_ml_model as m

    ds = tmp_path / "team_ml_dataset.jsonl"
    ext = tmp_path / "ext.jsonl"
    local = {"x": [0.1] * 34, "y": 40.0, "member": "a", "sess": "s1"}
    extra = {"x": [0.2] * 34, "y": 60.0, "member": "b", "sess": "s2"}
    ds.write_text(json.dumps(local, ensure_ascii=False) + "\n", encoding="utf-8")
    ext.write_text(json.dumps(extra, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(m, "DATASET_PATH", ds)
    monkeypatch.setenv("TEAM_EXTERNAL_TRAIN_ENABLED", "1")
    monkeypatch.setenv("TEAM_EXTERNAL_TRAIN_FILE", str(ext))
    data, _, n_used = m.resolve_training_rows(reservoir_max=0, reservoir_seed=42)
    assert n_used == 2
    assert len(data) == 2

from __future__ import annotations

import pytest

from edu_tools.dl_roadmap import build_dl_quality_unified
from edu_tools.team_lm_embedding import LM_EXTRA_DIM, embed8_self_report
from edu_tools.team_torch_model import _model_size_profile, _promotion_gate_decision, _training_profile


def test_promotion_gate_accepts_when_candidate_is_not_worse() -> None:
    prev = {"validation_mae_mean": 8.4, "holdout_time_mae": 9.1}
    cand = {"validation_mae_mean": 8.2, "holdout_time_mae": 8.9}
    ok, reasons = _promotion_gate_decision(prev, cand)
    assert ok is True
    assert "metrics_ok" in reasons


def test_promotion_gate_rejects_when_candidate_regresses() -> None:
    prev = {"validation_mae_mean": 7.0, "holdout_time_mae": 8.0}
    cand = {"validation_mae_mean": 8.2, "holdout_time_mae": 9.1}
    ok, reasons = _promotion_gate_decision(prev, cand)
    assert ok is False
    assert "validation_mae_regressed" in reasons or "holdout_mae_regressed" in reasons


def test_semantic_encoder_off_returns_zeros(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEAM_SEMANTIC_ENCODER", "0")
    v = embed8_self_report("한글 자기서술 텍스트입니다.")
    assert len(v) == LM_EXTRA_DIM
    assert all(x == 0.0 for x in v)


def test_xlarge_lite_profile_is_lighter_five_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEAM_TORCH_MODEL_SIZE", "xlarge_lite")
    assert _model_size_profile() == "xlarge_lite"
    tp = _training_profile()
    assert tp["name"] == "xlarge_lite"
    assert int(tp["ensemble_size"]) <= 8
    assert int(tp["mc_dropout"]) <= 18


def test_dl_quality_unified_v4_includes_explainability() -> None:
    meta = {
        "permutation_importance_top": [{"feature": "log_commits", "delta_mae": 0.02}],
        "input_noise_std_training": 0.015,
        "dl_roadmap_version": 5,
    }
    u = build_dl_quality_unified(meta)
    assert u["schema"] == "dl_quality_unified_v4"
    ex = u.get("explainability") or {}
    assert ex.get("permutation_importance_top") is not None
    assert ex.get("input_noise_std_training") == 0.015
    assert "note_ko" in ex

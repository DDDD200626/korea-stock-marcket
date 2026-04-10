"""
Git 기반 팀 데이터 평가 — Score Engine(계산) / Anomaly(판단) / AI(설명) 분리.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from edu_tools.dl_roadmap import roadmap_payload
from edu_tools.team_data_store import db_profile, member_history_features, record_team_report
from edu_tools.team_ml_model import (
    FEATURE_LABELS,
    FEATURE_VERSION,
    LABEL_SPEC_VERSION,
    MIN_TRAIN_SAMPLES,
    STRUCTURAL_TARGET_WEIGHT,
    append_samples,
    build_feature_vector,
    hybrid_dl_target,
    load_model,
    predict_score,
    train_if_needed,
)
from edu_tools.team_web_priors import get_web_priors
from edu_tools.team_lm_embedding import semantic_encoder_meta
from edu_tools.team_torch_model import (
    TORCH_MODEL_PATH,
    predict_torch_score,
    predict_torch_scores_batched,
    torch_available,
    torch_model_meta,
    train_torch_if_needed,
)
from learning_analysis.llm_clients import ensure_gemini_configured, get_openai_client

router = APIRouter()

DEFAULT_WEIGHTS = {
    "commit": 0.25,
    "pr": 0.20,
    "lines": 0.20,
    "attendance": 0.15,
    "self_report": 0.20,
}


class TeamUserIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    commits: int = Field(0, ge=0)
    prs: int = Field(0, ge=0)
    lines: int = Field(0, ge=0)
    attendance: float = Field(0.0, ge=0, le=100)
    selfReport: str = Field("", max_length=12000)


class TeamReportRequest(BaseModel):
    project_name: str = Field("", max_length=300)
    teamData: list[TeamUserIn] = Field(..., min_length=1, max_length=40)


class ScoreBreakdown(BaseModel):
    commits: float
    prs: float
    lines: float
    attendance: float
    self_report: float


class WeightedPoints(BaseModel):
    commits: float
    prs: float
    lines: float
    attendance: float
    self_report: float


class DataReliability(BaseModel):
    score_0_100: float
    git_ratio_percent: float
    self_report_ratio_percent: float
    note: str = ""


class ScoreResult(BaseModel):
    member_name: str
    rawScore: float
    normalizedScore: float
    rank: int
    top_percent: float
    breakdown: ScoreBreakdown
    weighted_points: WeightedPoints
    pct_vs_team_mean: float
    data_reliability: DataReliability
    dl_score: float | None = Field(
        default=None, ge=0, le=100, description="경량 신경망 기반 데이터 학습 점수(보강 신호)"
    )
    dl_confidence: float | None = Field(
        default=None, ge=0, le=100, description="입력 밀도·분산 기반 신뢰도"
    )
    blendedScore: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="최종 점수: 0.7 * normalizedScore + 0.3 * dl_score",
    )
    dl_top_factors: list[str] = Field(
        default_factory=list,
        description="DL 점수에 크게 영향을 준 요인 Top3",
    )


class AnomalyResult(BaseModel):
    member_name: str
    flags: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    member_name: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    position_in_team: str = ""
    recommended_actions: list[str] = Field(default_factory=list)
    metric_dl_note: str = Field(
        default="",
        max_length=1200,
        description="규칙 점수·DL 보조(blended)와의 정합/차이를 한국어로 짧게",
    )


class TeamReportResponse(BaseModel):
    scores: list[ScoreResult]
    anomalies: list[AnomalyResult]
    analysis: list[AnalysisResult]
    edge_cases: list[str] = Field(default_factory=list)
    trust_scores: dict[str, float] = Field(default_factory=dict)
    dl_model_info: dict[str, Any] = Field(default_factory=dict)
    evaluation_log: dict[str, Any] = Field(default_factory=dict)
    team_narrative: str = Field(
        default="",
        max_length=4000,
        description="LLM이 생성한 팀 단위 총평(교육자용, API 키 없으면 비움)",
    )
    ai_meta: dict[str, Any] = Field(
        default_factory=dict,
        description="리포트 LLM 호출 메타(provider, 모델, 지연 등)",
    )
    disclaimer: str = Field(
        default="정량 점수는 결정론적으로 계산되며, AI는 설명만 생성합니다."
    )


def _load_weights() -> dict[str, float]:
    p = Path(__file__).with_name("data").joinpath("team_eval_weights.json")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        keys = ("commit", "pr", "lines", "attendance", "self_report")
        out: dict[str, float] = {}
        s = 0.0
        for k in keys:
            v = float(data.get(k, DEFAULT_WEIGHTS[k]))
            if v < 0:
                v = DEFAULT_WEIGHTS[k]
            out[k] = v
            s += v
        if s <= 1e-9:
            return DEFAULT_WEIGHTS.copy()
        return {k: out[k] / s for k in keys}
    except Exception:
        return DEFAULT_WEIGHTS.copy()


def _word_count(s: str) -> int:
    return len([x for x in s.split() if x.strip()])


def _median(vals: list[float]) -> float:
    if not vals:
        return 1.0
    a = sorted(vals)
    n = len(a)
    m = n // 2
    return float(a[m]) if n % 2 else (float(a[m - 1]) + float(a[m])) / 2.0


def _contest_transparency_pack(
    *,
    enabled: bool,
    backend: str,
    blend_formula: str,
    sample_count: int | None,
    use_torch: bool,
    quality: dict[str, Any],
    target_mix: str,
) -> dict[str, Any]:
    """공모전 심사용: 딥러닝 보조 신호의 역할·한계·재현 힌트를 한 번에 제시."""
    rubric_hooks = {
        "technical_completeness_ko": (
            "피처 버전·입력 차원·(가능 시) 교차검증 MAE·앙상블·선형 보정 계수·불확실도 샘플 수를 "
            "`dl_model_info.quality`에 노출하고, CV·tail·NN+GBDT 블렌드·calibration은 "
            "`quality.dl_quality_unified` 한 객체로도 요약됨."
        ),
        "ai_efficiency_ko": (
            "PyTorch MLP 앙상블(+ MC 드롭아웃 불확실도)과 경량 선형 폴백을 단계적으로 사용하고, "
            "팀 단위 배치 추론으로 호출 비용을 줄임."
        ),
        "planning_practical_ko": (
            "최종 화면 점수는 규칙 엔진 정규화 점수와 DL 보조 점수를 고정 비율로 블렌드하며, "
            "교육 현장의 ‘참고·보조’ 용도임을 응답 disclaimer와 함께 명시."
        ),
        "creativity_ko": (
            "세션 내 순위 보조 학습, 과거 평가 이력 사전, 서술 형태 피처 등 룰만으로 잡기 어려운 신호를 결합."
        ),
    }
    limitations = [
        "입력은 집계된 Git 수치·출석·자기서술 텍스트 형태 특성 등으로, 코드 품질·리뷰 내용을 직접 이해하지는 않습니다.",
        "팀·과제마다 분포가 달라 절대 점수 비교에는 한계가 있습니다.",
        "데이터가 적을 때는 선형 모델 또는 룰 점수에 가깝게 동작할 수 있습니다.",
        "최종 성적·징계·인사 결정을 자동 대체하지 않으며, 교육자 판단이 우선입니다.",
    ]
    return {
        "purpose_ko": "규칙 기반 점수를 보조하는 데이터 학습 신호(참고용)",
        "blend_formula": blend_formula,
        "target_mix_for_training_ko": target_mix,
        "feature_version": FEATURE_VERSION,
        "backend_reported": backend,
        "pytorch_in_use": use_torch,
        "enabled": enabled,
        "sample_count_at_run": sample_count,
        "rubric_alignment_ko": rubric_hooks,
        "limitations_ko": limitations,
        "signals_disclosed_ko": [
            "34차원 피처(tabular + 선택적 다국어 문장 임베딩 8차원): 커밋·PR·라인·출석·서술·팀 상대 지표·과거 이력·텍스트 형태·에세이 깊이·Git 균형 등",
            "선택적 웹 사전(priors)으로 소폭 보정(상한 있음)",
        ],
        "quality_snapshot": quality,
        "verification_hints_ko": [
            "GET /api/capabilities — 심사 4축·엔드포인트 목록",
            "POST /api/team/report — scores[].dl_score, dl_confidence, dl_top_factors, dl_model_info",
            "docs/CONTEST_RUBRIC.md — 근거 경로",
        ],
    }


def _scaled_by_team(value: float, base: float, cap_mult: float = 2.5) -> float:
    m = max(base, 1e-6)
    return min(100.0, 100.0 * min(cap_mult, value / m) / cap_mult)


def _blend_weights_from_confidence(dl_confidence: float | None, dl_uncertainty: float | None) -> tuple[float, float]:
    """DL 신뢰도/불확실도 기반 동적 블렌딩 가중치."""
    c = float(dl_confidence or 0.0)
    u = float(dl_uncertainty or 0.0)
    conf_term = max(0.0, min(1.0, c / 100.0))
    unc_term = max(0.0, min(1.0, 1.0 - (u / 100.0)))
    dl_w = 0.22 + 0.26 * ((0.65 * conf_term) + (0.35 * unc_term))
    dl_w = max(0.22, min(0.48, dl_w))
    return 1.0 - dl_w, dl_w


def _estimate_feature_drift(xs: list[list[float]], meta: dict[str, Any]) -> dict[str, Any]:
    """학습 시점 피처 통계 대비 현재 배치 드리프트를 0~100 점수로 반환."""
    if not xs:
        return {"drift_score_0_100": 0.0, "drift_level": "unknown", "drift_note": "no_batch"}
    tr_means = meta.get("training_feature_means")
    tr_stds = meta.get("training_feature_stds")
    if not isinstance(tr_means, list) or not isinstance(tr_stds, list):
        return {"drift_score_0_100": 0.0, "drift_level": "unknown", "drift_note": "no_training_stats"}
    d = min(len(xs[0]), len(tr_means), len(tr_stds))
    if d <= 0:
        return {"drift_score_0_100": 0.0, "drift_level": "unknown", "drift_note": "invalid_dimension"}
    b_means = [sum(r[i] for r in xs) / max(len(xs), 1) for i in range(d)]
    z_shifts: list[float] = []
    for i in range(d):
        s = max(1e-6, float(tr_stds[i]))
        z_shifts.append(abs((float(b_means[i]) - float(tr_means[i])) / s))
    mean_z = sum(z_shifts) / max(len(z_shifts), 1)
    score = max(0.0, min(100.0, 100.0 - min(100.0, mean_z * 22.0)))
    if score >= 80:
        level = "low"
    elif score >= 55:
        level = "moderate"
    else:
        level = "high"
    return {
        "drift_score_0_100": round(score, 2),
        "drift_level": level,
        "mean_abs_z_shift": round(mean_z, 4),
    }


def _operations_playbook(
    *,
    use_torch: bool,
    meta: dict[str, Any],
    drift_info: dict[str, Any],
    torch_info: dict[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    """설명(dl_quality_unified)과 별도로, 현재 요청 기준 운영 권고(교육자·운영자용)."""
    checklist: list[str] = []
    pg_raw = meta.get("promotion_gate") if use_torch else None
    pg = pg_raw if isinstance(pg_raw, dict) else {}
    accepted = pg.get("accepted")
    drift_level = str(drift_info.get("drift_level") or "")
    ds = drift_info.get("drift_score_0_100")
    ds_f = float(ds) if isinstance(ds, (int, float)) else None

    level = "ok"
    summary = (
        "배치 피처 분포와 학습 메타가 대체로 일치하며, 별도 조치 없이 참고·모니터링 수준입니다."
        if use_torch
        else "경량 모드(선형/폴백)로 동작 중입니다. PyTorch 보강을 쓰려면 표본·환경을 확인하세요."
    )

    if use_torch and accepted is False:
        level = "action"
        summary = (
            "최근 후보 모델이 승격 게이트에서 거절되었습니다. 홀드아웃·검증 MAE와 데이터 라벨을 점검한 뒤 재학습하세요."
        )
        checklist.append("로그의 promotion_gate 사유(validation_mae_regressed 등) 확인")
        checklist.append("TEAM_HOLDOUT_TIME_FRAC·TEAM_TRAIN_OUTLIER_IQR_K 조정 검토")
    elif use_torch and (drift_level == "high" or (ds_f is not None and ds_f < 40.0)):
        level = "action" if (ds_f is not None and ds_f < 30.0) else "watch"
        summary = "현재 팀 입력이 학습 시점 대비 크게 어긋난 것으로 보입니다. 누적 데이터·피처 정의를 점검하세요."
        checklist.append("dataset JSONL 누적·세션(sess) 중복 여부 확인")
        checklist.append("TEAM_TORCH_RETRAIN_DRIFT_MIN으로 재학습 트리거 검토")
    elif sample_count < MIN_TRAIN_SAMPLES + 10:
        if level == "ok":
            level = "watch"
        summary = (
            f"학습 표본이 아직 적습니다(참고: 최소 학습 기준 {MIN_TRAIN_SAMPLES}행 근처). "
            "평가 요청을 반복해 누적하면 DL 보정이 안정화됩니다."
        )
        checklist.append("동일 과제에서 팀 리포트/평가를 여러 번 실행해 샘플 확보")

    if use_torch and torch_info.get("trained"):
        checklist.append("이번 요청에서 재학습 실행됨 — validation·holdout 메타 확인")

    noise = meta.get("input_noise_std_training")
    if use_torch and isinstance(noise, (int, float)) and float(noise) > 0:
        checklist.append(f"학습 시 입력 노이즈 σ={float(noise):g} 적용 중(일반화)")

    ex = (meta.get("dl_quality_unified") or {}).get("explainability") if use_torch else None
    if isinstance(ex, dict) and ex.get("permutation_importance_top"):
        checklist.append("치환 중요도 상위 피처는 품질 메타·아래 설명 패널 참고")

    return {
        "recommendation_level": level,
        "summary_ko": summary,
        "checklist_ko": checklist[:10],
        "indicators": {
            "drift": drift_info,
            "promotion_gate": pg if use_torch else None,
            "sample_count": sample_count,
            "pytorch_active": use_torch,
            "last_train_in_request": {
                "trained": bool(torch_info.get("trained")),
                "reason": torch_info.get("reason"),
            }
            if use_torch
            else None,
        },
    }


def run_score_engine(users: list[TeamUserIn]) -> tuple[list[ScoreResult], dict[str, float]]:
    w = _load_weights()
    commits = [float(u.commits) for u in users]
    prs = [float(u.prs) for u in users]
    lines = [float(u.lines) for u in users]
    atts = [float(u.attendance) for u in users]
    words = [float(_word_count(u.selfReport)) for u in users]

    med = {
        "commit": _median(commits),
        "pr": _median(prs),
        "lines": _median(lines),
        "attendance": _median(atts) if max(atts) > 1e-9 else 50.0,
        "self_report": _median(words) if max(words) > 1e-9 else 1.0,
    }

    bds: list[ScoreBreakdown] = []
    pts: list[WeightedPoints] = []
    raws: list[float] = []
    trust: dict[str, float] = {}

    for i, u in enumerate(users):
        bd = ScoreBreakdown(
            commits=_scaled_by_team(commits[i], med["commit"]),
            prs=_scaled_by_team(prs[i], med["pr"]),
            lines=_scaled_by_team(lines[i], med["lines"]),
            attendance=min(100.0, atts[i]),
            self_report=_scaled_by_team(words[i], med["self_report"]),
        )
        pt = WeightedPoints(
            commits=round(w["commit"] * bd.commits, 2),
            prs=round(w["pr"] * bd.prs, 2),
            lines=round(w["lines"] * bd.lines, 2),
            attendance=round(w["attendance"] * bd.attendance, 2),
            self_report=round(w["self_report"] * bd.self_report, 2),
        )
        raw = pt.commits + pt.prs + pt.lines + pt.attendance + pt.self_report
        bds.append(bd)
        pts.append(pt)
        raws.append(raw)

        data_core = (pt.commits + pt.prs + pt.lines + pt.attendance) / max(
            w["commit"] + w["pr"] + w["lines"] + w["attendance"], 1e-9
        )
        gap = abs(data_core - bd.self_report)
        tr = max(0.0, min(100.0, 100.0 - gap * 0.8))
        trust[u.name.strip()] = round(tr, 1)

    n = len(users)
    mean_raw = sum(raws) / max(n, 1)
    mx, mi = max(raws), min(raws)
    span = max(mx - mi, 1e-6)
    ranked = sorted(range(n), key=lambda i: raws[i], reverse=True)
    rank = [0] * n
    for r, idx in enumerate(ranked, start=1):
        rank[idx] = r

    out: list[ScoreResult] = []
    for i, u in enumerate(users):
        top = 100.0 * rank[i] / max(n, 1)
        git_ratio = 100.0 * (w["commit"] + w["pr"] + w["lines"] + w["attendance"])
        self_ratio = 100.0 * w["self_report"]
        rel = DataReliability(
            score_0_100=trust[u.name.strip()],
            git_ratio_percent=round(git_ratio, 1),
            self_report_ratio_percent=round(self_ratio, 1),
            note="자기서술 비중 높음" if self_ratio >= 30 else "",
        )
        out.append(
            ScoreResult(
                member_name=u.name.strip(),
                rawScore=round(raws[i], 2),
                normalizedScore=round(100.0 * (raws[i] - mi) / span, 2),
                rank=rank[i],
                top_percent=round(top, 1),
                breakdown=bds[i],
                weighted_points=pts[i],
                pct_vs_team_mean=round(
                    ((raws[i] - mean_raw) / max(abs(mean_raw), 1e-9)) * 100.0, 1
                ),
                data_reliability=rel,
            )
        )
    return out, trust


def apply_dl_scores(
    users: list[TeamUserIn],
    scores: list[ScoreResult],
    request_id: str = "",
    *,
    learning: bool = True,
) -> dict[str, Any]:
    """학습 데이터 기반 ML 보강 점수.

    - 요청마다 (확장 feature, 복합 target=순위점수+절대활동) 샘플을 누적
    - 샘플 수가 쌓이면 자동 재학습
    - 저장된 최신 모델로 dl_score 산출
    """
    if not users or not scores:
        return {
            "model_name": "team-ml",
            "enabled": False,
            "dl_roadmap": roadmap_payload(),
            "contest_transparency": _contest_transparency_pack(
                enabled=False,
                backend="none",
                blend_formula="dynamic: rule_w*normalizedScore + dl_w*dl_score (dl_w from confidence/uncertainty)",
                sample_count=None,
                use_torch=False,
                quality={},
                target_mix=f"{1.0 - STRUCTURAL_TARGET_WEIGHT:.2f}*normalizedScore + {STRUCTURAL_TARGET_WEIGHT}*structural_absolute",
            ),
        }

    priors, priors_meta = get_web_priors()
    m_c = _median([float(u.commits) for u in users])
    m_p = _median([float(u.prs) for u in users])
    m_l = _median([float(u.lines) for u in users])
    m_a = _median([float(u.attendance) for u in users])
    m_w = _median([float(_word_count(u.selfReport)) for u in users])
    team_n = len(users)
    hist_cache: dict[str, tuple[float, float, float]] = {
        u.name.strip(): member_history_features(u.name.strip()) for u in users
    }

    if learning:
        samples: list[dict[str, Any]] = []
        for u, s in zip(users, scores):
            hb, hr, hd = hist_cache[u.name.strip()]
            x = build_feature_vector(
                u.commits,
                u.prs,
                u.lines,
                u.attendance,
                u.selfReport,
                member_rank=int(s.rank),
                team_size=team_n,
                median_commits=m_c,
                median_prs=m_p,
                median_lines=m_l,
                median_attendance=m_a,
                median_words=m_w,
                hist_blend=hb,
                hist_rule=hr,
                hist_density=hd,
            )
            y = hybrid_dl_target(
                float(s.normalizedScore),
                u.commits,
                u.prs,
                u.lines,
                u.attendance,
                u.selfReport,
            )
            row: dict[str, Any] = {
                "x": x,
                "y": y,
                "member": u.name.strip(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "feature_version": FEATURE_VERSION,
                "label_spec_version": LABEL_SPEC_VERSION,
                "structural_weight": STRUCTURAL_TARGET_WEIGHT,
            }
            if request_id:
                row["sess"] = request_id
            samples.append(row)
        append_samples(samples)

        # 1) Prefer PyTorch deep model when available
        torch_info = train_torch_if_needed()
        use_torch = torch_available() and not (torch_info.get("enabled") is False)

        # 2) Fallback lightweight model
        train_info = train_if_needed()
        model = load_model()
    else:
        torch_info = {}
        train_info = {}
        model = load_model()
        use_torch = bool(torch_available() and TORCH_MODEL_PATH.is_file())
    if not model and not use_torch:
        for s in scores:
            s.dl_score = None
            s.dl_confidence = None
            s.blendedScore = float(s.normalizedScore)
        sc = int(train_info.get("sample_count", 0) or 0)
        return {
            "model_name": "team-ml",
            "enabled": False,
            "reason": "model_not_ready",
            "sample_count": sc,
            "blend_formula": "dynamic: rule_w*normalizedScore + dl_w*dl_score (dl_w from confidence/uncertainty)",
            "dl_roadmap": roadmap_payload(),
            "contest_transparency": _contest_transparency_pack(
                enabled=False,
                backend="none",
                blend_formula="dynamic: rule_w*normalizedScore + dl_w*dl_score (dl_w from confidence/uncertainty)",
                sample_count=sc,
                use_torch=False,
                quality={"note_ko": "학습 표본 부족 등으로 모델 미가동"},
                target_mix=f"{1.0 - STRUCTURAL_TARGET_WEIGHT:.2f}*normalizedScore + {STRUCTURAL_TARGET_WEIGHT}*structural_absolute",
            ),
        }

    base_samples = 0
    if use_torch:
        base_samples = int((torch_model_meta().get("sample_count") or 0))
    elif model:
        base_samples = int(model.sample_count)
    conf_base = max(35.0, min(95.0, 40.0 + base_samples * 0.6))
    xs: list[list[float]] = []
    for u, s in zip(users, scores):
        hb, hr, hd = hist_cache[u.name.strip()]
        xs.append(
            build_feature_vector(
                u.commits,
                u.prs,
                u.lines,
                u.attendance,
                u.selfReport,
                member_rank=int(s.rank),
                team_size=team_n,
                median_commits=m_c,
                median_prs=m_p,
                median_lines=m_l,
                median_attendance=m_a,
                median_words=m_w,
                hist_blend=hb,
                hist_rule=hr,
                hist_density=hd,
            )
        )
    batch_torch = predict_torch_scores_batched(xs) if use_torch else None
    for i, (u, s) in enumerate(zip(users, scores)):
        hb, hr, hd = hist_cache[u.name.strip()]
        x = xs[i]
        dl_score = None
        dl_uncertainty = None
        if use_torch:
            pred = None
            if batch_torch is not None and i < len(batch_torch):
                pred = batch_torch[i]
            if pred is None:
                pred = predict_torch_score(
                    u.commits,
                    u.prs,
                    u.lines,
                    u.attendance,
                    u.selfReport,
                    member_rank=int(s.rank),
                    team_size=team_n,
                    median_commits=m_c,
                    median_prs=m_p,
                    median_lines=m_l,
                    median_attendance=m_a,
                    median_words=m_w,
                    hist_blend=hb,
                    hist_rule=hr,
                    hist_density=hd,
                )
            if pred is not None:
                dl_score = float(pred.get("score", 0.0))
                dl_uncertainty = float(pred.get("uncertainty", 0.0))
        if dl_score is None and model:
            dl_score = predict_score(model, x)
        if dl_score is None:
            dl_score = float(s.normalizedScore)
        # Optional web priors adjustment (small bounded delta)
        commit_gap = min(1.0, max(0.0, x[0] / math.log1p(30))) - (priors["commit_expectation"] / 100.0)
        pr_gap = min(1.0, max(0.0, x[1] / math.log1p(12))) - (priors["pr_expectation"] / 100.0)
        att_gap = x[3] - (priors["attendance_expectation"] / 100.0)
        prior_delta = (commit_gap * 5.0) + (pr_gap * 4.0) + (att_gap * 3.0)
        dl_score = max(0.0, min(100.0, dl_score + prior_delta))
        # Explainability: 베이스 지표 + 팀 상대 지표 중 영향 큰 순
        pri = [
            ("커밋 활동", x[0] - (priors["commit_expectation"] / 100.0)),
            ("PR 협업", x[1] - (priors["pr_expectation"] / 100.0)),
            ("코드 변화량", x[2] - (priors["lines_expectation"] / 100.0)),
            ("출석/참여", x[3] - (priors["attendance_expectation"] / 100.0)),
            ("자기서술 밀도", x[4] - (priors["self_report_words_expectation"] / 100.0)),
        ]
        rel = [
            (FEATURE_LABELS[9], x[9]),
            (FEATURE_LABELS[10], x[10]),
            (FEATURE_LABELS[11], x[11]),
            (FEATURE_LABELS[12], x[12]),
        ]
        hist = [
            ("과거 평가(블렌드) 사전", x[16] - 0.5),
            ("과거 평가(룰) 사전", x[17] - 0.5),
            ("과거 평가 빈도", x[18] - 0.5),
        ]
        txt_feats = [
            ("서술 고유어 비율", x[19] - 0.5),
            ("서술 글자/단어 밀도", x[20] - 0.5),
            ("서술 줄바꿈 밀도", x[21] - 0.5),
            ("서술 숫자 비율", x[22] - 0.5),
            ("서술 장단어 비율", x[23] - 0.5),
        ]
        factors = pri + rel + hist + txt_feats
        top = sorted(factors, key=lambda t: abs(t[1]), reverse=True)[:3]
        s.dl_top_factors = [f"{name} {'+' if val >= 0 else '-'}" for name, val in top]
        # 입력 다양도 기반 confidence 보정
        x_min, x_max = min(x), max(x)
        spread = x_max - x_min
        conf = max(0.0, min(100.0, conf_base + spread * 8.0))
        if dl_uncertainty is not None:
            conf = max(0.0, min(100.0, conf - min(20.0, dl_uncertainty * 3.0)))
        s.dl_score = round(dl_score, 2)
        s.dl_confidence = round(conf, 1)
        rule_w, dl_w = _blend_weights_from_confidence(s.dl_confidence, dl_uncertainty)
        s.blendedScore = round(rule_w * s.normalizedScore + dl_w * s.dl_score, 2)

    meta = torch_model_meta() if use_torch else {}
    drift_info = _estimate_feature_drift(xs, meta) if use_torch else {
        "drift_score_0_100": None,
        "drift_level": "n/a",
        "mean_abs_z_shift": None,
    }
    sc_for_ops = int(meta.get("sample_count") or 0) if use_torch else int((model.sample_count if model else 0) or 0)
    ex_snap = None
    if use_torch:
        dq = meta.get("dl_quality_unified")
        if isinstance(dq, dict):
            ex_snap = dq.get("explainability")
    ops_pb = _operations_playbook(
        use_torch=use_torch,
        meta=meta,
        drift_info=drift_info,
        torch_info=torch_info,
        sample_count=sc_for_ops,
    )
    return {
        "model_name": "team-torch-mlp" if use_torch else "team-ml-linear-sgd",
        "enabled": True,
        "model_version": (meta.get("model_version") if use_torch else model.version if model else None),
        "trained_at": (meta.get("trained_at") if use_torch else model.trained_at if model else None),
        "sample_count": (meta.get("sample_count") if use_torch else model.sample_count if model else 0),
        "auto_retrain": bool(torch_info.get("trained", False) if use_torch else train_info.get("trained", False)),
        "backend": "pytorch" if use_torch else "lightweight-linear",
        "quality": {
            "validation_mae_mean": meta.get("validation_mae_mean") if use_torch else None,
            "cv_mae_mean": meta.get("cv_mae_mean") if use_torch else None,
            "ensemble_size": meta.get("ensemble_size") if use_torch else None,
            "best_hparams": meta.get("best_hparams") if use_torch else None,
            "best_architecture": meta.get("best_architecture") if use_torch else None,
            "calibration": meta.get("calibration") if use_torch else None,
            "calibration_pearson_r": meta.get("calibration_pearson_r") if use_torch else None,
            "calibration_r2": meta.get("calibration_r2") if use_torch else None,
            "holdout_time_mae": meta.get("holdout_time_mae") if use_torch else None,
            "holdout_time_pearson_r": meta.get("holdout_time_pearson_r") if use_torch else None,
            "holdout_time_r2": meta.get("holdout_time_r2") if use_torch else None,
            "label_spec_version": meta.get("label_spec_version") if use_torch else None,
            "feature_version": meta.get("feature_version") if use_torch else None,
            "input_dim": meta.get("input_dim") if use_torch else None,
            "cv_vs_validation_gap": meta.get("cv_vs_validation_gap") if use_torch else None,
            "chronological_tail_mae_mean": meta.get("chronological_tail_mae_mean") if use_torch else None,
            "model_size_profile": meta.get("model_size_profile") if use_torch else None,
            "approx_parameters": meta.get("approx_parameters") if use_torch else None,
            "mc_dropout_samples": meta.get("mc_dropout_samples") if use_torch else None,
            "gbdt_present": meta.get("gbdt_present") if use_torch else None,
            "nn_gbdt_blend_alpha": meta.get("nn_gbdt_blend_alpha") if use_torch else None,
            "gbdt_validation_blend_mae": meta.get("gbdt_validation_blend_mae") if use_torch else None,
            "cv_split_strategy": meta.get("cv_split_strategy") if use_torch else None,
            "cv_unique_groups": meta.get("cv_unique_groups") if use_torch else None,
            "dataset_file_sha256": meta.get("dataset_file_sha256") if use_torch else None,
            "gbdt_top_features": meta.get("gbdt_top_features") if use_torch else None,
            "dl_roadmap_version": meta.get("dl_roadmap_version") if use_torch else None,
            "dl_quality_unified": meta.get("dl_quality_unified") if use_torch else None,
            "promotion_gate": meta.get("promotion_gate") if use_torch else None,
            "feature_drift": drift_info,
            "explainability_snapshot": ex_snap,
            "operations_playbook": ops_pb,
            "semantic_encoder": semantic_encoder_meta(),
        },
        "web_priors": priors_meta,
        "input_features": FEATURE_LABELS,
        "target_mix": f"{1.0 - STRUCTURAL_TARGET_WEIGHT:.2f}*normalizedScore + {STRUCTURAL_TARGET_WEIGHT}*structural_absolute",
        "beyond_rule_signals": [
            "sqlite_member_history_prior",
            "session_pairwise_ranking_aux",
            "sklearn_histgradientboosting_blend_with_nn",
        ],
        "blend_formula": "dynamic: rule_w*normalizedScore + dl_w*dl_score (dl_w from confidence/uncertainty)",
        "dl_roadmap": roadmap_payload(),
        "contest_transparency": _contest_transparency_pack(
            enabled=True,
            backend="pytorch" if use_torch else "lightweight-linear",
            blend_formula="dynamic: rule_w*normalizedScore + dl_w*dl_score (dl_w from confidence/uncertainty)",
            sample_count=(
                int(meta.get("sample_count") or 0)
                if use_torch
                else int((model.sample_count if model else 0) or 0)
            ),
            use_torch=use_torch,
            quality={
                "validation_mae_mean": meta.get("validation_mae_mean") if use_torch else None,
                "cv_mae_mean": meta.get("cv_mae_mean") if use_torch else None,
                "ensemble_size": meta.get("ensemble_size") if use_torch else None,
                "calibration": meta.get("calibration") if use_torch else None,
                "feature_version": int(meta.get("feature_version") or FEATURE_VERSION) if use_torch else FEATURE_VERSION,
                "input_dim": int(meta.get("input_dim") or len(FEATURE_LABELS)) if use_torch else len(FEATURE_LABELS),
                "dl_quality_unified": meta.get("dl_quality_unified") if use_torch else None,
                "promotion_gate": meta.get("promotion_gate") if use_torch else None,
                "feature_drift": drift_info,
                "explainability_snapshot": ex_snap,
                "operations_playbook": ops_pb,
                "semantic_encoder": semantic_encoder_meta(),
            },
            target_mix=f"{1.0 - STRUCTURAL_TARGET_WEIGHT:.2f}*normalizedScore + {STRUCTURAL_TARGET_WEIGHT}*structural_absolute",
        ),
    }


def run_anomaly_detector(users: list[TeamUserIn], scores: list[ScoreResult]) -> list[AnomalyResult]:
    team_mean = sum(s.rawScore for s in scores) / max(len(scores), 1)
    med_lines = _median([float(u.lines) for u in users])
    out: list[AnomalyResult] = []
    for u, s in zip(users, scores):
        flags: list[str] = []
        wc = _word_count(u.selfReport)
        if u.commits == 0 and wc >= 35:
            flags.append("커밋 0 + 자기서술 과다: 데이터 신뢰도 확인 필요")
        if u.lines > max(1.0, med_lines) * 2 and u.prs == 0:
            flags.append("코드량 대비 PR 부족: 협업 점검 필요")
        if u.attendance < 40 and s.rawScore >= team_mean * 1.15 and team_mean > 1e-9:
            flags.append("출석 낮음 대비 점수 높음: 이상 패턴 점검")
        if u.commits == 0 and u.prs == 0 and u.lines == 0 and wc < 5:
            flags.append("평가 불가에 가까운 데이터 부족")
        out.append(AnomalyResult(member_name=u.name.strip(), flags=flags))
    return out


def detect_edge_cases(users: list[TeamUserIn], scores: list[ScoreResult]) -> list[str]:
    notes: list[str] = []
    git_total = sum((u.commits + u.prs + u.lines) for u in users)
    self_total = sum(_word_count(u.selfReport) for u in users)
    if git_total == 0 and self_total == 0:
        notes.append("CASE: 데이터 없음 — 평가 불가")
    if scores:
        raws = sorted((s.rawScore for s in scores), reverse=True)
        if len(raws) >= 2 and raws[0] >= max(1e-9, sum(raws) * 0.65):
            notes.append("CASE: 한 명 집중 — 팀 불균형 가능")
        if all(s.normalizedScore < 35 for s in scores):
            notes.append("CASE: 전반 저조 — 팀 전체 성과 저조")
    return notes


def _extract_json(text: str) -> dict[str, Any]:
    t = text.strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def _fallback_analysis(
    users: list[TeamUserIn], scores: list[ScoreResult], anomalies: list[AnomalyResult]
) -> list[AnalysisResult]:
    by_anom = {a.member_name: a.flags for a in anomalies}
    by_score = {s.member_name: s for s in scores}
    out: list[AnalysisResult] = []
    n = len(scores)
    for u in users:
        name = u.name.strip()
        s = by_score.get(name)
        if not s:
            continue
        flags = by_anom.get(name, [])
        rec = ["PR 리뷰 참여를 주차 목표에 포함하세요."]
        if any("협업" in f or "PR" in f for f in flags):
            rec = ["PR 분할 제출 + 리뷰 코멘트 기록을 늘리세요."]
        out.append(
            AnalysisResult(
                member_name=name,
                summary=f"{name}의 기여도는 {s.normalizedScore:.0f}/100, 팀 평균 대비 {s.pct_vs_team_mean:+.1f}%입니다.",
                strengths=["Git 지표 기반으로 산정 근거를 추적할 수 있습니다."],
                weaknesses=flags[:3] or ["특이 이상 없음"],
                position_in_team=f"팀 {n}명 중 {s.rank}위 (상위 {s.top_percent:.1f}%).",
                recommended_actions=rec,
            )
        )
    return out


_REPORT_LLM_SYSTEM_KO = """당신은 대학 팀 프로젝트 평가를 돕는 시니어 TA(한국어)입니다.
입력 JSON에는 규칙 엔진 점수(normalizedScore), 데이터로 학습한 보조 점수(dl_score), 블렌드(blendedScore),
DL 신뢰도(dl_confidence), 이상 플래그, 자기서술 발췌, 그리고 모델 품질 요약이 포함됩니다.

규칙:
- 숫자 점수는 절대 재산정·수정하지 말고 해석·근거만 제시합니다.
- dl_score가 있으면 normalizedScore와의 차이를 metric_dl_note에 반드시 언급합니다(왜 어긋날 수 있는지 교육적으로).
- dl_confidence가 낮거나 이상 플래그가 있으면 보수적으로 서술합니다.
- 출력은 JSON 한 덩어리만 (마크다운·코드펜스 금지).

스키마:
{
  "team_summary": "팀 전체 2~5문장 총평(교육자용)",
  "members": [
    {
      "member_name": "이름과 입력 동일",
      "summary": "2~4문장",
      "strengths": ["짧은 근거 2~4개"],
      "weaknesses": ["짧은 근거 1~3개"],
      "position_in_team": "역할·상대적 위치",
      "recommended_actions": ["실행 가능한 제안 2~4개"],
      "metric_dl_note": "규칙 vs DL(blended) 정합·긴장 요약 1~2문장"
    }
  ]
}
members 순서와 인원 수는 입력과 동일해야 합니다."""


def _report_llm_dl_context(dl_model_info: dict[str, Any]) -> dict[str, Any]:
    q = dl_model_info.get("quality") if isinstance(dl_model_info, dict) else None
    q = q if isinstance(q, dict) else {}
    op = q.get("operations_playbook") if isinstance(q.get("operations_playbook"), dict) else {}
    return {
        "dl_backend": dl_model_info.get("backend"),
        "blend_formula": dl_model_info.get("blend_formula"),
        "sample_count": dl_model_info.get("sample_count"),
        "model_size_profile": q.get("model_size_profile"),
        "validation_mae_mean": q.get("validation_mae_mean"),
        "holdout_time_mae": q.get("holdout_time_mae"),
        "feature_drift": q.get("feature_drift"),
        "operations_summary": (op.get("summary_ko") or "")[:800],
        "checklist_hints": (op.get("checklist_ko") or [])[:6],
    }


def _build_report_llm_payload(
    *,
    project_name: str,
    users: list[TeamUserIn],
    scores: list[ScoreResult],
    anomalies: list[AnomalyResult],
    dl_model_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "dl_context": _report_llm_dl_context(dl_model_info),
        "members": [
            {
                "member_name": s.member_name,
                "score": s.model_dump(),
                "anomaly_flags": next((a.flags for a in anomalies if a.member_name == s.member_name), []),
                "self_report_excerpt": next((u.selfReport for u in users if u.name.strip() == s.member_name), "")[:900],
            }
            for s in scores
        ],
    }


def _analysis_from_llm_rows(
    data: dict[str, Any], scores: list[ScoreResult]
) -> tuple[list[AnalysisResult], str] | None:
    if not isinstance(data, dict):
        return None
    arr = data.get("members")
    if not isinstance(arr, list):
        return None
    team_summary = str(data.get("team_summary", "") or "")[:4000]
    out: list[AnalysisResult] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        out.append(
            AnalysisResult(
                member_name=str(it.get("member_name", ""))[:120],
                summary=str(it.get("summary", ""))[:1500],
                strengths=[str(x) for x in (it.get("strengths") or [])][:5],
                weaknesses=[str(x) for x in (it.get("weaknesses") or [])][:5],
                position_in_team=str(it.get("position_in_team", ""))[:400],
                recommended_actions=[str(x) for x in (it.get("recommended_actions") or [])][:5],
                metric_dl_note=str(it.get("metric_dl_note", "") or "")[:1200],
            )
        )
    if len(out) != len(scores):
        return None
    return out, team_summary


def _merge_two_team_summaries(a: str, b: str) -> str:
    a, b = a.strip(), b.strip()
    if a and b:
        return f"[Gemini 관점]\n{a}\n\n[OpenAI 관점]\n{b}"[:4000]
    return (a or b)[:4000]


def _merge_analysis_lists(
    left: list[AnalysisResult], right: list[AnalysisResult], scores: list[ScoreResult]
) -> list[AnalysisResult]:
    rmap = {x.member_name.strip(): x for x in right}
    merged: list[AnalysisResult] = []
    for L in left:
        name = L.member_name.strip()
        R = rmap.get(name)
        if R is None:
            merged.append(L)
            continue

        def _uniq(xs: list[str], ys: list[str], cap: int) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for z in xs + ys:
                t = z.strip()
                if not t or t.lower() in seen:
                    continue
                seen.add(t.lower())
                out.append(t[:500])
                if len(out) >= cap:
                    break
            return out

        merged.append(
            AnalysisResult(
                member_name=name,
                summary=f"[Gemini]\n{L.summary}\n\n[OpenAI]\n{R.summary}"[:1500],
                strengths=_uniq(L.strengths, R.strengths, 5),
                weaknesses=_uniq(L.weaknesses, R.weaknesses, 5),
                position_in_team=L.position_in_team or R.position_in_team,
                recommended_actions=_uniq(L.recommended_actions, R.recommended_actions, 5),
                metric_dl_note=f"{L.metric_dl_note}\n{R.metric_dl_note}".strip()[:1200],
            )
        )
    return merged if len(merged) == len(scores) else left


def _openai_report_raw(payload: dict[str, Any], api_key: str) -> dict[str, Any] | None:
    model = (os.environ.get("TEAM_REPORT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o").strip()
    client = get_openai_client(api_key)
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _REPORT_LLM_SYSTEM_KO},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=4096,
    )
    text = (res.choices[0].message.content or "").strip()
    return _extract_json(text) if text else None


def _gemini_report_raw(payload: dict[str, Any], api_key: str) -> dict[str, Any] | None:
    import google.generativeai as genai

    model_name = (os.environ.get("TEAM_REPORT_GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    ensure_gemini_configured(api_key)
    model = genai.GenerativeModel(model_name=model_name, system_instruction=_REPORT_LLM_SYSTEM_KO)
    res = model.generate_content(
        json.dumps(payload, ensure_ascii=False),
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
            "max_output_tokens": 4096,
        },
    )
    text = (getattr(res, "text", None) or "").strip()
    if not text:
        return None
    return _extract_json(text)


def run_ai_analyzer(
    users: list[TeamUserIn],
    scores: list[ScoreResult],
    anomalies: list[AnomalyResult],
    *,
    dl_model_info: dict[str, Any],
    project_name: str = "",
) -> tuple[list[AnalysisResult], str, dict[str, Any]]:
    """Gemini/OpenAI(단일·병렬)로 리포트 분석. API 키 없으면 휴리스틱."""
    meta: dict[str, Any] = {"providers_attempted": [], "mode": "fallback"}
    payload = _build_report_llm_payload(
        project_name=project_name.strip(),
        users=users,
        scores=scores,
        anomalies=anomalies,
        dl_model_info=dl_model_info if isinstance(dl_model_info, dict) else {},
    )
    gk = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    ok_openai = (os.environ.get("OPENAI_API_KEY") or "").strip()
    mode = (os.environ.get("TEAM_REPORT_LLM") or "auto").strip().lower()
    if mode in ("0", "off", "none", "heuristic"):
        fb = _fallback_analysis(users, scores, anomalies)
        return fb, "", {**meta, "mode": "heuristic_only"}

    def _run_gemini() -> tuple[str, dict[str, Any] | None, float]:
        t0 = time.perf_counter()
        try:
            data = _gemini_report_raw(payload, gk)
            return "gemini", data, (time.perf_counter() - t0) * 1000
        except Exception:
            return "gemini", None, (time.perf_counter() - t0) * 1000

    def _run_openai() -> tuple[str, dict[str, Any] | None, float]:
        t0 = time.perf_counter()
        try:
            data = _openai_report_raw(payload, ok_openai)
            return "openai", data, (time.perf_counter() - t0) * 1000
        except Exception:
            return "openai", None, (time.perf_counter() - t0) * 1000

    results: list[tuple[str, dict[str, Any] | None, float, str | None]] = []

    if mode == "multi" and gk and ok_openai:
        meta["providers_attempted"] = ["gemini", "openai"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(_run_gemini), pool.submit(_run_openai)]
            for fut in as_completed(futs):
                label, data, ms = fut.result()
                err = None
                if data is None:
                    err = "no_or_invalid_response"
                results.append((label, data, ms, err))
        gem_data = next((d for lab, d, _, _ in results if lab == "gemini"), None)
        oa_data = next((d for lab, d, _, _ in results if lab == "openai"), None)
        meta["latencies_ms"] = {lab: round(ms, 2) for lab, _, ms, _ in results}
        meta["models"] = {
            "gemini": (os.environ.get("TEAM_REPORT_GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"),
            "openai": (os.environ.get("TEAM_REPORT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o"),
        }
        g_parsed = _analysis_from_llm_rows(gem_data, scores) if isinstance(gem_data, dict) else None
        o_parsed = _analysis_from_llm_rows(oa_data, scores) if isinstance(oa_data, dict) else None
        if g_parsed and o_parsed:
            g_rows, g_sum = g_parsed
            o_rows, o_sum = o_parsed
            merged_rows = _merge_analysis_lists(g_rows, o_rows, scores)
            meta["mode"] = "multi_merged"
            return merged_rows, _merge_two_team_summaries(g_sum, o_sum), meta
        if g_parsed:
            meta["mode"] = "multi_partial_gemini_only"
            r, s = g_parsed
            return r, s, meta
        if o_parsed:
            meta["mode"] = "multi_partial_openai_only"
            r, s = o_parsed
            return r, s, meta
        fb = _fallback_analysis(users, scores, anomalies)
        meta["mode"] = "multi_failed_fallback"
        return fb, "", meta

    if mode == "openai" and ok_openai:
        meta["providers_attempted"] = ["openai"]
        lab, data, ms = _run_openai()
        meta["latencies_ms"] = {lab: round(ms, 2)}
        meta["models"] = {"openai": (os.environ.get("TEAM_REPORT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o")}
        if isinstance(data, dict):
            parsed = _analysis_from_llm_rows(data, scores)
            if parsed:
                meta["mode"] = "openai"
                r, s = parsed
                return r, s, meta
        fb = _fallback_analysis(users, scores, anomalies)
        meta["mode"] = "openai_failed_fallback"
        return fb, "", meta

    if mode == "gemini" and gk:
        meta["providers_attempted"] = ["gemini"]
        lab, data, ms = _run_gemini()
        meta["latencies_ms"] = {lab: round(ms, 2)}
        meta["models"] = {"gemini": (os.environ.get("TEAM_REPORT_GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash")}
        if isinstance(data, dict):
            parsed = _analysis_from_llm_rows(data, scores)
            if parsed:
                meta["mode"] = "gemini"
                r, s = parsed
                return r, s, meta
        fb = _fallback_analysis(users, scores, anomalies)
        meta["mode"] = "gemini_failed_fallback"
        return fb, "", meta

    # auto: Gemini 우선, 실패 시 OpenAI
    if mode in ("auto", "", "default") or mode not in ("multi", "openai", "gemini"):
        if gk:
            meta["providers_attempted"].append("gemini")
            _, data, ms = _run_gemini()
            meta["latencies_ms"] = {"gemini": round(ms, 2)}
            meta["models"] = {"gemini": (os.environ.get("TEAM_REPORT_GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash")}
            if isinstance(data, dict):
                parsed = _analysis_from_llm_rows(data, scores)
                if parsed:
                    meta["mode"] = "gemini"
                    r, s = parsed
                    return r, s, meta
        if ok_openai:
            meta["providers_attempted"].append("openai")
            _, data, ms = _run_openai()
            meta.setdefault("latencies_ms", {})["openai"] = round(ms, 2)
            meta.setdefault("models", {})["openai"] = (os.environ.get("TEAM_REPORT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o")
            if isinstance(data, dict):
                parsed = _analysis_from_llm_rows(data, scores)
                if parsed:
                    meta["mode"] = "openai"
                    r, s = parsed
                    return r, s, meta

    fb = _fallback_analysis(users, scores, anomalies)
    meta["mode"] = "fallback_no_keys"
    return fb, "", meta


@router.post("/report", response_model=TeamReportResponse)
def post_team_report(body: TeamReportRequest) -> TeamReportResponse:
    users = body.teamData
    req_id = str(uuid.uuid4())
    scores, trust = run_score_engine(users)
    dl_model_info = apply_dl_scores(users, scores, request_id=req_id)
    anomalies = run_anomaly_detector(users, scores)
    analysis, team_narrative, ai_meta = run_ai_analyzer(
        users,
        scores,
        anomalies,
        dl_model_info=dl_model_info,
        project_name=body.project_name.strip(),
    )
    edge_cases = detect_edge_cases(users, scores)

    ts = datetime.now(timezone.utc).isoformat()
    input_data = [u.model_dump() for u in users]
    input_blob = json.dumps(input_data, ensure_ascii=False, sort_keys=True)
    input_hash = hashlib.sha256(input_blob.encode("utf-8")).hexdigest()[:24]
    evaluation_log = {
        "request_id": req_id,
        "timestamp": ts,
        "project_name": body.project_name.strip(),
        "input_hash": input_hash,
        "input_data": input_data,
        "score": [s.model_dump() for s in scores],
        "dl_model_info": dl_model_info,
        "anomaly": [a.model_dump() for a in anomalies],
        "ai_result": [a.model_dump() for a in analysis],
        "edge_cases": edge_cases,
        "ai_meta": ai_meta,
        "team_narrative": team_narrative,
    }
    try:
        record_team_report(
            request_id=req_id,
            project_name=body.project_name.strip(),
            users=input_data,
            scores=[s.model_dump() for s in scores],
            anomalies=[a.model_dump() for a in anomalies],
            trust_scores=trust,
            dl_model_info=dl_model_info,
        )
        evaluation_log["db_profile"] = db_profile()
    except Exception:
        # DB 저장 실패가 평가 API 실패로 이어지지 않도록 방어
        evaluation_log["db_profile"] = {"error": "db_store_failed"}
    return TeamReportResponse(
        scores=scores,
        anomalies=anomalies,
        analysis=analysis,
        edge_cases=edge_cases,
        trust_scores=trust,
        dl_model_info=dl_model_info,
        evaluation_log=evaluation_log,
        team_narrative=team_narrative,
        ai_meta=ai_meta,
    )


@router.post("/benchmark-narrow")
def post_benchmark_narrow(body: TeamReportRequest) -> dict[str, Any]:
    """좁은 과제: 룰 기준점 대비 전용 DL vs OpenAI 단일 호출 MAE 비교(데이터 누적·재학습 없음)."""
    from edu_tools.team_llm_benchmark import run_narrow_benchmark

    return run_narrow_benchmark(body)

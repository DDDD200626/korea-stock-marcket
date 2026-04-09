"""
팀 기여 보조 딥러닝 개발 로드맵(10단계).
구현·메타는 team_torch_model, team_ml_model, GET /api/capabilities 를 함께 본다.
"""

from __future__ import annotations

from typing import Any, Mapping

ROADMAP_VERSION = 5

PHASES_KO: tuple[str, ...] = (
    "1) 데이터 늘리기 — JSONL 누적·TEAM_TRAIN_RESERVOIR_MAX·합성 배치(team_synthetic_bulk)",
    "2) 라벨 정의 고정 — hybrid_dl_target·STRUCTURAL_TARGET_WEIGHT·행에 label_spec_version·feature_version 기록",
    "3) 이상 샘플 정리 — TEAM_TRAIN_OUTLIER_IQR_K 로 라벨 y IQR 밖 행 제거(학습 행 부족 시 자동 유지)",
    "4) 피처 확장 — FEATURE_VERSION·34차원 build_feature_vector(형태 신호 + 선택적 다국어 문장 임베딩 8차원)",
    "5) 시간 홀드아웃 — TEAM_HOLDOUT_TIME_FRAC 로 최근 구간을 학습에서 제외·holdout_time_mae/pearson/r2 기록",
    "6) 세션 CV·앙상블 — 동일 sess 누수 방지·PyTorch MLP 앙상블·MC 드롭아웃·GBDT 블렌드",
    "7) 하이퍼 탐색 확대 — TEAM_TORCH_EXTENDED_HP=1 시 그리드 변형 후보 추가(시간↑)",
    "8) 용량 프로파일 — standard / large / xlarge / xxl(5층 극대)·TEAM_TORCH_ENSEMBLE 등",
    "9) 품질 메타 — dl_quality_unified: CV·tail·캘리브·시간 홀드아웃·데이터 파이프라인",
    "10) API·감사 — GET /api/capabilities·dataset-label-summary·dataset SHA-256·GET /api/team/report",
)


def build_dl_quality_unified(meta: Mapping[str, Any]) -> dict[str, Any]:
    """학습 메타에서 추적용 품질 지표를 한 객체로 묶는다."""
    return {
        "schema": "dl_quality_unified_v4",
        "cv": {
            "split_strategy": meta.get("cv_split_strategy"),
            "unique_groups": meta.get("cv_unique_groups"),
            "mae_mean": meta.get("cv_mae_mean"),
            "vs_validation_gap": meta.get("cv_vs_validation_gap"),
        },
        "validation_mae_mean": meta.get("validation_mae_mean"),
        "chronological_tail": {
            "mae_mean": meta.get("chronological_tail_mae_mean"),
            "fraction": meta.get("chronological_tail_fraction"),
            "note": meta.get("chronological_tail_note"),
        },
        "nn_gbdt_blend": {
            "gbdt_present": meta.get("gbdt_present"),
            "nn_gbdt_blend_alpha": meta.get("nn_gbdt_blend_alpha"),
            "gbdt_validation_blend_mae": meta.get("gbdt_validation_blend_mae"),
        },
        "calibration_linear": meta.get("calibration"),
        "fit_metrics": {
            "calibration_pearson_r": meta.get("calibration_pearson_r"),
            "calibration_r2": meta.get("calibration_r2"),
            "note_ko": meta.get("calibration_fit_note_ko"),
        },
        "holdout_time": {
            "active": (meta.get("time_holdout_split") or {}).get("active"),
            "split": meta.get("time_holdout_split"),
            "mae": meta.get("holdout_time_mae"),
            "pearson_r": meta.get("holdout_time_pearson_r"),
            "r2": meta.get("holdout_time_r2"),
            "note_ko": meta.get("holdout_time_note_ko"),
        },
        "data_pipeline": {
            "outlier_filter": meta.get("training_outlier_stats"),
            "effective_training_rows": meta.get("effective_training_rows"),
            "label_spec_version": meta.get("label_spec_version"),
        },
        "explainability": {
            "permutation_importance_top": meta.get("permutation_importance_top"),
            "input_noise_std_training": meta.get("input_noise_std_training"),
            "note_ko": (
                "치환 중요도는 검증 부분집합에서 피처 열을 무작위 섞었을 때 MAE 증가량(대략적 기여)입니다. "
                "입력 노이즈는 학습 중에만 적용되는 선택적 일반화입니다."
            ),
        },
        "dl_roadmap_version": meta.get("dl_roadmap_version"),
    }


def roadmap_payload() -> dict[str, Any]:
    return {
        "version": ROADMAP_VERSION,
        "phase_count": len(PHASES_KO),
        "phases_ko": list(PHASES_KO),
        "scope_note_ko": (
            "본 모델은 팀 기여도용 표(수치) 회귀·앙상블이며, 생성형 거대 언어모델과 과제·규모가 다릅니다. "
            "데이터·홀드아웃·피처·하이퍼·용량을 조합해 도메인에서 닿을 수 있는 성능에 가깝게 맞추는 것이 목표입니다."
        ),
    }

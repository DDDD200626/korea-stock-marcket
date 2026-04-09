"""PyTorch-based contribution model (ensemble + uncertainty)."""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edu_tools.dl_roadmap import ROADMAP_VERSION, build_dl_quality_unified
from edu_tools.team_lm_embedding import semantic_encoder_meta
from edu_tools.team_ml_model import (
    DATASET_PATH,
    FEATURE_DIM,
    FEATURE_LABELS,
    FEATURE_VERSION,
    LABEL_SPEC_VERSION,
    MIN_TRAIN_SAMPLES,
    RETRAIN_INTERVAL,
    build_feature_vector,
    dataset_file_sha256,
    filter_training_rows_outliers,
    last_training_resolve_meta,
    pad_feature_vector,
    retrain_forced_by_env,
    resolve_training_rows,
    sort_rows_by_created_at,
    time_holdout_split,
)

DATA_DIR = Path(__file__).with_name("data")
TORCH_MODEL_PATH = DATA_DIR / "team_torch_model.pt"
TORCH_META_PATH = DATA_DIR / "team_torch_model.meta.json"

# 기본: 경량. TEAM_TORCH_MODEL_SIZE=large → 깊은 대형 MLP + 앙상블 확대 (로컬/CPU 탑재용)
STANDARD_ENSEMBLE_SIZE = 4
STANDARD_CV_FOLDS = 5
STANDARD_MC_DROPOUT = 14
HYPERPARAM_CANDIDATES_STANDARD: list[dict[str, Any]] = [
    {"hidden1": 40, "hidden2": 20, "hidden3": 0, "dropout": 0.08, "lr": 0.008, "weight_decay": 1e-4, "epochs": 340},
    {"hidden1": 72, "hidden2": 28, "hidden3": 0, "dropout": 0.12, "lr": 0.006, "weight_decay": 5e-5, "epochs": 440},
    {"hidden1": 56, "hidden2": 28, "hidden3": 0, "dropout": 0.10, "lr": 0.005, "weight_decay": 1e-4, "epochs": 400},
    # 소표본·노이즈에 강한 쪽 (CV에서 자주 선택되도록 추가 후보)
    {"hidden1": 64, "hidden2": 26, "hidden3": 0, "dropout": 0.15, "lr": 0.0045, "weight_decay": 2e-4, "epochs": 460},
]
# 대형: 4층 MLP (~수만~십만 단위 파라미터, 입력 차원에 비례)
HYPERPARAM_CANDIDATES_LARGE: list[dict[str, Any]] = [
    {
        "hidden1": 320,
        "hidden2": 160,
        "hidden3": 80,
        "dropout": 0.14,
        "lr": 0.0038,
        "weight_decay": 1e-4,
        "epochs": 540,
    },
    {
        "hidden1": 256,
        "hidden2": 128,
        "hidden3": 64,
        "dropout": 0.13,
        "lr": 0.004,
        "weight_decay": 9e-5,
        "epochs": 520,
    },
    {
        "hidden1": 384,
        "hidden2": 192,
        "hidden3": 96,
        "dropout": 0.15,
        "lr": 0.0032,
        "weight_decay": 1.15e-4,
        "epochs": 580,
    },
    {
        "hidden1": 288,
        "hidden2": 144,
        "hidden3": 72,
        "dropout": 0.16,
        "lr": 0.003,
        "weight_decay": 1e-4,
        "epochs": 560,
    },
]
ARCH_CANDIDATES: list[dict[str, Any]] = [
    {"name": "relu", "activation": "relu"},
    {"name": "gelu", "activation": "gelu"},
    {"name": "silu", "activation": "silu"},
]
ARCH_CANDIDATES_LARGE: list[dict[str, Any]] = [
    {"name": "gelu", "activation": "gelu"},
    {"name": "silu", "activation": "silu"},
]
# 초대형: 5층 MLP (hidden4 포함) — CPU에서 느릴 수 있음. TEAM_TORCH_MODEL_SIZE=xlarge
HYPERPARAM_CANDIDATES_XL: list[dict[str, Any]] = [
    {
        "hidden1": 640,
        "hidden2": 320,
        "hidden3": 160,
        "hidden4": 80,
        "dropout": 0.16,
        "lr": 0.0022,
        "weight_decay": 1.12e-4,
        "epochs": 700,
    },
    {
        "hidden1": 576,
        "hidden2": 288,
        "hidden3": 144,
        "hidden4": 72,
        "dropout": 0.16,
        "lr": 0.0022,
        "weight_decay": 1.2e-4,
        "epochs": 680,
    },
    {
        "hidden1": 512,
        "hidden2": 256,
        "hidden3": 128,
        "hidden4": 64,
        "dropout": 0.15,
        "lr": 0.0025,
        "weight_decay": 1.05e-4,
        "epochs": 660,
    },
    {
        "hidden1": 704,
        "hidden2": 352,
        "hidden3": 176,
        "hidden4": 88,
        "dropout": 0.17,
        "lr": 0.002,
        "weight_decay": 1.25e-4,
        "epochs": 720,
    },
    {
        "hidden1": 768,
        "hidden2": 384,
        "hidden3": 192,
        "hidden4": 96,
        "dropout": 0.17,
        "lr": 0.0019,
        "weight_decay": 1.28e-4,
        "epochs": 740,
    },
]
# 초대형·CPU 경량: 동일 5층 구조, 폭·앙상블·MC·에폭 축소. TEAM_TORCH_MODEL_SIZE=xlarge_lite
HYPERPARAM_CANDIDATES_XL_LITE: list[dict[str, Any]] = [
    {
        "hidden1": 384,
        "hidden2": 192,
        "hidden3": 96,
        "hidden4": 48,
        "dropout": 0.14,
        "lr": 0.0028,
        "weight_decay": 1e-4,
        "epochs": 520,
    },
    {
        "hidden1": 320,
        "hidden2": 160,
        "hidden3": 80,
        "hidden4": 40,
        "dropout": 0.13,
        "lr": 0.003,
        "weight_decay": 1e-4,
        "epochs": 500,
    },
    {
        "hidden1": 448,
        "hidden2": 224,
        "hidden3": 112,
        "hidden4": 56,
        "dropout": 0.15,
        "lr": 0.0026,
        "weight_decay": 1.05e-4,
        "epochs": 540,
    },
]
# 극대형: 5층 MLP 폭·에폭 상한 — GPU·고사양 권장. TEAM_TORCH_MODEL_SIZE=xxl
HYPERPARAM_CANDIDATES_XXL: list[dict[str, Any]] = [
    {
        "hidden1": 896,
        "hidden2": 448,
        "hidden3": 224,
        "hidden4": 112,
        "dropout": 0.17,
        "lr": 0.00175,
        "weight_decay": 1.18e-4,
        "epochs": 780,
    },
    {
        "hidden1": 1024,
        "hidden2": 512,
        "hidden3": 256,
        "hidden4": 128,
        "dropout": 0.18,
        "lr": 0.00155,
        "weight_decay": 1.22e-4,
        "epochs": 820,
    },
    {
        "hidden1": 768,
        "hidden2": 384,
        "hidden3": 192,
        "hidden4": 96,
        "dropout": 0.16,
        "lr": 0.0019,
        "weight_decay": 1.1e-4,
        "epochs": 760,
    },
    {
        "hidden1": 1152,
        "hidden2": 576,
        "hidden3": 288,
        "hidden4": 144,
        "dropout": 0.19,
        "lr": 0.00145,
        "weight_decay": 1.28e-4,
        "epochs": 840,
    },
]

_TORCH_PREDICT_CACHE: tuple[str, float, Any] | None = None


def invalidate_torch_predict_cache() -> None:
    global _TORCH_PREDICT_CACHE
    _TORCH_PREDICT_CACHE = None


def _early_stop_patience(n_samples: int) -> int:
    """표본 수에 맞춘 조기 종료 인내 — 너무 짧으면 과소적합, 너무 길면 CV 비용만 증가."""
    return max(28, min(52, 18 + n_samples // 4))


def _resolve_torch_seed() -> int:
    """학습 전 구간(CV 셔플/부트스트랩 포함)에 공통 적용할 시드."""
    try:
        return int((os.environ.get("TEAM_TORCH_SEED", "42") or "42").strip())
    except Exception:
        return 42


def _row_group_ids(sess_ids: list[str]) -> list[int]:
    """세션 문자열이 같으면 동일 그룹. 비어 있으면 행마다 별도 그룹(누수 최소화)."""
    key_to_gid: dict[str, int] = {}
    gids: list[int] = []
    next_id = 0
    for i, raw in enumerate(sess_ids):
        s = (raw or "").strip()
        key = s if s else f"__row_{i}"
        if key not in key_to_gid:
            key_to_gid[key] = next_id
            next_id += 1
        gids.append(key_to_gid[key])
    return gids


def _split_groups_into_k(groups: list[int], k: int) -> list[list[int]]:
    """고유 그룹 id 목록을 k개 파트로 균등 분할."""
    ug = sorted(set(groups))
    if k <= 0 or not ug:
        return []
    n_g = len(ug)
    k = min(k, n_g)
    base, rem = divmod(n_g, k)
    parts: list[list[int]] = []
    pos = 0
    for i in range(k):
        sz = base + (1 if i < rem else 0)
        parts.append(ug[pos : pos + sz])
        pos += sz
    return parts


def _cv_fold_train_val_indices(
    n: int,
    sess_ids: list[str],
    fold_count: int,
    rng: random.Random,
) -> tuple[list[tuple[list[int], list[int]]], str]:
    """(tr_idx, va_idx) 목록(원래 행 인덱스 0..n-1), 전략 이름."""
    fold_count = max(2, min(fold_count, n))
    use_sess = (os.environ.get("TEAM_TORCH_SESSION_CV", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    gids = _row_group_ids(sess_ids)
    ug = sorted(set(gids))

    if use_sess and len(ug) >= fold_count:
        parts = _split_groups_into_k(gids, fold_count)
        if len(parts) == fold_count:
            folds: list[tuple[list[int], list[int]]] = []
            for part in parts:
                val_g = set(part)
                va_idx = [i for i in range(n) if gids[i] in val_g]
                tr_idx = [i for i in range(n) if gids[i] not in val_g]
                if va_idx and tr_idx:
                    folds.append((tr_idx, va_idx))
            if len(folds) == fold_count:
                return folds, "session_group"

    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = max(1, n // fold_count)
    folds_fb: list[tuple[list[int], list[int]]] = []
    for f in range(fold_count):
        va_start = f * fold_size
        va_end = (f + 1) * fold_size if f < fold_count - 1 else n
        va_idx = idx[va_start:va_end]
        tr_idx = idx[:va_start] + idx[va_end:]
        folds_fb.append((tr_idx, va_idx))
    return folds_fb, "index_shuffle"


def _rolling_time_fold_indices(n: int, fold_count: int) -> list[tuple[list[int], list[int]]]:
    """시간 순서 보존 롤링 검증 (과거→미래)."""
    if n < 8:
        return []
    fold_count = max(2, min(fold_count, 6))
    step = max(2, n // (fold_count + 1))
    min_train = max(6, n // 5)
    out: list[tuple[list[int], list[int]]] = []
    cut = min_train
    for _ in range(fold_count):
        va_start = cut
        va_end = min(n, va_start + step)
        if va_end - va_start < 2 or va_start < min_train:
            break
        tr_idx = list(range(0, va_start))
        va_idx = list(range(va_start, va_end))
        if tr_idx and va_idx:
            out.append((tr_idx, va_idx))
        cut += step
        if cut >= n - 2:
            break
    return out


def _model_size_profile() -> str:
    raw = (os.environ.get("TEAM_TORCH_MODEL_SIZE") or "standard").strip().lower()
    if raw in ("xxl", "2xl", "giant", "max", "ultra"):
        return "xxl"
    if raw in ("xlarge_lite", "xlarge-lite", "xlarge_fast", "xlarge-cpu", "xlarge_cpu", "xlcpu"):
        return "xlarge_lite"
    if raw == "xlarge":
        return "xlarge"
    if raw in ("large", "xl", "huge", "big"):
        return "large"
    return "standard"


def _maybe_extend_hparam_grid(grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TEAM_TORCH_EXTENDED_HP=1 이면 그리드에 변형 후보를 추가해 탐색 시간·범위 확대."""
    if (os.environ.get("TEAM_TORCH_EXTENDED_HP", "0") or "0").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return grid
    extra: list[dict[str, Any]] = []
    for h in grid[: min(4, len(grid))]:
        extra.append(
            {
                **h,
                "lr": float(h["lr"]) * 0.88,
                "epochs": int(int(h["epochs"]) * 1.1),
            }
        )
        extra.append(
            {
                **h,
                "lr": float(h["lr"]) * 1.05,
                "weight_decay": float(h.get("weight_decay", 1e-4)) * 0.72,
            }
        )
    return grid + extra


def _training_profile() -> dict[str, Any]:
    """standard | large | xlarge | xlarge_lite | xxl — 학습·추론 하이퍼파라미터 묶음."""
    prof = _model_size_profile()
    if prof == "xxl":
        return {
            "name": "xxl",
            "ensemble_size": int(os.environ.get("TEAM_TORCH_ENSEMBLE", "11") or 11),
            "cv_folds": min(5, max(3, int(os.environ.get("TEAM_TORCH_CV_FOLDS", "5") or 5))),
            "mc_dropout": int(os.environ.get("TEAM_TORCH_MC_SAMPLES", "32") or 32),
            "hparam_grid": _maybe_extend_hparam_grid(HYPERPARAM_CANDIDATES_XXL),
            "arch_grid": ARCH_CANDIDATES_LARGE,
        }
    if prof == "xlarge_lite":
        return {
            "name": "xlarge_lite",
            "ensemble_size": int(os.environ.get("TEAM_TORCH_ENSEMBLE", "6") or 6),
            "cv_folds": min(4, max(2, int(os.environ.get("TEAM_TORCH_CV_FOLDS", "4") or 4))),
            "mc_dropout": int(os.environ.get("TEAM_TORCH_MC_SAMPLES", "14") or 14),
            "hparam_grid": _maybe_extend_hparam_grid(HYPERPARAM_CANDIDATES_XL_LITE),
            "arch_grid": ARCH_CANDIDATES_LARGE,
        }
    if prof == "xlarge":
        return {
            "name": "xlarge",
            "ensemble_size": int(os.environ.get("TEAM_TORCH_ENSEMBLE", "10") or 10),
            "cv_folds": min(5, max(2, int(os.environ.get("TEAM_TORCH_CV_FOLDS", "5") or 5))),
            "mc_dropout": int(os.environ.get("TEAM_TORCH_MC_SAMPLES", "28") or 28),
            "hparam_grid": _maybe_extend_hparam_grid(HYPERPARAM_CANDIDATES_XL),
            "arch_grid": ARCH_CANDIDATES_LARGE,
        }
    if prof == "large":
        return {
            "name": "large",
            "ensemble_size": int(os.environ.get("TEAM_TORCH_ENSEMBLE", "8") or 8),
            "cv_folds": min(5, max(2, int(os.environ.get("TEAM_TORCH_CV_FOLDS", "5") or 5))),
            "mc_dropout": int(os.environ.get("TEAM_TORCH_MC_SAMPLES", "20") or 20),
            "hparam_grid": _maybe_extend_hparam_grid(HYPERPARAM_CANDIDATES_LARGE),
            "arch_grid": ARCH_CANDIDATES_LARGE,
        }
    return {
        "name": "standard",
        "ensemble_size": STANDARD_ENSEMBLE_SIZE,
        "cv_folds": STANDARD_CV_FOLDS,
        "mc_dropout": STANDARD_MC_DROPOUT,
        "hparam_grid": _maybe_extend_hparam_grid(HYPERPARAM_CANDIDATES_STANDARD),
        "arch_grid": ARCH_CANDIDATES,
    }


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _save_meta(meta: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_meta() -> dict[str, Any] | None:
    if not TORCH_META_PATH.exists():
        return None
    try:
        raw = json.loads(TORCH_META_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _h_use_residual_mlp(h: dict[str, Any]) -> bool:
    v = h.get("use_residual_mlp")
    if v is True or v == 1:
        return True
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _build_residual_tabular_mlp(nn: Any, h: dict[str, Any], in_dim: int) -> Any:
    """3층 직렬 MLP 대체: 잔차 블록 + LayerNorm(깊이·표현력 상한 확장, 표 형태 입력용)."""
    act = str(h.get("activation", "relu")).lower()
    if act == "gelu":
        act_layer = nn.GELU
    elif act == "silu":
        act_layer = nn.SiLU
    else:
        act_layer = nn.ReLU
    h1 = int(h["hidden1"])
    h2 = int(h["hidden2"])
    drop = float(h["dropout"])

    class _ResTab(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin1 = nn.Linear(int(in_dim), h1)
            self.ln = nn.LayerNorm(h1)
            self.act = act_layer()
            self.drop = nn.Dropout(p=drop)
            self.lin2 = nn.Linear(h1, int(in_dim))
            self.head1 = nn.Linear(int(in_dim), h2)
            self.act2 = act_layer()
            self.head2 = nn.Linear(h2, 1)

        def forward(self, x: Any) -> Any:
            r = self.lin1(x)
            r = self.ln(r)
            r = self.act(r)
            r = self.drop(r)
            r = self.lin2(r)
            z = x + r
            z = self.act2(self.head1(z))
            return self.head2(z)

    return _ResTab()


def _build_model(nn: Any, h: dict[str, Any], in_dim: int) -> Any:
    act = str(h.get("activation", "relu")).lower()
    if act == "gelu":
        act_layer = nn.GELU
    elif act == "silu":
        act_layer = nn.SiLU
    else:
        act_layer = nn.ReLU
    h3 = int(h.get("hidden3") or 0)
    h4 = int(h.get("hidden4") or 0)
    drop = float(h["dropout"])
    if _h_use_residual_mlp(h) and h3 == 0 and h4 == 0:
        return _build_residual_tabular_mlp(nn, h, in_dim)
    if h3 > 0 and h4 > 0:
        drop2 = min(0.35, drop * 0.88)
        drop3 = min(0.32, drop * 0.76)
        return nn.Sequential(
            nn.Linear(int(in_dim), int(h["hidden1"])),
            act_layer(),
            nn.Dropout(p=drop),
            nn.Linear(int(h["hidden1"]), int(h["hidden2"])),
            act_layer(),
            nn.Dropout(p=drop2),
            nn.Linear(int(h["hidden2"]), h3),
            act_layer(),
            nn.Dropout(p=drop3),
            nn.Linear(h3, h4),
            act_layer(),
            nn.Linear(h4, 1),
        )
    if h3 > 0:
        drop2 = min(0.35, drop * 0.88)
        return nn.Sequential(
            nn.Linear(int(in_dim), int(h["hidden1"])),
            act_layer(),
            nn.Dropout(p=drop),
            nn.Linear(int(h["hidden1"]), int(h["hidden2"])),
            act_layer(),
            nn.Dropout(p=drop2),
            nn.Linear(int(h["hidden2"]), h3),
            act_layer(),
            nn.Linear(h3, 1),
        )
    return nn.Sequential(
        nn.Linear(int(in_dim), int(h["hidden1"])),
        act_layer(),
        nn.Dropout(p=drop),
        nn.Linear(int(h["hidden1"]), int(h["hidden2"])),
        act_layer(),
        nn.Linear(int(h["hidden2"]), 1),
    )


def _count_params(model: Any) -> int:
    try:
        return sum(int(p.numel()) for p in model.parameters())
    except Exception:
        return 0


def _gbdt_feature_importance_top(hgb: Any, feat_labels: list[str], k: int = 5) -> list[dict[str, Any]]:
    try:
        imp = getattr(hgb, "feature_importances_", None)
        if imp is None or len(imp) != len(feat_labels):
            return []
        order = sorted(range(len(imp)), key=lambda i: float(imp[i]), reverse=True)[:k]
        return [
            {"feature_index": i, "label": feat_labels[i], "importance": round(float(imp[i]), 6)}
            for i in order
        ]
    except Exception:
        return []


def _sklearn_gbdt_available() -> bool:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401

        return True
    except Exception:
        return False


def _fit_gbdt_and_blend_alpha(
    *,
    xz_all: Any,
    y_all: Any,
    ensemble_states: list[dict[str, Any]],
    ensemble_member_weights: list[float] | None,
    best_h: dict[str, Any],
    in_dim: int,
    idx: list[int],
    n: int,
    calib_a: float,
    calib_b: float,
) -> tuple[bytes | None, float | None, float | None, list[dict[str, Any]] | None]:
    """검증 구간에서 NN(캘리브레이션 적용)과 GBDT를 블렌드할 가중치 추정 후 전체 데이터로 GBDT 재학습."""
    if not _sklearn_gbdt_available() or n < 48:
        return None, None, None, None
    import torch
    import torch.nn as nn
    from sklearn.ensemble import HistGradientBoostingRegressor

    va_n = max(3, min(n - 4, int(n * 0.15)))
    va_idx = idx[-va_n:]
    tr_idx = idx[:-va_n]
    if len(tr_idx) < 12:
        return None, None, None, None
    try:
        X_tr = xz_all[tr_idx].detach().cpu().numpy()
        y_tr = y_all[tr_idx].detach().cpu().numpy().ravel()
        X_va = xz_all[va_idx].detach().cpu().numpy()
        y_va = y_all[va_idx].detach().cpu().numpy().ravel()
        hgb = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=5,
            min_samples_leaf=2,
            learning_rate=0.07,
            l2_regularization=1e-4,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=14,
        )
        hgb.fit(X_tr, y_tr)

        xz_va_t = xz_all[va_idx]
        acc = torch.zeros((xz_va_t.shape[0],), device=xz_all.device, dtype=xz_all.dtype)
        k = len(ensemble_states)
        w_list = ensemble_member_weights
        if not w_list or len(w_list) != k:
            w_list = [1.0 / max(k, 1)] * max(k, 1)
        sw = sum(w_list) or 1.0
        w_norm = [float(w) / sw for w in w_list]
        for wi, sd in enumerate(ensemble_states):
            model = _build_model(nn, best_h, in_dim)
            model.load_state_dict(sd)
            model.eval()
            with torch.inference_mode():
                acc += w_norm[wi] * model(xz_va_t).squeeze(-1)
        nn_m = acc
        cal_nn = ((nn_m * calib_a) + calib_b).clamp(0.0, 100.0)
        h_va = torch.as_tensor(hgb.predict(X_va), device=xz_all.device, dtype=xz_all.dtype).clamp(0.0, 100.0)

        best_a = 0.65
        best_mae = float(torch.mean(torch.abs(cal_nn - torch.as_tensor(y_va, device=xz_all.device, dtype=xz_all.dtype))).item())
        for a in (0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
            blended = (float(a) * cal_nn + (1.0 - float(a)) * h_va).clamp(0.0, 100.0)
            mae = float(torch.mean(torch.abs(blended - torch.as_tensor(y_va, device=xz_all.device, dtype=xz_all.dtype))).item())
            if mae < best_mae:
                best_mae = mae
                best_a = float(a)

        X_full = xz_all.detach().cpu().numpy()
        y_full = y_all.detach().cpu().numpy().ravel()
        hgb_full = HistGradientBoostingRegressor(
            max_iter=260,
            max_depth=5,
            min_samples_leaf=2,
            learning_rate=0.06,
            l2_regularization=1e-4,
            random_state=43,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=16,
        )
        hgb_full.fit(X_full, y_full)
        import pickle

        top = _gbdt_feature_importance_top(hgb_full, list(FEATURE_LABELS), 5)
        return (
            pickle.dumps(hgb_full, protocol=pickle.HIGHEST_PROTOCOL),
            best_a,
            best_mae,
            top or None,
        )
    except Exception:
        return None, None, None, None


def _linear_calibration(preds: list[float], ys: list[float]) -> tuple[float, float]:
    if not preds or not ys or len(preds) != len(ys):
        return (1.0, 0.0)
    n = len(preds)
    mean_p = sum(preds) / max(n, 1)
    mean_y = sum(ys) / max(n, 1)
    var_p = sum((p - mean_p) ** 2 for p in preds) / max(n, 1)
    if var_p < 1e-9:
        return (1.0, mean_y - mean_p)
    cov = sum((p - mean_p) * (y - mean_y) for p, y in zip(preds, ys)) / max(n, 1)
    a = cov / var_p
    b = mean_y - a * mean_p
    return (float(a), float(b))


def _pearson_r_squared(preds: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    """검증 풀에 대한 선형 상관·R²(과적합 가능 — 메타 설명과 함께 사용)."""
    n = len(preds)
    if n < 3 or len(ys) != n:
        return None, None
    mp = sum(preds) / n
    mt = sum(ys) / n
    num = sum((p - mp) * (y - mt) for p, y in zip(preds, ys))
    dps = sum((p - mp) ** 2 for p in preds)
    dts = sum((y - mt) ** 2 for y in ys)
    if dps < 1e-12 or dts < 1e-12:
        return 0.0, None
    r = num / (math.sqrt(dps) * math.sqrt(dts))
    r = max(-1.0, min(1.0, float(r)))
    ss_res = sum((y - p) ** 2 for p, y in zip(preds, ys))
    r2 = 1.0 - ss_res / dts if dts > 1e-12 else None
    return r, float(r2) if r2 is not None else None


def _uncertainty_calibration_stats(preds: list[float], ys: list[float]) -> tuple[float, float]:
    """|오차| 분포를 이용해 uncertainty를 보정할 scale/bias를 추정."""
    if not preds or not ys or len(preds) != len(ys):
        return 1.0, 0.0
    abs_err = sorted(abs(float(p) - float(y)) for p, y in zip(preds, ys))
    if not abs_err:
        return 1.0, 0.0
    p50 = abs_err[len(abs_err) // 2]
    p90 = abs_err[min(len(abs_err) - 1, int(len(abs_err) * 0.9))]
    spread = max(0.5, p90 - p50)
    scale = max(0.6, min(2.2, spread / 6.0))
    bias = max(-8.0, min(12.0, p50 - 2.5))
    return float(scale), float(bias)


def _estimate_feature_drift_score(
    rows: list[dict[str, Any]],
    tr_means: list[float] | None,
    tr_stds: list[float] | None,
) -> float | None:
    if not rows or not tr_means or not tr_stds:
        return None
    d = min(len(tr_means), len(tr_stds), FEATURE_DIM)
    if d <= 0:
        return None
    try:
        cur_means = [0.0] * d
        n = 0
        for r in rows:
            x = pad_feature_vector([float(v) for v in (r.get("x") or [])])
            for i in range(d):
                cur_means[i] += x[i]
            n += 1
        if n <= 0:
            return None
        cur_means = [v / n for v in cur_means]
        z = [
            abs((cur_means[i] - float(tr_means[i])) / max(1e-6, float(tr_stds[i])))
            for i in range(d)
        ]
        mean_abs_z = sum(z) / max(len(z), 1)
        return max(0.0, min(100.0, 100.0 - min(100.0, mean_abs_z * 22.0)))
    except Exception:
        return None


def _promotion_gate_decision(
    prev_meta: dict[str, Any] | None,
    cand_meta: dict[str, Any],
) -> tuple[bool, list[str]]:
    """신규 학습 결과를 실제 운영 모델로 승격할지 판단."""
    if not prev_meta:
        return True, ["first_model"]
    reasons: list[str] = []
    allow = True
    mae_tol = float(os.environ.get("TEAM_TORCH_PROMOTE_MAX_MAE_REGRESSION", "0.35") or 0.35)
    hold_tol = float(os.environ.get("TEAM_TORCH_PROMOTE_MAX_HOLDOUT_REGRESSION", "0.5") or 0.5)
    prev_val = prev_meta.get("validation_mae_mean")
    cand_val = cand_meta.get("validation_mae_mean")
    if isinstance(prev_val, (int, float)) and isinstance(cand_val, (int, float)):
        if float(cand_val) > float(prev_val) + mae_tol:
            allow = False
            reasons.append("validation_mae_regressed")
    prev_ho = prev_meta.get("holdout_time_mae")
    cand_ho = cand_meta.get("holdout_time_mae")
    if isinstance(prev_ho, (int, float)) and isinstance(cand_ho, (int, float)):
        if float(cand_ho) > float(prev_ho) + hold_tol:
            allow = False
            reasons.append("holdout_mae_regressed")
    if allow:
        reasons.append("metrics_ok")
    return allow, reasons


def _resolve_input_noise_std_training() -> float:
    """학습 순전파에만 더하는 가우시안 노이즈 표준편차(0이면 비활성)."""
    try:
        v = float((os.environ.get("TEAM_TORCH_INPUT_NOISE_STD", "0") or "0").strip())
    except Exception:
        v = 0.0
    return max(0.0, min(0.5, v))


def _permutation_importance_top_k(
    nn_module: Any,
    state_dict: dict[str, Any],
    best_h: dict[str, Any],
    in_dim: int,
    xz_all: Any,
    y_all: Any,
    va_idx: list[int],
    labels: list[str],
    seed: int,
) -> list[dict[str, Any]] | None:
    """검증 구간에서 피처 열 치환 시 MAE 증가(대략적 중요도). 앙상블 1번째 멤버로 측정."""
    import torch

    if len(va_idx) < 4 or not state_dict:
        return None
    # 큰 검증 집합은 균등 간격으로 줄여 CPU 비용 상한
    if len(va_idx) > 400:
        step = max(1, len(va_idx) // 400)
        va_idx = va_idx[::step][:400]
    xz_va = xz_all[va_idx]
    y_va = y_all[va_idx]
    model = _build_model(nn_module, best_h, in_dim)
    model.load_state_dict(state_dict)
    model.eval()
    g = torch.Generator()
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    with torch.no_grad():
        baseline = float(torch.mean(torch.abs(model(xz_va) - y_va)).item())
    rows: list[dict[str, Any]] = []
    for d in range(in_dim):
        xz_p = xz_va.clone()
        perm = torch.randperm(xz_p.size(0), device=xz_p.device, generator=g)
        xz_p[:, d] = xz_p[perm, d]
        with torch.no_grad():
            mae_p = float(torch.mean(torch.abs(model(xz_p) - y_va)).item())
        lab = labels[d] if d < len(labels) else f"dim_{d}"
        rows.append(
            {
                "feature": lab,
                "delta_mae": round(mae_p - baseline, 6),
            }
        )
    rows.sort(key=lambda r: -abs(float(r["delta_mae"])))
    return rows[:10]


def train_torch_if_needed() -> dict[str, Any]:
    if not torch_available():
        return {"enabled": False, "reason": "torch_unavailable"}

    import torch
    import torch.nn as nn

    meta = _load_meta() or {}
    trained_count = int(meta.get("sample_count", 0) or 0)
    rmax = int(os.environ.get("TEAM_TRAIN_RESERVOIR_MAX", "0") or 0)
    rseed = int(os.environ.get("TEAM_TRAIN_RESERVOIR_SEED", "42") or 42)
    data, n_pool, n_used = resolve_training_rows(reservoir_max=rmax, reservoir_seed=rseed)
    data = sort_rows_by_created_at(data)
    iqr_k = float(os.environ.get("TEAM_TRAIN_OUTLIER_IQR_K", "0") or 0)
    data, training_outlier_stats = filter_training_rows_outliers(data, iqr_k)
    hold_frac = float(os.environ.get("TEAM_HOLDOUT_TIME_FRAC", "0") or 0)
    data_train, data_hold, ho_info = time_holdout_split(data, hold_frac)
    data_eff = data_train if ho_info.get("active") else data
    data_hold_list: list[dict] = list(data_hold) if ho_info.get("active") else []
    n = len(data_eff)
    sess_ids = [str(r.get("sess") or "") for r in data_eff]
    tp = _training_profile()
    saved_prof = str(meta.get("model_size_profile") or "standard")
    incompat = (
        int(meta.get("feature_version", 0) or 0) != FEATURE_VERSION
        or int(meta.get("input_dim", 0) or 0) != FEATURE_DIM
        or saved_prof != tp["name"]
    )
    if n_pool < MIN_TRAIN_SAMPLES or n < MIN_TRAIN_SAMPLES:
        return {"enabled": True, "trained": False, "sample_count": n_pool, "reason": "insufficient_samples"}
    if (
        TORCH_MODEL_PATH.exists()
        and (n_pool - trained_count) < RETRAIN_INTERVAL
        and not incompat
        and not retrain_forced_by_env()
    ):
        drift_score = _estimate_feature_drift_score(
            data_eff,
            meta.get("training_feature_means"),
            meta.get("training_feature_stds"),
        )
        drift_min = float(os.environ.get("TEAM_TORCH_RETRAIN_DRIFT_MIN", "58") or 58.0)
        if drift_score is not None and drift_score < drift_min:
            # 분포가 학습 시점과 크게 달라졌으면 간격 미도달이어도 재학습한다.
            pass
        else:
            return {
                "enabled": True,
                "trained": False,
                "sample_count": n_pool,
                "model_version": meta.get("model_version"),
                "trained_at": meta.get("trained_at"),
                "reason": "interval_not_reached",
            }

    run_seed = _resolve_torch_seed()
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    input_noise_std = _resolve_input_noise_std_training()
    residual_flag = (os.environ.get("TEAM_TORCH_RESIDUAL_MLP", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    x_all = torch.tensor(
        [pad_feature_vector([float(v) for v in r["x"]]) for r in data_eff], dtype=torch.float32
    )
    in_dim = int(x_all.shape[1])
    y_all = torch.tensor([[max(0.0, min(100.0, float(r["y"])))] for r in data_eff], dtype=torch.float32)
    means = x_all.mean(dim=0, keepdim=True)
    stds = x_all.std(dim=0, keepdim=True)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    xz_all = (x_all - means) / stds

    idx = list(range(n))
    random.Random(run_seed).shuffle(idx)
    fold_count = max(2, min(int(tp["cv_folds"]), n))
    fold_specs, cv_strategy_name = _cv_fold_train_val_indices(n, sess_ids, fold_count, random.Random(run_seed))
    use_rolling = (os.environ.get("TEAM_TORCH_ROLLING_CV", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if use_rolling:
        roll = _rolling_time_fold_indices(n, fold_count)
        if roll:
            fold_specs = roll
            cv_strategy_name = "rolling_time"
    cv_unique_groups = len(set(_row_group_ids(sess_ids)))

    # hyperparameter + architecture search with CV MAE
    best_h = {**tp["hparam_grid"][0], **tp["arch_grid"][0], "use_residual_mlp": residual_flag}
    best_cv_mae = float("inf")
    for h_base in tp["hparam_grid"]:
        for arch in tp["arch_grid"]:
            h = {**h_base, **arch, "use_residual_mlp": residual_flag}
            fold_maes: list[float] = []
            for tr_idx, va_idx in fold_specs:
                if not va_idx or not tr_idx:
                    continue
                x_tr = xz_all[tr_idx]
                y_tr = y_all[tr_idx]
                x_va = xz_all[va_idx]
                y_va = y_all[va_idx]

                model = _build_model(nn, h, in_dim)
                opt = torch.optim.AdamW(
                    model.parameters(), lr=float(h["lr"]), weight_decay=float(h["weight_decay"])
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(12, int(h["epochs"])))
                loss_fn = nn.SmoothL1Loss()
                best_state = None
                best_loss = float("inf")
                no_improve = 0
                patience = _early_stop_patience(n)
                model.train()
                for _ in range(int(h["epochs"])):
                    opt.zero_grad()
                    pred = model(x_tr)
                    loss = loss_fn(pred, y_tr)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    scheduler.step()

                    model.eval()
                    with torch.no_grad():
                        val_loss = float(loss_fn(model(x_va), y_va).item())
                    model.train()
                    if val_loss + 1e-7 < best_loss:
                        best_loss = val_loss
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break
                if best_state is not None:
                    model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    mae = torch.mean(torch.abs(model(x_va) - y_va)).item()
                fold_maes.append(float(mae))
            if fold_maes:
                cv_mae = sum(fold_maes) / len(fold_maes)
                if cv_mae < best_cv_mae:
                    best_cv_mae = cv_mae
                    best_h = h

    ensemble_states: list[dict[str, Any]] = []
    val_maes: list[float] = []
    calib_preds: list[float] = []
    calib_targets: list[float] = []
    for m_i in range(int(tp["ensemble_size"])):
        split = max(1, int(n * 0.86))
        tr_idx = idx[:split]
        va_idx = idx[split:] if split < n else idx[-1:]
        rng = random.Random(run_seed + 1000 + m_i)
        bs_idx = [tr_idx[rng.randrange(len(tr_idx))] for _ in range(len(tr_idx))]
        x_tr = xz_all[bs_idx]
        y_tr = y_all[bs_idx]
        x_va = xz_all[va_idx]
        y_va = y_all[va_idx]

        model = _build_model(nn, best_h, in_dim)
        opt = torch.optim.AdamW(
            model.parameters(), lr=float(best_h["lr"]), weight_decay=float(best_h["weight_decay"])
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(12, int(best_h["epochs"])))
        loss_fn = nn.SmoothL1Loss()
        best_state = None
        best_loss = float("inf")
        no_improve = 0
        ema_state = None
        patience = _early_stop_patience(n)
        model.train()
        for _ in range(int(best_h["epochs"])):
            opt.zero_grad()
            x_in = x_tr
            if input_noise_std > 0:
                x_in = x_tr + torch.randn_like(x_tr) * input_noise_std
            pred = model(x_in)
            loss = loss_fn(pred, y_tr)
            # 세션(동일 요청) 내 순위 보조 손실 — 룰 순위와 독립적인 학습 신호
            gp: dict[str, list[int]] = defaultdict(list)
            for pos, row_i in enumerate(bs_idx):
                sid = sess_ids[row_i] if row_i < len(sess_ids) else ""
                if sid:
                    gp[sid].append(pos)
            rank_pen = torch.zeros((), device=pred.device, dtype=pred.dtype)
            pair_n = 0
            for positions in gp.values():
                if len(positions) < 2:
                    continue
                for ia in positions:
                    for ib in positions:
                        if ia == ib:
                            continue
                        if float(y_tr[ia]) > float(y_tr[ib]) + 0.75:
                            rank_pen = rank_pen + torch.relu(
                                1.0 - (pred[ia] - pred[ib])
                            )
                            pair_n += 1
            if pair_n > 0:
                loss = loss + 0.06 * (rank_pen / pair_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            with torch.no_grad():
                if ema_state is None:
                    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(0.99).add_(v.detach(), alpha=0.01)

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x_va), y_va).item())
            model.train()
            if val_loss + 1e-7 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        mae_ema: float | None = None
        mae_best: float | None = None
        if ema_state is not None:
            model.load_state_dict(ema_state)
            model.eval()
            with torch.no_grad():
                mae_ema = float(torch.mean(torch.abs(model(x_va) - y_va)).item())
        if best_state is not None:
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                mae_best = float(torch.mean(torch.abs(model(x_va) - y_va)).item())
        if mae_ema is not None and mae_best is not None:
            if mae_best <= mae_ema:
                model.load_state_dict(best_state)
            else:
                model.load_state_dict(ema_state)
        elif best_state is not None:
            model.load_state_dict(best_state)
        elif ema_state is not None:
            model.load_state_dict(ema_state)
        model.eval()
        with torch.no_grad():
            p_va = model(x_va)
            mae = torch.mean(torch.abs(p_va - y_va)).item()
        val_maes.append(float(mae))
        calib_preds.extend([float(v) for v in p_va.view(-1).tolist()])
        calib_targets.extend([float(v) for v in y_va.view(-1).tolist()])
        ensemble_states.append(model.state_dict())

    eps_w = 1e-3
    w_raw = [1.0 / (float(v) + eps_w) for v in val_maes]
    sw = sum(w_raw) or 1.0
    ensemble_member_weights = [w / sw for w in w_raw]

    split_perm = max(1, int(n * 0.86))
    va_idx_perm = idx[split_perm:] if split_perm < n else idx[-1:]
    perm_top = _permutation_importance_top_k(
        nn,
        ensemble_states[0],
        best_h,
        in_dim,
        xz_all,
        y_all,
        va_idx_perm,
        FEATURE_LABELS,
        run_seed + 4242,
    )

    tail_k = max(3, min(max(1, int(n * 0.12)), n - 1))
    x_tail = xz_all[-tail_k:]
    y_tail = y_all[-tail_k:]
    tail_maes: list[float] = []
    for sd in ensemble_states:
        model = _build_model(nn, best_h, in_dim)
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            p_t = model(x_tail)
            tail_maes.append(float(torch.mean(torch.abs(p_t - y_tail)).item()))
    chronological_tail_mae_mean = round(sum(tail_maes) / max(len(tail_maes), 1), 4)

    probe = _build_model(nn, best_h, in_dim)
    approx_params = _count_params(probe)

    calib_a, calib_b = _linear_calibration(calib_preds, calib_targets)
    calib_b = max(-12.0, min(12.0, calib_b))
    calib_a = max(0.6, min(1.4, calib_a))
    unc_scale, unc_bias = _uncertainty_calibration_stats(calib_preds, calib_targets)
    calib_pr, calib_r2 = _pearson_r_squared(calib_preds, calib_targets)
    calib_fit_note_ko = (
        "검증에 쓰인 예측(선형 보정 전)·라벨의 피어슨·R². 학습 데이터 위에서 측정되어 낙관될 수 있음."
    )

    holdout_time_mae = None
    holdout_time_pearson_r = None
    holdout_time_r2 = None
    holdout_time_note_ko = (
        "created_at 기준 최근 구간을 학습에서 제외한 뒤, 그 구간에서만 MAE·피어슨·R²를 계산합니다."
        if ho_info.get("active")
        else None
    )
    if data_hold_list and len(data_hold_list) >= 3 and ho_info.get("active"):
        nh = len(data_hold_list)
        x_h = torch.tensor(
            [pad_feature_vector([float(v) for v in r["x"]]) for r in data_hold_list],
            dtype=torch.float32,
            device=x_all.device,
        )
        xz_h = (x_h - means) / stds
        acc_p = torch.zeros((nh,), device=x_all.device, dtype=x_all.dtype)
        for wi, sd in enumerate(ensemble_states):
            model = _build_model(nn, best_h, in_dim)
            model.load_state_dict(sd)
            model.eval()
            with torch.inference_mode():
                acc_p += float(ensemble_member_weights[wi]) * model(xz_h).squeeze(-1)
        pred_raw = acc_p
        pred_cal = (pred_raw * calib_a + calib_b).clamp(0.0, 100.0)
        y_vec = torch.tensor(
            [max(0.0, min(100.0, float(r["y"]))) for r in data_hold_list],
            device=x_all.device,
            dtype=x_all.dtype,
        )
        holdout_time_mae = float(torch.mean(torch.abs(pred_cal - y_vec)).item())
        py = [float(v) for v in pred_cal.detach().cpu().tolist()]
        yy = [float(v) for v in y_vec.detach().cpu().tolist()]
        hpr, hr2 = _pearson_r_squared(py, yy)
        holdout_time_pearson_r = round(float(hpr), 4) if hpr is not None else None
        holdout_time_r2 = round(float(hr2), 4) if hr2 is not None else None

    gbdt_blob, gbdt_blend_a, gbdt_blend_mae_va, gbdt_top_feats = _fit_gbdt_and_blend_alpha(
        xz_all=xz_all,
        y_all=y_all,
        ensemble_states=ensemble_states,
        ensemble_member_weights=ensemble_member_weights,
        best_h=best_h,
        in_dim=in_dim,
        idx=idx,
        n=n,
        calib_a=calib_a,
        calib_b=calib_b,
    )
    ds_sha = dataset_file_sha256()

    state = {
        "ensemble_state_dicts": ensemble_states,
        "ensemble_member_weights": ensemble_member_weights,
        "ensemble_validation_maes": [round(float(v), 4) for v in val_maes],
        "means": means.squeeze(0).tolist(),
        "stds": stds.squeeze(0).tolist(),
        "calibration_a": calib_a,
        "calibration_b": calib_b,
        "uncertainty_scale": unc_scale,
        "uncertainty_bias": unc_bias,
        "best_hparams": best_h,
        "best_architecture": best_h.get("name", "relu"),
        "feature_version": FEATURE_VERSION,
        "input_dim": in_dim,
        "model_size_profile": tp["name"],
        "mc_dropout_samples": int(tp["mc_dropout"]),
        "torch_seed": run_seed,
        "approx_parameters": approx_params,
        "gbdt_present": bool(gbdt_blob is not None and gbdt_blend_a is not None),
        "nn_gbdt_blend_alpha": float(gbdt_blend_a) if gbdt_blend_a is not None else None,
    }
    if gbdt_blob is not None and gbdt_blend_a is not None:
        state["gbdt_pickled"] = gbdt_blob
    now = datetime.now(timezone.utc)
    out_meta = {
        "model_version": f"team-torch-{now.strftime('%Y%m%d%H%M%S')}",
        "trained_at": now.isoformat(),
        "sample_count": n_pool,
        "training_rows_used": n_used,
        "dataset_path": str(DATASET_PATH),
        "ensemble_size": int(tp["ensemble_size"]),
        "model_size_profile": tp["name"],
        "approx_parameters": approx_params,
        "mc_dropout_samples": int(tp["mc_dropout"]),
        "torch_seed": run_seed,
        "validation_mae_mean": round(sum(val_maes) / max(len(val_maes), 1), 4),
        "validation_mae_each": [round(v, 4) for v in val_maes],
        "cv_mae_mean": round(best_cv_mae, 4) if best_cv_mae != float("inf") else None,
        "cv_split_strategy": cv_strategy_name,
        "cv_unique_groups": cv_unique_groups,
        "cv_vs_validation_gap": (
            round(
                abs(
                    float(best_cv_mae)
                    - (sum(val_maes) / max(len(val_maes), 1))
                ),
                4,
            )
            if best_cv_mae != float("inf")
            else None
        ),
        "chronological_tail_mae_mean": chronological_tail_mae_mean,
        "chronological_tail_fraction": round(tail_k / max(n, 1), 4),
        "chronological_tail_note": "동일 데이터로 학습된 모델의 시간순 꼬리 구간 MAE; 드리프트·포화 감지용(순수 홀드아웃 아님).",
        "best_hparams": best_h,
        "best_architecture": best_h.get("name", "relu"),
        "calibration": {"a": round(calib_a, 4), "b": round(calib_b, 4)},
        "uncertainty_calibration": {"scale": round(unc_scale, 4), "bias": round(unc_bias, 4)},
        "feature_version": FEATURE_VERSION,
        "input_dim": in_dim,
        "training_feature_means": [round(float(v), 6) for v in means.squeeze(0).tolist()],
        "training_feature_stds": [round(float(v), 6) for v in stds.squeeze(0).tolist()],
        "gbdt_present": bool(gbdt_blob is not None and gbdt_blend_a is not None),
        "nn_gbdt_blend_alpha": round(float(gbdt_blend_a), 4) if gbdt_blend_a is not None else None,
        "gbdt_validation_blend_mae": round(float(gbdt_blend_mae_va), 4) if gbdt_blend_mae_va is not None else None,
        "dataset_file_sha256": ds_sha,
        "gbdt_top_features": gbdt_top_feats if gbdt_top_feats else None,
        "dl_roadmap_version": ROADMAP_VERSION,
        "calibration_pearson_r": round(float(calib_pr), 4) if calib_pr is not None else None,
        "calibration_r2": round(float(calib_r2), 4) if calib_r2 is not None else None,
        "calibration_fit_note_ko": calib_fit_note_ko,
        "label_spec_version": LABEL_SPEC_VERSION,
        "training_outlier_stats": training_outlier_stats,
        "time_holdout_split": ho_info,
        "effective_training_rows": n,
        "holdout_time_mae": round(float(holdout_time_mae), 4) if holdout_time_mae is not None else None,
        "holdout_time_pearson_r": holdout_time_pearson_r,
        "holdout_time_r2": holdout_time_r2,
        "holdout_time_note_ko": holdout_time_note_ko,
        "input_noise_std_training": round(float(input_noise_std), 6),
        "permutation_importance_top": perm_top,
        "semantic_encoder": semantic_encoder_meta(),
        "training_pool_resolve": last_training_resolve_meta(),
        "ensemble_stacking": {
            "weights": [round(float(w), 6) for w in ensemble_member_weights],
            "strategy": "inverse_validation_mae",
        },
    }
    out_meta["dl_quality_unified"] = build_dl_quality_unified(out_meta)
    promote_ok, promote_reasons = _promotion_gate_decision(meta, out_meta)
    out_meta["promotion_gate"] = {
        "accepted": promote_ok,
        "reasons": promote_reasons,
    }
    if not promote_ok:
        return {
            "enabled": True,
            "trained": False,
            "sample_count": n_pool,
            "model_version": meta.get("model_version"),
            "trained_at": meta.get("trained_at"),
            "reason": "promotion_gate_rejected",
            "candidate": out_meta,
        }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(state, TORCH_MODEL_PATH)
    _save_meta(out_meta)
    invalidate_torch_predict_cache()
    return {"enabled": True, "trained": True, **out_meta}


def _get_predict_state() -> dict[str, Any] | None:
    global _TORCH_PREDICT_CACHE
    if not torch_available() or not TORCH_MODEL_PATH.exists():
        return None
    import torch

    key = str(TORCH_MODEL_PATH.resolve())
    try:
        mtime = TORCH_MODEL_PATH.stat().st_mtime
    except OSError:
        return None
    if _TORCH_PREDICT_CACHE and _TORCH_PREDICT_CACHE[0] == key and _TORCH_PREDICT_CACHE[1] == mtime:
        return _TORCH_PREDICT_CACHE[2]
    st = torch.load(TORCH_MODEL_PATH, map_location="cpu")
    if not isinstance(st, dict):
        return None
    _TORCH_PREDICT_CACHE = (key, mtime, st)
    return st


def predict_torch_scores_batched(feature_rows: list[list[float]]) -> list[dict[str, float]] | None:
    """팀 단위 배치 추론 — 멤버 수만큼 반복 호출 대비 대폭 감축."""
    if not feature_rows:
        return []
    if not torch_available():
        return None
    import torch
    import torch.nn as nn

    state = _get_predict_state()
    if state is None:
        return None
    sds = state.get("ensemble_state_dicts")
    if not isinstance(sds, list) or not sds:
        return None
    in_dim = int(state.get("input_dim") or FEATURE_DIM)
    rows = [pad_feature_vector(r) for r in feature_rows]
    if any(len(r) != len(rows[0]) for r in rows):
        return None
    means_list = state.get("means")
    if not isinstance(means_list, list) or len(means_list) != in_dim:
        return None
    x = torch.tensor(rows, dtype=torch.float32)
    if x.shape[1] != in_dim:
        return None
    means = torch.tensor(means_list, dtype=torch.float32).unsqueeze(0)
    stds = torch.tensor(state["stds"], dtype=torch.float32).unsqueeze(0)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    xz = (x - means) / stds
    h = state.get("best_hparams") or {**HYPERPARAM_CANDIDATES_STANDARD[0], **ARCH_CANDIDATES[0]}
    mc_n = int(state.get("mc_dropout_samples") or STANDARD_MC_DROPOUT)
    raw_w = state.get("ensemble_member_weights")
    n_mem = len(sds)
    if isinstance(raw_w, list) and len(raw_w) == n_mem:
        sw = sum(float(w) for w in raw_w) or 1.0
        w_mem = [float(w) / sw for w in raw_w]
    else:
        w_mem = [1.0 / max(n_mem, 1)] * n_mem
    det_rows: list[torch.Tensor] = []
    mc_std_rows: list[torch.Tensor] = []
    for sd in sds:
        model = _build_model(nn, h, in_dim)
        model.load_state_dict(sd)
        model.eval()
        with torch.inference_mode():
            det = model(xz).squeeze(-1)
        det_rows.append(det)
        model.train()
        mc_rows: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(mc_n):
                mc_rows.append(model(xz).squeeze(-1))
        if mc_rows:
            mc_stack = torch.stack(mc_rows, dim=0)
            mc_std_rows.append(mc_stack.std(dim=0, unbiased=False))
        else:
            mc_std_rows.append(torch.zeros_like(det))
    det_mat = torch.stack(det_rows, dim=0)
    w_t = torch.tensor(w_mem, device=xz.device, dtype=xz.dtype).view(-1, 1)
    mean = (det_mat * w_t).sum(dim=0)
    mean_exp = mean.unsqueeze(0).expand_as(det_mat)
    std_between = (w_t * (det_mat - mean_exp) ** 2).sum(dim=0).sqrt()
    mc_mat = torch.stack(mc_std_rows, dim=0)
    std_mc = (mc_mat * w_t).sum(dim=0)
    std = (std_mc**2 + std_between**2).sqrt()
    calib_a = float(state.get("calibration_a", 1.0))
    calib_b = float(state.get("calibration_b", 0.0))
    calibrated = (mean * calib_a) + calib_b
    z_dist = xz.abs().mean(dim=1)
    ood_penalty = (z_dist - 2.2).clamp(min=0.0) * 4.0
    uncertainty = (std + ood_penalty).clamp(0.0, 100.0)
    u_scale = float(state.get("uncertainty_scale") or 1.0)
    u_bias = float(state.get("uncertainty_bias") or 0.0)
    uncertainty = (uncertainty * u_scale + u_bias).clamp(0.0, 100.0)
    calibrated = calibrated.clamp(0.0, 100.0)
    if state.get("gbdt_present") and state.get("gbdt_pickled"):
        try:
            import pickle

            hgb = pickle.loads(state["gbdt_pickled"])
            alpha = float(state.get("nn_gbdt_blend_alpha") or 0.65)
            g = torch.as_tensor(
                hgb.predict(xz.detach().cpu().numpy()),
                device=calibrated.device,
                dtype=calibrated.dtype,
            ).clamp(0.0, 100.0)
            calibrated = (alpha * calibrated + (1.0 - alpha) * g).clamp(0.0, 100.0)
        except Exception:
            pass
    out: list[dict[str, float]] = []
    for i in range(x.shape[0]):
        out.append(
            {
                "score": float(calibrated[i].item()),
                "uncertainty": float(uncertainty[i].item()),
            }
        )
    return out


def predict_torch_score(
    commits: int,
    prs: int,
    lines: int,
    attendance: float,
    self_report: str,
    *,
    member_rank: int = 1,
    team_size: int = 1,
    median_commits: float = 0.0,
    median_prs: float = 0.0,
    median_lines: float = 0.0,
    median_attendance: float = 50.0,
    median_words: float = 1.0,
    hist_blend: float = 0.5,
    hist_rule: float = 0.5,
    hist_density: float = 0.0,
) -> dict[str, float] | None:
    fv = build_feature_vector(
        commits,
        prs,
        lines,
        attendance,
        self_report,
        member_rank=member_rank,
        team_size=team_size,
        median_commits=median_commits,
        median_prs=median_prs,
        median_lines=median_lines,
        median_attendance=median_attendance,
        median_words=median_words,
        hist_blend=hist_blend,
        hist_rule=hist_rule,
        hist_density=hist_density,
    )
    batched = predict_torch_scores_batched([fv])
    if batched is None or len(batched) != 1:
        return None
    return batched[0]


def torch_model_meta() -> dict[str, Any]:
    m = _load_meta() or {}
    if m and "dl_quality_unified" not in m:
        m = {**m, "dl_quality_unified": build_dl_quality_unified(m)}
    return m


"""Lightweight data-driven contribution model (no external ML deps)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edu_tools.team_lm_embedding import embed8_self_report

DATA_DIR = Path(__file__).with_name("data")
DATASET_PATH = DATA_DIR / "team_ml_dataset.jsonl"
MODEL_PATH = DATA_DIR / "team_ml_model.json"

MIN_TRAIN_SAMPLES = 40
RETRAIN_INTERVAL = 20

# 확장 피처 + 복합 타깃 + DB 이력 + 자기서술 형태 + 선택적 문장 임베딩(v6)
FEATURE_VERSION = 6
FEATURE_DIM = 34
# hybrid_dl_target 정의(저장 라벨과 함께 기록) — 바꾸면 과거 행과 혼합 시 주의
LABEL_SPEC_VERSION = 1
STRUCTURAL_TARGET_WEIGHT = 0.42


def retrain_forced_by_env() -> bool:
    """TEAM_RETRAIN_FORCE=1 등이면 재학습 간격(RETRAIN_INTERVAL)을 무시."""
    v = (os.environ.get("TEAM_RETRAIN_FORCE", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def dataset_file_sha256() -> str | None:
    """team_ml_dataset.jsonl 전체 바이트 SHA-256(재현성). 없으면 None."""
    if not DATASET_PATH.is_file():
        return None
    h = hashlib.sha256()
    with DATASET_PATH.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class TeamMlModel:
    version: str
    trained_at: str
    sample_count: int
    means: list[float]
    stds: list[float]
    weights: list[float]
    bias: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wc(text: str) -> int:
    return len([x for x in text.split() if x.strip()])


def feature_row(commits: int, prs: int, lines: int, attendance: float, self_report: str) -> list[float]:
    return [
        math.log1p(max(0, commits)),
        math.log1p(max(0, prs)),
        math.log1p(max(0, lines)),
        max(0.0, min(100.0, float(attendance))) / 100.0,
        math.log1p(max(0, _wc(self_report))),
    ]


FEATURE_LABELS = [
    "log_commits",
    "log_prs",
    "log_lines",
    "attendance_norm",
    "log_self_report_words",
    "log_report_chars",
    "pr_vs_commit",
    "lines_per_pr",
    "activity_density",
    "rel_log_commits",
    "rel_log_prs",
    "rel_log_lines",
    "rel_log_words",
    "rank_in_team",
    "team_size_norm",
    "att_vs_team_median",
    "hist_blended_prior",
    "hist_rule_prior",
    "hist_eval_density",
    "essay_depth_norm",
    "rel_git_balance",
    "text_unique_ratio",
    "text_chars_per_word",
    "text_newline_density",
    "text_digit_ratio",
    "text_long_token_ratio",
    # TEAM_SEMANTIC_ENCODER=1 일 때만 비영(의미); 0이면 0으로 채움
    "lm_semantic_e0",
    "lm_semantic_e1",
    "lm_semantic_e2",
    "lm_semantic_e3",
    "lm_semantic_e4",
    "lm_semantic_e5",
    "lm_semantic_e6",
    "lm_semantic_e7",
]


def pad_feature_vector(x: list[float]) -> list[float]:
    """이전 5차원 샘플을 신규 차원으로 패딩."""
    out = [float(v) for v in x[:FEATURE_DIM]]
    while len(out) < FEATURE_DIM:
        out.append(0.0)
    return out


def structural_absolute_score(commits: int, prs: int, lines: int, attendance: float, self_report: str) -> float:
    """팀 내 순위와 무관한 포화형 절대 활동 점수(0~100). 룰 엔진과 다른 신호를 학습에 섞는다."""
    w = _wc(self_report)
    c = max(0, commits)
    p = max(0, prs)
    ln = max(0, lines)
    att = max(0.0, min(100.0, float(attendance)))
    s_git = min(38.0, 7.5 * math.log1p(c + 1) + 9.0 * math.log1p(p + 1))
    s_lines = min(34.0, 5.8 * math.log1p(ln + 1))
    s_att = min(16.0, att * 0.14)
    s_txt = min(12.0, 2.2 * math.log1p(w + 1) + 0.35 * math.log1p(max(0, len(self_report.strip())) + 1))
    return max(0.0, min(100.0, s_git + s_lines + s_att + s_txt))


def text_shape_features(self_report: str) -> list[float]:
    """자기서술 텍스트에서 추출하는 결정론적 형태 피처(외부 모델 없음)."""
    t = (self_report or "").strip()
    if not t:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    words = [x for x in t.split() if x.strip()]
    n_w = len(words)
    uniq = len({w.lower() for w in words}) if words else 0
    uniq_ratio = uniq / max(1, n_w)
    c_pw = len(t) / max(1, n_w)
    c_pw_n = min(1.0, math.log1p(c_pw) / 4.5)
    nl = t.count("\n")
    nl_n = min(1.0, (nl / max(1, len(t))) * 12.0)
    digits = sum(1 for ch in t if ch.isdigit())
    dig_r = min(1.0, (digits / max(1, len(t))) * 8.0)
    long_w = sum(1 for w in words if len(w) > 6)
    long_r = long_w / max(1, n_w)
    return [
        max(0.0, min(1.0, uniq_ratio)),
        max(0.0, min(1.0, c_pw_n)),
        max(0.0, min(1.0, nl_n)),
        max(0.0, min(1.0, dig_r)),
        max(0.0, min(1.0, long_r)),
    ]


def hybrid_dl_target(normalized_score: float, commits: int, prs: int, lines: int, attendance: float, self_report: str) -> float:
    """순위 정규화 점수 + 절대 활동 점수 혼합(저장 라벨)."""
    s = structural_absolute_score(commits, prs, lines, attendance, self_report)
    a = STRUCTURAL_TARGET_WEIGHT
    return max(0.0, min(100.0, (1.0 - a) * float(normalized_score) + a * s))


def build_feature_vector(
    commits: int,
    prs: int,
    lines: int,
    attendance: float,
    self_report: str,
    *,
    member_rank: int,
    team_size: int,
    median_commits: float,
    median_prs: float,
    median_lines: float,
    median_attendance: float,
    median_words: float,
    hist_blend: float = 0.5,
    hist_rule: float = 0.5,
    hist_density: float = 0.0,
) -> list[float]:
    """팀 문맥(상대 지표·순위)을 포함한 확장 입력 벡터."""
    base = feature_row(commits, prs, lines, attendance, self_report)
    c = max(0, commits)
    p = max(0, prs)
    ln = max(0, lines)
    w = _wc(self_report)
    chars = len(self_report.strip())
    med_c = max(0.0, float(median_commits))
    med_p = max(0.0, float(median_prs))
    med_l = max(0.0, float(median_lines))
    med_w = max(0.0, float(median_words))
    med_att = max(0.0, min(100.0, float(median_attendance)))

    log_chars = math.log1p(max(0, chars))
    pr_vs_commit = math.log1p(p + 1) - math.log1p(max(1, c))
    lines_per_pr = math.log1p(ln / max(1, p))
    density = math.log1p(ln + 1) / max(0.45, math.log1p(c + p + 2))
    density = min(3.5, density)

    rel_c = math.log1p(c) - math.log1p(max(1.0, med_c))
    rel_p = math.log1p(p) - math.log1p(max(1.0, med_p))
    rel_l = math.log1p(ln) - math.log1p(max(1.0, med_l))
    rel_w = math.log1p(w) - math.log1p(max(1.0, med_w))

    ts = max(1, int(team_size))
    rank_norm = (float(max(1, member_rank)) - 1.0) / max(1.0, float(ts - 1)) if ts > 1 else 0.5
    team_size_norm = (ts - 1) / 39.0
    att_vs = (max(0.0, min(100.0, float(attendance))) - med_att) / 100.0

    hb = max(0.0, min(1.0, float(hist_blend)))
    hr = max(0.0, min(1.0, float(hist_rule)))
    hd = max(0.0, min(1.0, float(hist_density)))
    essay_depth = min(1.0, math.log1p(max(0, chars)) / max(1e-6, math.log1p(max(1, w)) * 2.0))
    rel_git_balance = min(1.0, abs(rel_c - rel_p) / 3.0)
    txt = text_shape_features(self_report)

    vec = base + [
        log_chars,
        pr_vs_commit,
        lines_per_pr,
        density,
        rel_c,
        rel_p,
        rel_l,
        rel_w,
        rank_norm,
        team_size_norm,
        att_vs,
        hb,
        hr,
        hd,
        essay_depth,
        rel_git_balance,
    ] + txt + embed8_self_report(self_report)
    assert len(vec) == FEATURE_DIM
    return vec


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


FP_RING_PATH = DATA_DIR / "team_ml_fp_ring.json"
FP_RING_MAX = 8192


def _append_fingerprint(row: dict) -> str:
    x = row.get("x")
    y = float(row.get("y", 0.0))
    memb = str(row.get("member", ""))
    sess = str(row.get("sess", ""))
    xs = [round(float(v), 6) for v in (x or [])]
    key = json.dumps(
        {"m": memb, "s": sess, "x": xs, "y": round(y, 4)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _fp_ring_load() -> list[str]:
    if not FP_RING_PATH.is_file():
        return []
    try:
        raw = json.loads(FP_RING_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("fingerprints"), list):
            return [str(x) for x in raw["fingerprints"] if isinstance(x, str)]
    except Exception:
        pass
    return []


def _fp_ring_save(ring: list[str]) -> None:
    _ensure_data_dir()
    ring = ring[-FP_RING_MAX:]
    FP_RING_PATH.write_text(
        json.dumps({"fingerprints": ring}, ensure_ascii=False),
        encoding="utf-8",
    )


def append_samples(rows: list[dict]) -> dict[str, Any]:
    """JSONL에 추가. TEAM_APPEND_DEDUP=1 이면 최근 링에 있는 동일 지문 행은 건너뜀."""
    _ensure_data_dir()
    dedup = (os.environ.get("TEAM_APPEND_DEDUP", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    ring = _fp_ring_load() if dedup else []
    ring_set = set(ring)
    written = 0
    skipped = 0
    with DATASET_PATH.open("a", encoding="utf-8") as f:
        for r in rows:
            if dedup:
                fp = _append_fingerprint(r)
                if fp in ring_set:
                    skipped += 1
                    continue
                ring.append(fp)
                ring_set.add(fp)
                if len(ring) > FP_RING_MAX:
                    ring = ring[-FP_RING_MAX:]
                    ring_set = set(ring)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1
    if dedup and ring:
        _fp_ring_save(ring)
    n_lines = count_dataset_lines()
    return {
        "written": written,
        "skipped_duplicate": skipped,
        "dataset_line_count": n_lines,
        "dedup_enabled": dedup,
    }


def dataset_label_streaming_stats(*, max_lines: int = 100_000) -> dict[str, Any]:
    """대용량 JSONL도 한 패스로 라벨(y) 분포 요약(수집 품질 모니터링)."""
    if not DATASET_PATH.is_file():
        return {
            "lines_scanned": 0,
            "y_present": 0,
            "error": "no_dataset",
            "dataset_path": str(DATASET_PATH),
        }
    n = 0
    y_sum = 0.0
    y_sq = 0.0
    y_min = float("inf")
    y_max = float("-inf")
    ys_sample: list[float] = []
    cap = max(100, min(int(max_lines), 500_000))
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if n >= cap:
                break
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict) or "y" not in obj:
                    continue
                y = float(obj["y"])
                y = max(0.0, min(100.0, y))
            except Exception:
                continue
            n += 1
            y_sum += y
            y_sq += y * y
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            if len(ys_sample) < 2000:
                ys_sample.append(y)
    if n == 0:
        return {
            "lines_scanned": 0,
            "y_present": 0,
            "error": "empty_or_unreadable",
            "dataset_path": str(DATASET_PATH),
        }
    mean = y_sum / n
    var = max(0.0, y_sq / n - mean * mean)
    std = math.sqrt(var)
    ys_sample.sort()

    def _q(qv: float) -> float | None:
        if not ys_sample:
            return None
        i = int(round((len(ys_sample) - 1) * qv))
        i = max(0, min(len(ys_sample) - 1, i))
        return round(float(ys_sample[i]), 4)

    return {
        "lines_scanned": n,
        "y_present": n,
        "y_min": round(y_min, 4),
        "y_max": round(y_max, 4),
        "y_mean": round(mean, 4),
        "y_std": round(std, 4),
        "y_p05": _q(0.05),
        "y_p95": _q(0.95),
        "dataset_path": str(DATASET_PATH),
    }


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        return []
    out: list[dict] = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
            if isinstance(x, dict) and "x" in x and "y" in x:
                x = dict(x)
                x["x"] = pad_feature_vector([float(v) for v in x["x"]])
                out.append(x)
        except Exception:
            continue
    return out


def count_dataset_lines() -> int:
    """JSONL 줄 수(대용량 데이터 풀 크기 추정용, 파싱 없이 순회)."""
    if not DATASET_PATH.exists():
        return 0
    n = 0
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _dedupe_training_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """지문 기준 중복 제거(먼저 나온 행 유지)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        fp = _append_fingerprint(r)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(r)
    return out, len(rows) - len(out)


def _reservoir_subsample_list(rows: list[dict], max_rows: int, seed: int) -> list[dict]:
    """메모리에 있는 행 목록에 reservoir 서브샘플."""
    import random

    rng = random.Random(seed)
    reservoir: list[dict] = []
    n_seen = 0
    for x in rows:
        n_seen += 1
        if len(reservoir) < max_rows:
            reservoir.append(x)
        else:
            j = rng.randint(1, n_seen)
            if j <= max_rows:
                reservoir[j - 1] = x
    return reservoir


_LAST_RESOLVE_META: dict[str, Any] = {}


def last_training_resolve_meta() -> dict[str, Any]:
    """직전 resolve_training_rows()의 풀·외부 데이터 요약(메타 기록용)."""
    return dict(_LAST_RESOLVE_META)


def load_dataset_reservoir(max_rows: int, seed: int = 42) -> list[dict]:
    """전체를 메모리에 올리지 않고 reservoir 샘플링으로 학습용 서브셋만 구축."""
    import random

    if max_rows <= 0 or not DATASET_PATH.exists():
        return []
    rng = random.Random(seed)
    reservoir: list[dict] = []
    n_seen = 0
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
                if not isinstance(x, dict) or "x" not in x or "y" not in x:
                    continue
                x = dict(x)
                x["x"] = pad_feature_vector([float(v) for v in x["x"]])
            except Exception:
                continue
            n_seen += 1
            if len(reservoir) < max_rows:
                reservoir.append(x)
            else:
                j = rng.randint(1, n_seen)
                if j <= max_rows:
                    reservoir[j - 1] = x
    return reservoir


def resolve_training_rows(
    *,
    reservoir_max: int,
    reservoir_seed: int,
) -> tuple[list[dict], int, int]:
    """(학습용 행, 풀 크기 추정치, 학습에 쓴 행 수). reservoir_max<=0 이면 로컬 전체 로드.

    TEAM_EXTERNAL_TRAIN_ENABLED=1 이면 URL/파일에서 JSONL을 추가로 합친 뒤(중복 제거) 학습한다.
    """
    global _LAST_RESOLVE_META
    from edu_tools.team_external_train import load_external_training_rows

    ext_rows, ext_meta = load_external_training_rows()
    n_local = count_dataset_lines()

    if reservoir_max > 0 and n_local > reservoir_max:
        data = load_dataset_reservoir(reservoir_max, reservoir_seed)
    else:
        data = load_dataset()

    ext_info: dict[str, Any] = {
        "enabled": bool(ext_meta.get("enabled")),
        "merged": False,
    }
    for k in ("source", "rows_loaded", "parse_skipped_lines", "error", "cache_hit", "url_host", "path", "cache_fetched_at"):
        if k in ext_meta:
            ext_info[k] = ext_meta[k]

    if ext_rows:
        local_len = len(data)
        combined = data + ext_rows
        data, dedup_drop = _dedupe_training_rows(combined)
        ext_info["merged"] = True
        ext_info["local_rows_before_merge"] = local_len
        ext_info["external_rows_attempted"] = len(ext_rows)
        ext_info["dedupe_dropped"] = dedup_drop
        ext_info["rows_after_merge"] = len(data)
        if reservoir_max > 0 and len(data) > reservoir_max:
            data = _reservoir_subsample_list(data, reservoir_max, reservoir_seed + 97_831)
            ext_info["reservoir_after_merge"] = True
    else:
        ext_info["merged"] = bool(ext_meta.get("enabled")) and not ext_meta.get("error")

    n_ext_pool = int(ext_meta.get("rows_loaded") or 0) if ext_info["enabled"] else 0
    if ext_meta.get("error"):
        n_ext_pool = 0
    n_pool = n_local + n_ext_pool
    n_used = len(data)
    _LAST_RESOLVE_META = {
        "n_pool_local_lines": n_local,
        "n_pool_reported": n_pool,
        "external_train": ext_info,
    }
    return data, n_pool, n_used


def _row_created_ts(row: dict) -> float:
    s = row.get("created_at") or ""
    if not s or not isinstance(s, str):
        return 0.0
    try:
        ts = s.replace("Z", "+00:00") if s.endswith("Z") else s
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


def sort_rows_by_created_at(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=_row_created_ts)


def filter_training_rows_outliers(rows: list[dict], iqr_k: float) -> tuple[list[dict], dict[str, Any]]:
    """라벨 y 이상치(IQR 밖) 제거. TEAM_TRAIN_OUTLIER_IQR_K>0 일 때만."""
    if iqr_k <= 0 or len(rows) < 8:
        return rows, {"enabled": False, "skipped": 0}
    ys: list[float] = []
    for r in rows:
        try:
            ys.append(float(r.get("y", 0)))
        except Exception:
            continue
    if len(ys) < 8:
        return rows, {"enabled": False, "skipped": 0}
    ys_sorted = sorted(ys)
    n = len(ys_sorted)
    q1 = ys_sorted[n // 4]
    q3 = ys_sorted[(3 * n) // 4]
    iqr = max(1e-6, q3 - q1)
    lo = q1 - iqr_k * iqr
    hi = q3 + iqr_k * iqr
    out: list[dict] = []
    skipped = 0
    for r in rows:
        try:
            y = float(r.get("y", 0))
        except Exception:
            skipped += 1
            continue
        if y < lo or y > hi:
            skipped += 1
            continue
        out.append(r)
    if len(out) < max(MIN_TRAIN_SAMPLES // 2, 12):
        return rows, {
            "enabled": False,
            "skipped": 0,
            "note_ko": "이상치 제거 시 학습 행이 너무 적어 원본 유지",
        }
    return out, {
        "enabled": True,
        "skipped": skipped,
        "y_bounds": [round(lo, 4), round(hi, 4)],
        "iqr_k": iqr_k,
    }


def time_holdout_split(
    rows: list[dict], frac: float, *, min_hold: int = 5
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """시간순 정렬 후 최근 frac 비율을 홀드아웃(학습에서 제외)."""
    rows = sort_rows_by_created_at(rows)
    n = len(rows)
    if frac <= 0:
        return rows, [], {"active": False, "reason": "frac_zero"}
    if n < MIN_TRAIN_SAMPLES + min_hold:
        return rows, [], {"active": False, "reason": "pool_small"}
    k = max(min_hold, int(n * frac))
    k = min(k, n - MIN_TRAIN_SAMPLES)
    if k < min_hold or n - k < MIN_TRAIN_SAMPLES:
        return rows, [], {"active": False, "reason": "train_below_min"}
    return rows[:-k], rows[-k:], {
        "active": True,
        "holdout_rows": k,
        "frac_requested": frac,
        "frac_effective": round(k / max(n, 1), 4),
    }


def load_model() -> TeamMlModel | None:
    if not MODEL_PATH.exists():
        return None
    try:
        raw = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        if (
            len(raw.get("means", [])) != FEATURE_DIM
            or len(raw.get("stds", [])) != FEATURE_DIM
            or len(raw.get("weights", [])) != FEATURE_DIM
        ):
            return None
        return TeamMlModel(
            version=str(raw["version"]),
            trained_at=str(raw["trained_at"]),
            sample_count=int(raw["sample_count"]),
            means=[float(v) for v in raw["means"]],
            stds=[float(v) for v in raw["stds"]],
            weights=[float(v) for v in raw["weights"]],
            bias=float(raw["bias"]),
        )
    except Exception:
        return None


def _save_model(m: TeamMlModel) -> None:
    _ensure_data_dir()
    MODEL_PATH.write_text(
        json.dumps(
            {
                "version": m.version,
                "trained_at": m.trained_at,
                "sample_count": m.sample_count,
                "means": m.means,
                "stds": m.stds,
                "weights": m.weights,
                "bias": m.bias,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _z(x: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(x[i] - means[i]) / max(stds[i], 1e-6) for i in range(len(x))]


def train_if_needed() -> dict:
    model = load_model()
    trained_count = model.sample_count if model else 0
    rmax = int(os.environ.get("TEAM_TRAIN_RESERVOIR_MAX", "0") or 0)
    rseed = int(os.environ.get("TEAM_TRAIN_RESERVOIR_SEED", "42") or 42)
    data, n_pool, n_used = resolve_training_rows(reservoir_max=rmax, reservoir_seed=rseed)
    data = sort_rows_by_created_at(data)
    iqr_k = float(os.environ.get("TEAM_TRAIN_OUTLIER_IQR_K", "0") or 0)
    data, _ = filter_training_rows_outliers(data, iqr_k)
    n = len(data)

    if n_pool < MIN_TRAIN_SAMPLES or n < MIN_TRAIN_SAMPLES:
        return {
            "enabled": model is not None,
            "trained": False,
            "sample_count": n_pool,
            "reason": "insufficient_samples",
        }
    if model and (n_pool - trained_count) < RETRAIN_INTERVAL and not retrain_forced_by_env():
        return {
            "enabled": True,
            "trained": False,
            "sample_count": n_pool,
            "model_version": model.version,
            "trained_at": model.trained_at,
            "reason": "interval_not_reached",
        }

    xs = [pad_feature_vector([float(v) for v in r["x"]]) for r in data]
    ys = [max(0.0, min(100.0, float(r["y"]))) for r in data]
    d = len(xs[0])
    means = [sum(row[i] for row in xs) / n for i in range(d)]
    stds = []
    for i in range(d):
        var = sum((row[i] - means[i]) ** 2 for row in xs) / n
        stds.append(math.sqrt(var) if var > 1e-10 else 1.0)
    xz = [_z(row, means, stds) for row in xs]

    w = [0.0] * d
    b = 50.0
    lr = 0.01
    epochs = 220
    for _ in range(epochs):
        for i in range(n):
            pred = b + sum(w[j] * xz[i][j] for j in range(d))
            err = pred - ys[i]
            b -= lr * err
            for j in range(d):
                w[j] -= lr * err * xz[i][j]

    m = TeamMlModel(
        version=f"team-ml-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        trained_at=_now_iso(),
        sample_count=n_pool,
        means=means,
        stds=stds,
        weights=w,
        bias=b,
    )
    _save_model(m)
    return {
        "enabled": True,
        "trained": True,
        "sample_count": n_pool,
        "training_rows_used": n_used,
        "model_version": m.version,
        "trained_at": m.trained_at,
    }


def predict_score(model: TeamMlModel, x: list[float]) -> float:
    x = pad_feature_vector(x)
    zx = _z(x, model.means, model.stds)
    y = model.bias + sum(model.weights[i] * zx[i] for i in range(len(zx)))
    return max(0.0, min(100.0, y))


"""HTTPS(또는 로컬 파일)에서 team 학습용 JSONL을 불러와 캐시한다.

각 줄은 {"x": [...], "y": <0~100>, ...} 형식이어야 하며 team_ml_dataset.jsonl과 동일 스키마다.
환경변수 TEAM_EXTERNAL_TRAIN_ENABLED=1 일 때만 동작한다.

라이선스·개인정보: 공개 허용된 URL만 사용하고, 수집 전 약관을 확인할 것.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from edu_tools.team_ml_model import DATA_DIR, pad_feature_vector

CACHE_JSONL = DATA_DIR / "team_external_train.cache.jsonl"
CACHE_META = DATA_DIR / "team_external_train.cache.meta.json"


def _external_enabled() -> bool:
    return (os.environ.get("TEAM_EXTERNAL_TRAIN_ENABLED", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _allow_http() -> bool:
    return (os.environ.get("TEAM_EXTERNAL_TRAIN_ALLOW_HTTP", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or str(default)).strip())
    except Exception:
        return default


def _parse_jsonl_block(text: str, *, max_rows: int) -> tuple[list[dict[str, Any]], int]:
    """JSONL 텍스트에서 유효 행만 파싱. (rows, parse_errors) 카운트는 생략하고 스킵만 집계."""
    out: list[dict[str, Any]] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(out) >= max_rows:
            break
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict) or "x" not in obj or "y" not in obj:
                skipped += 1
                continue
            row = dict(obj)
            row["x"] = pad_feature_vector([float(v) for v in row["x"]])
            out.append(row)
        except Exception:
            skipped += 1
            continue
    return out, skipped


def _read_cache_if_fresh() -> tuple[str | None, dict[str, Any]]:
    if not CACHE_JSONL.is_file():
        return None, {}
    try:
        meta_raw = json.loads(CACHE_META.read_text(encoding="utf-8")) if CACHE_META.is_file() else {}
    except Exception:
        meta_raw = {}
    if not isinstance(meta_raw, dict):
        meta_raw = {}
    ttl = max(0, _int_env("TEAM_EXTERNAL_TRAIN_CACHE_SEC", 3600))
    fetched_at = meta_raw.get("fetched_at")
    if ttl > 0 and isinstance(fetched_at, str):
        try:
            ts = datetime.fromisoformat(fetched_at.replace("Z", "+00:00")).timestamp()
            age = datetime.now(timezone.utc).timestamp() - ts
            if age >= 0 and age < ttl:
                return CACHE_JSONL.read_text(encoding="utf-8"), meta_raw
        except Exception:
            pass
    return None, meta_raw


def _write_cache(body: str, *, source_url: str | None, from_file: str | None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_JSONL.write_text(body, encoding="utf-8")
    CACHE_META.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source_url": source_url,
                "from_file": from_file,
                "bytes": len(body.encode("utf-8")),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _fetch_url(url: str) -> tuple[bytes | None, str | None]:
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        return None, "unsupported_scheme"
    if parsed.scheme == "http" and not _allow_http():
        return None, "http_not_allowed_set_TEAM_EXTERNAL_TRAIN_ALLOW_HTTP"
    max_b = max(4096, _int_env("TEAM_EXTERNAL_TRAIN_MAX_BYTES", 8_000_000))
    timeout = max(3, _int_env("TEAM_EXTERNAL_TRAIN_TIMEOUT_SEC", 30))
    req = Request(
        url,
        headers={
            "User-Agent": "stock-predict-team-external-train/1.0",
            "Accept": "application/x-ndjson, application/json, text/plain, */*",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            chunk = resp.read(max_b + 1)
    except HTTPError as e:
        return None, f"http_{e.code}"
    except URLError as e:
        return None, f"url_error:{e.reason!r}"
    except Exception as e:
        return None, str(e)[:200]
    if len(chunk) > max_b:
        return None, "response_too_large"
    return chunk, None


def load_external_training_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """비활성화 시 ([], meta). 활성화 시 파싱된 행과 수집 메타."""
    meta: dict[str, Any] = {
        "enabled": _external_enabled(),
        "source": None,
        "rows_loaded": 0,
        "parse_skipped_lines": 0,
        "error": None,
        "cache_hit": False,
        "url_host": None,
    }
    if not meta["enabled"]:
        return [], meta

    max_rows = max(1, min(_int_env("TEAM_EXTERNAL_TRAIN_MAX_ROWS", 8000), 100_000))
    file_path = (os.environ.get("TEAM_EXTERNAL_TRAIN_FILE") or "").strip()
    url = (os.environ.get("TEAM_EXTERNAL_TRAIN_URL") or "").strip()

    text: str | None = None
    if file_path:
        p = Path(file_path)
        if not p.is_file():
            meta["error"] = "file_not_found"
            meta["source"] = "file"
            return [], meta
        try:
            text = p.read_text(encoding="utf-8")
            meta["source"] = "file"
            meta["path"] = str(p.resolve())
        except Exception as e:
            meta["error"] = f"read_error:{e!s}"[:200]
            meta["source"] = "file"
            return [], meta
    elif url:
        meta["source"] = "url"
        try:
            host = urlparse(url).netloc or url
            meta["url_host"] = host[:200]
        except Exception:
            meta["url_host"] = None
        cached_text, cache_meta = _read_cache_if_fresh()
        if cached_text is not None:
            text = cached_text
            meta["cache_hit"] = True
            if isinstance(cache_meta.get("fetched_at"), str):
                meta["cache_fetched_at"] = cache_meta["fetched_at"]
        else:
            raw, err = _fetch_url(url)
            if err or raw is None:
                meta["error"] = err or "fetch_failed"
                return [], meta
            try:
                text = raw.decode("utf-8")
            except Exception:
                meta["error"] = "utf8_decode"
                return [], meta
            _write_cache(text, source_url=url, from_file=None)
    else:
        meta["error"] = "set_TEAM_EXTERNAL_TRAIN_URL_or_TEAM_EXTERNAL_TRAIN_FILE"
        return [], meta

    assert text is not None
    rows, skipped = _parse_jsonl_block(text, max_rows=max_rows)
    meta["rows_loaded"] = len(rows)
    meta["parse_skipped_lines"] = skipped
    return rows, meta

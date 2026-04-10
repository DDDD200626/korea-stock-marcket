"""API 스모크·회귀 테스트 — CI에서 기술적 완성도 검증."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from learning_analysis.main import app

client = TestClient(app)


def test_health_ok_and_request_id_header() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "providers" in body
    assert body.get("version")
    assert body.get("app_name")
    assert body.get("openapi_docs") == "/docs"
    assert r.headers.get("X-Request-ID")
    assert r.headers.get("X-Process-Time-Ms")


def test_version_endpoint() -> None:
    r = client.get("/api/version")
    assert r.status_code == 200
    data = r.json()
    assert data["name"]
    assert data["version"]
    assert "/docs" in data.get("openapi_docs", "")


def test_perf_recent() -> None:
    client.get("/api/health")
    r = client.get("/api/perf/recent")
    assert r.status_code == 200
    data = r.json()
    assert "recent" in data
    assert isinstance(data["recent"], list)
    assert data.get("ring_buffer_enabled") is True
    assert r.headers.get("Cache-Control", "").startswith("no-store")


def test_analyze_includes_perf_breakdown() -> None:
    body = {
        "course_name": "테스트",
        "student_or_group_label": "익명",
        "learning": {},
        "exam": {},
        "context_for_educator": "",
    }
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "perf" in data and data["perf"]
    assert "llm_parallel_ms" in data["perf"]
    assert "local_ms" in data["perf"]
    assert "total_ms" in data["perf"]


def test_observability_ready_live() -> None:
    r = client.get("/api/observability")
    assert r.status_code == 200
    data = r.json()
    assert data.get("service")
    assert data.get("version")
    assert "uptime_seconds" in data
    assert data.get("environment")
    assert r.headers.get("Cache-Control", "").startswith("no-store")
    r2 = client.get("/api/ready")
    assert r2.status_code == 200
    assert r2.json().get("status") == "ready"
    r3 = client.get("/api/live")
    assert r3.status_code == 200
    assert r3.json().get("status") == "live"


def test_capabilities_contest_rubric() -> None:
    r = client.get("/api/capabilities")
    assert r.status_code == 200
    data = r.json()
    assert data["version"]
    rub = data["rubric"]
    assert "technical_completeness" in rub
    assert "ai_efficiency" in rub
    assert "planning_practical" in rub
    assert "creativity" in rub
    assert "deep_learning_assist" in rub
    assert data.get("dl_roadmap", {}).get("phase_count") == 10
    assert data.get("dl_roadmap", {}).get("version") == 5
    assert data.get("dl_roadmap", {}).get("scope_note_ko")
    csp = data.get("contest_submission_pack") or {}
    assert csp.get("verification_order")
    assert csp.get("json_paths", {}).get("dl_quality_unified")
    assert data["endpoints"]["team_evaluate"] == "POST /api/team/evaluate"
    assert data["endpoints"].get("observability") == "GET /api/observability"
    assert data["endpoints"].get("ready") == "GET /api/ready"
    assert data["endpoints"].get("perf_recent") == "GET /api/perf/recent"
    assert data["endpoints"].get("team_dataset_label_summary") == "GET /api/team/data/dataset-label-summary"
    assert r.headers.get("X-Request-ID")


def test_dataset_label_summary_ok() -> None:
    r = client.get("/api/team/data/dataset-label-summary")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "lines_scanned" in (body.get("stats") or {})


def test_dataset_label_summary_schema_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    """내부 stats 계산기가 일부 키를 누락해도 API 응답 스키마는 유지되어야 한다."""
    monkeypatch.setattr(
        "learning_analysis.main.dataset_label_streaming_stats",
        lambda: {"error": "forced_missing_keys"},
    )
    r = client.get("/api/team/data/dataset-label-summary")
    assert r.status_code == 200
    body = r.json()
    stats = body.get("stats") or {}
    assert body.get("status") == "ok"
    assert "lines_scanned" in stats
    assert "y_present" in stats
    assert "dataset_path" in stats


def test_rubric_draft_heuristic_returns_criteria() -> None:
    r = client.post(
        "/api/rubric/draft",
        json={
            "learning_objectives": "REST API를 설계하고 문서화할 수 있다.\n보안·오류 처리를 적용한다.",
            "course_name": "웹서비스",
            "assignment_type": "팀 프로젝트",
            "max_criteria": 4,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("mode") in ("heuristic", "ai")
    assert len(data.get("criteria") or []) >= 3
    assert data.get("rubric_markdown")
    assert "disclaimer" in data


def test_team_evaluate_heuristic_has_creative_insights() -> None:
    body = {
        "project_name": "테스트 프로젝트",
        "project_description": "",
        "evaluation_criteria": "",
        "members": [
            {
                "name": "김개발",
                "role": "백엔드",
                "commits": 12,
                "pull_requests": 3,
                "lines_changed": 400,
                "tasks_completed": 4,
                "meetings_attended": 3,
                "self_report": "구현과 리뷰를 맡았습니다.",
                "peer_notes": "",
                "timeline": [],
            },
            {
                "name": "이문서",
                "role": "기획",
                "commits": 2,
                "pull_requests": 1,
                "lines_changed": 50,
                "tasks_completed": 5,
                "meetings_attended": 4,
                "self_report": "문서와 회의 진행.",
                "peer_notes": "",
                "timeline": [],
            },
        ],
        "collaboration_edges": [],
    }
    r = client.post("/api/team/evaluate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert len(data["members"]) == 2
    assert data.get("freerider_detection_overview")
    assert "product_tagline_ko" in data
    assert isinstance(data.get("team_dashboard"), list)
    assert len(data["team_dashboard"]) == 2
    assert "member_name" in data["team_dashboard"][0]
    assert "freerider_detection" in data["members"][0]
    fd0 = data["members"][0]["freerider_detection"]
    assert "basic_low_contribution" in fd0
    assert "rule_metrics" in fd0
    assert fd0["rule_metrics"] is not None
    assert "rubric_report" in data
    assert data["rubric_report"]["members"]
    assert "evaluation_trust" in data
    assert data["evaluation_trust"]["level_ko"]
    assert "dl_model_info" in data and isinstance(data["dl_model_info"], dict)
    assert "members" in data and "dl_score" in data["members"][0]
    assert "team_risk" in data
    assert "improvement_chain" in data
    assert data["improvement_chain"]["items"]
    assert "creative_insights" in data
    ci = data["creative_insights"]
    assert ci["reflection_kit"]["team_storyline"]
    assert len(ci["explain_facts"]) == 2
    assert "team_health_score" in ci
    assert 0 <= ci["team_health_score"] <= 100
    assert ci.get("team_health_hint")
    assert "practical_toolkit" in data
    pt = data["practical_toolkit"]
    assert len(pt["teacher_checklist"]) >= 3
    assert "request_id" in data
    assert r.headers.get("X-Request-ID")


def test_team_evaluate_compare_no_api_keys_returns_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """키가 없을 때도 비교 엔드포인트는 200 + 모델별 오류 메시지로 응답."""
    for k in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    body = {
        "project_name": "비교 테스트",
        "project_description": "",
        "evaluation_criteria": "",
        "members": [
            {
                "name": "A",
                "role": "",
                "commits": 5,
                "pull_requests": 1,
                "lines_changed": 100,
                "tasks_completed": 2,
                "meetings_attended": 2,
                "self_report": "",
                "peer_notes": "",
                "timeline": [],
            },
        ],
        "collaboration_edges": [],
    }
    r = client.post("/api/team/evaluate/compare", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) >= 1
    assert data.get("comparison_summary")
    assert data.get("product_mode") == "ai_multi_eval"
    assert len(data.get("pipeline_steps") or []) == 6
    assert data.get("divergence") is None
    assert data.get("trust_scores") is None
    assert data.get("explainability") == []
    assert "request_id" in data
    assert r.headers.get("X-Request-ID")


def test_team_report_unified_pipeline() -> None:
    """Score Engine → Anomaly → Analysis 구조 분리."""
    r = client.post(
        "/api/team/report",
        json={
            "project_name": "통합 리포트 테스트",
            "teamData": [
                {
                    "name": "A",
                    "commits": 20,
                    "prs": 4,
                    "lines": 800,
                    "attendance": 90,
                    "selfReport": "백엔드 API",
                },
                {
                    "name": "B",
                    "commits": 5,
                    "prs": 1,
                    "lines": 100,
                    "attendance": 70,
                    "selfReport": "문서",
                },
            ],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["scores"]) == 2
    assert len(data["anomalies"]) == 2
    assert len(data["analysis"]) == 2
    assert "trust_scores" in data and data["trust_scores"]
    assert "evaluation_log" in data
    assert data["evaluation_log"].get("request_id")
    assert data["evaluation_log"].get("input_hash")
    s0 = data["scores"][0]
    assert "rawScore" in s0 and "normalizedScore" in s0 and "rank" in s0
    assert "breakdown" in s0
    assert r.headers.get("X-Request-ID")
    dl = data.get("dl_model_info") or {}
    assert "contest_transparency" in dl
    ct = dl["contest_transparency"]
    assert ct.get("blend_formula")
    assert "limitations_ko" in ct and isinstance(ct["limitations_ko"], list)
    assert ct.get("rubric_alignment_ko")
    q = dl.get("quality") or {}
    assert "feature_drift" in q
    fd = q.get("feature_drift") or {}
    assert "drift_level" in fd
    assert "promotion_gate" in q
    assert "operations_playbook" in q
    op = q.get("operations_playbook") or {}
    assert op.get("recommendation_level") in ("ok", "watch", "action")
    assert isinstance(op.get("checklist_ko"), list)
    assert "ai_meta" in data and isinstance(data["ai_meta"], dict)
    assert "team_narrative" in data
    assert isinstance(data.get("team_narrative"), str)

import "./style.css";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/** 백엔드 `/api/health` — 한 번의 요청으로 버전·제공자 확인 */
interface BackendHealth {
  providers?: ProviderKeys;
  version?: string;
  app_name?: string;
  openapi_docs?: string;
}

let lastHealthFetchedAt = 0;
let lastHealthAttemptAt = 0;
let lastHealthLatencyMs: number | null = null;
let lastHealthCheckedAt = 0;
/** 마지막으로 health OK였던 시각 — 일시 실패 시 UI를 바로 OFFLINE로 두지 않음 */
let lastSuccessfulHealthAt = 0;
/** health 요청은 실패했지만 2분 이내 성공 이력이 있어 마지막 스냅샷 유지 */
let connectionUnstable = false;
const HEALTH_CACHE_MS = 5000;
const HEALTH_RETRY_MS = 2500;
const HEALTH_STICKY_MS = 120_000;
const HEALTH_FETCH_ATTEMPTS = 6;
const HEALTH_ATTEMPT_TIMEOUT_MS = 45_000;
/** 백엔드 미연결·불안정 시 짧은 주기로 /api/health 재시도 */
let healthPollHandle: number | null = null;
/** 정상일 때도 주기 핑으로 프록시·탭 유휴 끊김 완화 */
let healthKeepAliveHandle: number | null = null;

function clearHealthPoll(): void {
  if (healthPollHandle != null) {
    window.clearInterval(healthPollHandle);
    healthPollHandle = null;
  }
}

function ensureHealthPollIfNeeded(): void {
  const need = !state.health || connectionUnstable;
  if (!need) {
    clearHealthPoll();
    return;
  }
  clearHealthPoll();
  const everyMs = connectionUnstable ? 2500 : 5000;
  healthPollHandle = window.setInterval(() => {
    void refreshHealth(true).then(() => renderSync());
  }, everyMs);
}

function startHealthKeepAlive(): void {
  if (healthKeepAliveHandle != null) return;
  healthKeepAliveHandle = window.setInterval(() => {
    if (document.visibilityState !== "visible") return;
    void refreshHealth(true).then(() => renderSync());
  }, 30_000);
}

function sleepMs(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** GET 기본 12s / POST·LLM 180s — 무한 대기 방지 */
function apiFetch(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<Response> {
  const url = path.startsWith("http") ? path : apiUrl(path);
  const timeoutMs =
    init?.timeoutMs ?? (!init?.method || init.method === "GET" ? 12_000 : 180_000);
  const ctrl = new AbortController();
  const tid = window.setTimeout(() => ctrl.abort(), timeoutMs);
  const { timeoutMs: _ignore, ...rest } = init ?? {};
  return fetch(url, { ...rest, signal: ctrl.signal }).finally(() => {
    clearTimeout(tid);
  });
}

type SiteView =
  | "home"
  | "hub"
  | "analyze"
  | "team"
  | "at-risk"
  | "feedback"
  | "rubric";

interface ProviderKeys {
  gemini?: boolean;
  openai?: boolean;
  claude?: boolean;
  grok?: boolean;
}

interface ModelJudgment {
  provider: string;
  model_label: string;
  ok: boolean;
  error?: string | null;
  cheating_likelihood?: number | null;
  learning_state_summary: string;
  mismatch_analysis: string;
  future_prediction: string;
  confidence_note: string;
}

interface AnalyzeResponse {
  providers_used: string[];
  providers_skipped: string[];
  judgments: ModelJudgment[];
  consensus_cheating_avg: number | null;
  consensus_summary: string;
  disclaimer: string;
}

interface TeamMemberRow {
  name: string;
  role: string;
  commits: string;
  pull_requests: string;
  lines_changed: string;
  tasks_completed: string;
  meetings_attended: string;
  /** 프로젝트·동료 평가 등 결과 점수(선택, 기여 추정과 비교) */
  outcome_score: string;
  self_report: string;
  peer_notes: string;
  /** 출석·참여 % (0–100), 비우면 회의 출석 횟수로 추정 */
  attendance: string;
  /** 주차별 활동(선택): 입력 시 타임라인에 반영 */
  timeline: { period_label: string; activity_score: string }[];
}

interface TeamDimensionScores {
  technical: number;
  collaboration: number;
  initiative: number;
}

interface TimelinePointOut {
  period_label: string;
  share_percent: number;
  activity_score?: number | null;
}

interface FreeriderRuleMetrics {
  activity_score?: number;
  collaboration_score?: number;
  consistency_score?: number;
  blended_score?: number;
  final_score?: number;
  penalty_applied?: boolean;
  activity_risk?: boolean;
  collaboration_risk?: boolean;
  consistency_risk?: boolean;
  interaction_risk?: boolean;
  rule_conditions_met?: number;
  risk_level?: string;
  grade_ko?: string;
  analysis_lines?: string[];
}

interface FreeriderDetectionReport {
  basic_low_contribution?: boolean;
  basic_reasons?: string[];
  advanced_pattern_flags?: string[];
  advanced_pattern_score?: number;
  collaboration_isolated?: boolean;
  collaboration_isolation_reasons?: string[];
  below_team_average?: boolean;
  team_mean_contribution?: number;
  delta_vs_team_mean?: number;
  ai_detection_summary?: string;
  rule_metrics?: FreeriderRuleMetrics;
}

interface MemberDashboardRow {
  member_name: string;
  rank: number;
  final_rule_score: number;
  bar_fill_percent: number;
  grade_ko: string;
  rule_conditions_met: number;
  risk_level: string;
  suspected_highlight: boolean;
}

interface TeamMemberOut {
  name: string;
  role?: string;
  contribution_index: number;
  dimensions: TeamDimensionScores;
  evidence_summary?: string;
  caveats?: string;
  free_rider_suspected?: boolean;
  free_rider_risk?: number;
  free_rider_signals?: string[];
  ai_feedback?: string;
  timeline?: TimelinePointOut[];
  contribution_type_label?: string;
  role_scores?: Record<string, number>;
  freerider_detection?: FreeriderDetectionReport;
}

interface NetworkNode {
  id: string;
  label: string;
  x: number;
  y: number;
  contribution_index: number;
}

interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
}

interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

interface MismatchItem {
  member_name: string;
  contribution_index: number;
  outcome_score?: number | null;
  gap?: number | null;
  note?: string;
  severity?: string;
}

interface AnomalyAlert {
  member_name: string;
  code: string;
  severity?: string;
  message?: string;
}

interface MemberExplainFact {
  member_name: string;
  facts: string[];
}

interface TeamRoleBalance {
  dev: number;
  doc: number;
  leader: number;
  supporter: number;
  balance_hint: string;
}

interface ReflectionKit {
  team_storyline: string;
  teacher_questions: string[];
  encouragement_line: string;
}

interface CreativeInsights {
  explain_facts: MemberExplainFact[];
  team_role_balance: TeamRoleBalance;
  reflection_kit: ReflectionKit;
  team_health_score?: number;
  team_health_hint?: string;
}

interface PracticalToolkit {
  teacher_checklist: string[];
  checklist_note?: string;
}

interface RubricMemberRow {
  member_name: string;
  contribution: number;
  collaboration: number;
  persistence: number;
  problem_solving: number;
}

interface RubricReport {
  team_average?: RubricMemberRow;
  members?: RubricMemberRow[];
  criteria_note?: string;
  ai_explanation?: string;
}

interface EvaluationTrustBlock {
  score_0_100?: number;
  level_ko?: string;
  factors?: string[];
  short_note?: string;
}

interface TeamRiskBlock {
  flags?: string[];
  summary_ko?: string;
  ai_team_risk?: string;
}

interface ImprovementChainItem {
  problem?: string;
  explanation?: string;
  suggestion?: string;
  predicted_outcome?: string;
}

interface ImprovementChainBlock {
  headline?: string;
  items?: ImprovementChainItem[];
}

interface TeamEvaluateResponse {
  mode: string;
  members: TeamMemberOut[];
  fairness_notes?: string;
  free_rider_summary?: string;
  freerider_detection_overview?: string;
  product_tagline_ko?: string;
  team_dashboard?: MemberDashboardRow[];
  rubric_report?: RubricReport;
  evaluation_trust?: EvaluationTrustBlock;
  team_risk?: TeamRiskBlock;
  improvement_chain?: ImprovementChainBlock;
  collaboration_network?: NetworkGraph;
  contribution_outcome_summary?: string;
  mismatches?: MismatchItem[];
  anomaly_alerts?: AnomalyAlert[];
  advanced_mode?: string;
  creative_insights?: CreativeInsights;
  practical_toolkit?: PracticalToolkit;
  request_id?: string;
  generated_at?: string;
  processing_ms?: number;
  disclaimer?: string;
}

/** POST /api/team/report — Score Engine → Anomaly → AI(설명) */
interface TeamUnifiedScoreBreakdown {
  commits: number;
  prs: number;
  lines: number;
  attendance: number;
  self_report: number;
}

interface TeamUnifiedScoreResult {
  member_name: string;
  rawScore: number;
  normalizedScore: number;
  rank: number;
  top_percent: number;
  breakdown: TeamUnifiedScoreBreakdown;
  weighted_points: TeamUnifiedScoreBreakdown;
  pct_vs_team_mean: number;
  data_reliability: {
    score_0_100: number;
    git_ratio_percent: number;
    self_report_ratio_percent: number;
    note?: string;
  };
  dl_score?: number;
  dl_confidence?: number;
  blendedScore?: number;
  dl_top_factors?: string[];
}

interface TeamUnifiedAnomaly {
  member_name: string;
  flags: string[];
}

interface TeamUnifiedAnalysis {
  member_name: string;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  position_in_team: string;
  recommended_actions: string[];
  metric_dl_note?: string;
}

interface TeamUnifiedReport {
  scores: TeamUnifiedScoreResult[];
  anomalies: TeamUnifiedAnomaly[];
  analysis: TeamUnifiedAnalysis[];
  edge_cases: string[];
  trust_scores: Record<string, number>;
  dl_model_info?: Record<string, unknown>;
  evaluation_log: Record<string, unknown>;
  disclaimer: string;
  team_narrative?: string;
  ai_meta?: Record<string, unknown>;
}

interface PromotionGateInfo {
  accepted?: boolean;
  reasons?: string[];
}

interface TeamTrendPoint {
  date: string;
  rule_score: number;
  dl_score: number;
  blended_score: number;
  samples: number;
}

interface TeamTrendsPayload {
  window_days: number;
  member_filter?: string | null;
  members: string[];
  series: Record<string, TeamTrendPoint[]>;
}

interface TeamModelMonitorPayload {
  window_days: number;
  samples: number;
  rule_avg: number;
  dl_avg: number;
  blended_avg: number;
  rule_dl_drift: number;
  recent_auto_retrain_count: number;
}

interface JudgeCriteriaScore {
  technical: number;
  ai_efficiency: number;
  planning_practicality: number;
  creativity: number;
  overall: number;
}

interface WhatIfInput {
  member_name: string;
  add_commits: number;
  add_prs: number;
  add_lines: number;
}

interface TeamRecheckResult {
  runs: number;
  mean_overall: number;
  spread_overall: number;
  stability_label: "HIGH" | "MEDIUM" | "LOW";
  notes: string[];
}

interface AtRiskResponse {
  mode: string;
  dropout_risk: number;
  trend_summary: string;
  signals: string[];
  intervention_suggestions: string;
  disclaimer?: string;
}

interface FeedbackResponse {
  mode: string;
  draft_feedback: string;
  strengths: string[];
  improvements: string[];
  disclaimer?: string;
}

interface RubricAlignResponse {
  mode: string;
  alignment_score: number;
  matched_rubric_points: string[];
  gaps: string[];
  suggestions: string;
  disclaimer?: string;
}

interface RubricCriterionGen {
  name: string;
  description: string;
  weight_percent: number;
}

interface RubricGenerateResponse {
  mode: string;
  rubric_markdown: string;
  criteria: RubricCriterionGen[];
  disclaimer?: string;
}

function emptyTimelineRow(): { period_label: string; activity_score: string } {
  return { period_label: "", activity_score: "" };
}

function emptyTeamMember(): TeamMemberRow {
  return {
    name: "",
    role: "",
    commits: "",
    pull_requests: "",
    lines_changed: "",
    tasks_completed: "",
    meetings_attended: "",
    outcome_score: "",
    self_report: "",
    peer_notes: "",
    attendance: "",
    timeline: [
      emptyTimelineRow(),
      emptyTimelineRow(),
      emptyTimelineRow(),
      emptyTimelineRow(),
    ],
  };
}

/** 데모용 샘플 팀 데이터 */
function applyDemoTeamData(): void {
  state.team.project_name = "데모: 풀스택 팀 프로젝트";
  state.team.project_description = "공모전·시연용 샘플입니다. 실제 과제와 무관합니다.";
  state.team.evaluation_criteria = "";
  state.team.collaboration_edges_json = "";
  const a = emptyTeamMember();
  a.name = "김팀장";
  a.role = "리더·백엔드";
  a.commits = "42";
  a.pull_requests = "8";
  a.lines_changed = "2400";
  a.tasks_completed = "7";
  a.meetings_attended = "6";
  a.outcome_score = "88";
  a.self_report = "API 설계·코드 리뷰·스프린트 회의 진행을 맡았습니다.";
  a.peer_notes = "리뷰가 빠릅니다.";
  const b = emptyTeamMember();
  b.name = "이프론트";
  b.role = "프론트엔드";
  b.commits = "35";
  b.pull_requests = "10";
  b.lines_changed = "1800";
  b.tasks_completed = "7";
  b.meetings_attended = "6";
  b.outcome_score = "85";
  b.self_report = "UI 구현, 접근성 개선, 스토리북 작성.";
  b.peer_notes = "";
  const c = emptyTeamMember();
  c.name = "박문서";
  c.role = "문서·기획";
  c.commits = "8";
  c.pull_requests = "3";
  c.lines_changed = "240";
  c.tasks_completed = "9";
  c.meetings_attended = "7";
  c.outcome_score = "82";
  c.self_report = "요구사항 정리·회의록·사용자 가이드 작성.";
  c.peer_notes = "";
  a.attendance = "85";
  b.attendance = "80";
  c.attendance = "88";
  state.team.members = [a, b, c];
  state.team.report = null;
  state.team.error = null;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function downloadJsonExport(filenameBase: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" });
  const safe = filenameBase.replace(/[\\/:*?"<>|]/g, "_").slice(0, 80);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${safe}-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

const state: {
  view: SiteView;
  course_name: string;
  student_or_group_label: string;
  weekly_study_hours: string;
  lms_video_watch_ratio: string;
  quiz_average: string;
  assignment_on_time_ratio: string;
  discussion_count: string;
  attendance_ratio: string;
  learning_notes: string;
  midterm_score: string;
  final_score: string;
  exam_time_anomaly_note: string;
  exam_notes: string;
  context_for_educator: string;
  result: AnalyzeResponse | null;
  loading: boolean;
  error: string | null;
  health: BackendHealth | null;
  team: {
    project_name: string;
    project_description: string;
    evaluation_criteria: string;
    members: TeamMemberRow[];
    /** JSON 배열: [{"source":"이름","target":"이름","weight":0-100}] */
    collaboration_edges_json: string;
    report: TeamUnifiedReport | null;
    trends: TeamTrendsPayload | null;
    trends_days: number;
    trends_member: string;
    trends_loading: boolean;
    model_monitor: TeamModelMonitorPayload | null;
    what_if: WhatIfInput;
    recheck_loading: boolean;
    recheck_result: TeamRecheckResult | null;
    loading: boolean;
    error: string | null;
  };
  atRisk: {
    course_name: string;
    student_label: string;
    /** 출석·출첵 비율 % (0–100) */
    attendance_ratio: string;
    /** 과제 제출 비율 % (0–100) */
    assignment_ratio: string;
    notes: string;
    result: AtRiskResponse | null;
    loading: boolean;
    error: string | null;
  };
  feedback: {
    rubric: string;
    assignment_prompt: string;
    submission: string;
    result: FeedbackResponse | null;
    loading: boolean;
    error: string | null;
  };
  rubricAlign: {
    rubric: string;
    grader_rationale: string;
    student_work: string;
    result: RubricAlignResponse | null;
    loading: boolean;
    error: string | null;
  };
  rubricGen: {
    course_name: string;
    learning_objectives: string;
    assignment_type: string;
    max_criteria: string;
    result: RubricGenerateResponse | null;
    loading: boolean;
    error: string | null;
  };
} = {
  view: "home",
  course_name: "",
  student_or_group_label: "",
  weekly_study_hours: "",
  lms_video_watch_ratio: "",
  quiz_average: "",
  assignment_on_time_ratio: "",
  discussion_count: "",
  attendance_ratio: "",
  learning_notes: "",
  midterm_score: "",
  final_score: "",
  exam_time_anomaly_note: "",
  exam_notes: "",
  context_for_educator: "",
  result: null,
  loading: false,
  error: null,
  health: null,
  team: {
    project_name: "",
    project_description: "",
    evaluation_criteria: "",
    members: [emptyTeamMember(), emptyTeamMember()],
    collaboration_edges_json: "",
    report: null,
    trends: null,
    trends_days: 30,
    trends_member: "",
    trends_loading: false,
    model_monitor: null,
    what_if: { member_name: "", add_commits: 0, add_prs: 0, add_lines: 0 },
    recheck_loading: false,
    recheck_result: null,
    loading: false,
    error: null,
  },
  atRisk: {
    course_name: "",
    student_label: "",
    attendance_ratio: "",
    assignment_ratio: "",
    notes: "",
    result: null,
    loading: false,
    error: null,
  },
  feedback: {
    rubric: "",
    assignment_prompt: "",
    submission: "",
    result: null,
    loading: false,
    error: null,
  },
  rubricAlign: {
    rubric: "",
    grader_rationale: "",
    student_work: "",
    result: null,
    loading: false,
    error: null,
  },
  rubricGen: {
    course_name: "",
    learning_objectives: "",
    assignment_type: "",
    max_criteria: "5",
    result: null,
    loading: false,
    error: null,
  },
};

function parseOptFloat(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  const n = parseFloat(t);
  return Number.isFinite(n) ? n : null;
}

function parseOptInt(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  const n = parseInt(t, 10);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

const TEAM_DRAFT_KEY = "team_eval_draft_v2";
let teamDraftTimer: ReturnType<typeof setTimeout> | null = null;
function scheduleTeamDraftSave(): void {
  if (state.view !== "team") return;
  if (teamDraftTimer) clearTimeout(teamDraftTimer);
  teamDraftTimer = setTimeout(() => {
    teamDraftTimer = null;
    readTeamForm();
    try {
      const payload = {
        project_name: state.team.project_name,
        project_description: state.team.project_description,
        evaluation_criteria: state.team.evaluation_criteria,
        collaboration_edges_json: state.team.collaboration_edges_json,
        members: state.team.members,
      };
      localStorage.setItem(TEAM_DRAFT_KEY, JSON.stringify(payload));
    } catch {
      /* 저장소 불가 시 무시 */
    }
  }, 750);
}

function clearTeamDraft(): void {
  try {
    localStorage.removeItem(TEAM_DRAFT_KEY);
  } catch {
    /* ignore */
  }
}

function hydrateTeamDraftIfEmpty(): void {
  const hasInput =
    state.team.project_name.trim() !== "" ||
    state.team.project_description.trim() !== "" ||
    state.team.members.some((m) => m.name.trim() !== "");
  if (hasInput) return;
  try {
    const raw = localStorage.getItem(TEAM_DRAFT_KEY);
    if (!raw) return;
    const d = JSON.parse(raw) as {
      project_name?: string;
      project_description?: string;
      evaluation_criteria?: string;
      collaboration_edges_json?: string;
      members?: TeamMemberRow[];
    };
    if (typeof d.project_name === "string") state.team.project_name = d.project_name;
    if (typeof d.project_description === "string") state.team.project_description = d.project_description;
    if (typeof d.evaluation_criteria === "string") state.team.evaluation_criteria = d.evaluation_criteria;
    if (typeof d.collaboration_edges_json === "string") state.team.collaboration_edges_json = d.collaboration_edges_json;
    if (Array.isArray(d.members) && d.members.length > 0) {
      state.team.members = d.members.map((m) => ({
        ...emptyTeamMember(),
        ...m,
        timeline: m.timeline?.length ? m.timeline : emptyTeamMember().timeline,
      }));
    }
  } catch {
    /* ignore */
  }
}

function teamUnifiedExportSummaryText(res: TeamUnifiedReport): string {
  const lines: string[] = [];
  lines.push(`프로젝트: ${state.team.project_name || "(이름 없음)"}`);
  const log = res.evaluation_log as { request_id?: string; timestamp?: string };
  if (log.request_id) lines.push(`요청 ID: ${log.request_id}`);
  if (log.timestamp) lines.push(`시각: ${log.timestamp}`);
  lines.push("");
  res.scores.forEach((s) => {
    const n = res.analysis.find((x) => x.member_name === s.member_name);
    lines.push(`— ${s.member_name}`);
    lines.push(`  정규화 점수 ${s.normalizedScore.toFixed(0)} · 순위 ${s.rank}/${res.scores.length} · 평균 대비 ${s.pct_vs_team_mean >= 0 ? "+" : ""}${s.pct_vs_team_mean.toFixed(1)}%`);
    const t = res.trust_scores[s.member_name];
    if (t != null) lines.push(`  신뢰도 ${t.toFixed(0)}%`);
    if (n?.summary) lines.push(`  ${n.summary}`);
  });
  return lines.join("\n");
}

function clampScore(n: number): number {
  return Math.max(0, Math.min(100, n));
}

function computeJudgeCriteriaScore(res: TeamUnifiedReport): JudgeCriteriaScore {
  const avgNorm = res.scores.length
    ? res.scores.reduce((a, s) => a + s.normalizedScore, 0) / res.scores.length
    : 0;
  const trustVals = Object.values(res.trust_scores || {});
  const avgTrust = trustVals.length ? trustVals.reduce((a, x) => a + x, 0) / trustVals.length : 0;
  const anomalyCount = (res.anomalies || []).reduce((a, x) => a + (x.flags?.length || 0), 0);
  const explainChars = (res.analysis || []).reduce(
    (a, x) => a + (x.summary || "").length + (x.recommended_actions || []).join(" ").length,
    0
  );
  const explainDensity = res.analysis.length ? explainChars / res.analysis.length : 0;
  const spread =
    res.scores.length > 1
      ? Math.max(...res.scores.map((s) => s.normalizedScore)) - Math.min(...res.scores.map((s) => s.normalizedScore))
      : 0;

  const technical = clampScore(avgNorm * 0.72 + avgTrust * 0.28 - anomalyCount * 1.1);
  const ai_efficiency = clampScore(45 + Math.min(40, explainDensity / 6) + Math.min(15, avgTrust * 0.15));
  const planning_practicality = clampScore(55 + Math.min(25, (res.edge_cases?.length || 0) * 6) + Math.min(20, avgTrust * 0.2));
  const creativity = clampScore(50 + Math.min(20, spread * 0.35) + Math.min(30, explainDensity / 7));
  const overall = clampScore(technical * 0.35 + ai_efficiency * 0.25 + planning_practicality * 0.25 + creativity * 0.15);
  return {
    technical: Number(technical.toFixed(1)),
    ai_efficiency: Number(ai_efficiency.toFixed(1)),
    planning_practicality: Number(planning_practicality.toFixed(1)),
    creativity: Number(creativity.toFixed(1)),
    overall: Number(overall.toFixed(1)),
  };
}

function buildTeamEvidencePackage(res: TeamUnifiedReport): Record<string, unknown> {
  const judge = computeJudgeCriteriaScore(res);
  const trustVals = Object.values(res.trust_scores || {});
  const avgTrust = trustVals.length ? trustVals.reduce((a, x) => a + x, 0) / trustVals.length : 0;
  const avgNorm = res.scores.length
    ? res.scores.reduce((a, s) => a + s.normalizedScore, 0) / res.scores.length
    : 0;
  const judge_comments: string[] = [
    `기술 완성도 ${judge.technical.toFixed(1)}점: 정량 지표(평균 정규화 ${avgNorm.toFixed(1)})와 신뢰도(평균 ${avgTrust.toFixed(1)}) 기반으로 산출되었습니다.`,
    `AI 활용·효율성 ${judge.ai_efficiency.toFixed(1)}점: 설명 텍스트 밀도와 규칙 기반 결과 정합성을 함께 반영했습니다.`,
    `종합 ${judge.overall.toFixed(1)}점: 기획·실무 접합성(${judge.planning_practicality.toFixed(1)}) 및 창의성(${judge.creativity.toFixed(1)})을 포함한 가중 합산 결과입니다.`,
  ];
  return {
    package_version: "team-evidence-v1",
    submission_type: "judge-ready",
    generated_at: new Date().toISOString(),
    request_id: (res.evaluation_log as { request_id?: string })?.request_id ?? null,
    judge_criteria_scores: judge,
    judge_comments,
    judge_criteria_rationale: {
      input_metrics: {
        average_normalized_score: Number(avgNorm.toFixed(2)),
        average_trust_score: Number(avgTrust.toFixed(2)),
        anomaly_flag_count: (res.anomalies || []).reduce((a, x) => a + (x.flags?.length || 0), 0),
        edge_case_count: (res.edge_cases || []).length,
      },
      weighting_policy: {
        technical: "avg normalized + trust - anomaly penalties",
        ai_efficiency: "explanation density + trust",
        planning_practicality: "edge-case coverage + trust",
        creativity: "score spread + explanation richness",
        overall: "0.35 technical + 0.25 ai + 0.25 planning + 0.15 creativity",
      },
    },
    input_snapshot: {
      project_name: state.team.project_name,
      project_description: state.team.project_description,
      evaluation_criteria: state.team.evaluation_criteria,
      collaboration_edges_json: state.team.collaboration_edges_json,
      members: state.team.members,
    },
    output_report: res,
  };
}

async function refreshHealth(force = false): Promise<void> {
  const now = Date.now();
  if (!force && state.health && !connectionUnstable && now - lastHealthFetchedAt < HEALTH_CACHE_MS) return;
  if (!force && !state.health && !connectionUnstable && now - lastHealthAttemptAt < HEALTH_RETRY_MS) return;
  lastHealthAttemptAt = now;

  for (let attempt = 0; attempt < HEALTH_FETCH_ATTEMPTS; attempt++) {
    const t0 = performance.now();
    try {
      const r = await apiFetch("/api/health", { timeoutMs: HEALTH_ATTEMPT_TIMEOUT_MS });
      lastHealthLatencyMs = Math.max(0, Math.round(performance.now() - t0));
      lastHealthCheckedAt = Date.now();
      if (r.ok) {
        state.health = (await r.json()) as BackendHealth;
        lastHealthFetchedAt = Date.now();
        lastSuccessfulHealthAt = Date.now();
        connectionUnstable = false;
        clearHealthPoll();
        return;
      }
    } catch {
      lastHealthLatencyMs = null;
      lastHealthCheckedAt = Date.now();
    }
    if (attempt < HEALTH_FETCH_ATTEMPTS - 1) {
      await sleepMs(600 * Math.pow(1.55, attempt));
    }
  }

  const tFail = Date.now();
  const sticky = state.health != null && tFail - lastSuccessfulHealthAt < HEALTH_STICKY_MS;
  if (sticky) {
    connectionUnstable = true;
    ensureHealthPollIfNeeded();
    return;
  }
  state.health = null;
  connectionUnstable = false;
  ensureHealthPollIfNeeded();
}

async function submitAnalyze(): Promise<void> {
  readForm();
  state.error = null;
  state.result = null;
  state.loading = true;
  renderSync();

  const body = {
    course_name: state.course_name.trim(),
    student_or_group_label: state.student_or_group_label.trim(),
    learning: {
      weekly_study_hours_self_report: parseOptFloat(state.weekly_study_hours),
      lms_video_watch_ratio: parseOptFloat(state.lms_video_watch_ratio),
      quiz_average: parseOptFloat(state.quiz_average),
      assignment_on_time_ratio: parseOptFloat(state.assignment_on_time_ratio),
      discussion_or_forum_count: parseOptInt(state.discussion_count),
      attendance_or_checkin_ratio: parseOptFloat(state.attendance_ratio),
      notes: state.learning_notes,
    },
    exam: {
      midterm_score: parseOptFloat(state.midterm_score),
      final_or_recent_exam_score: parseOptFloat(state.final_score),
      exam_time_anomaly_note: state.exam_time_anomaly_note,
      notes: state.exam_notes,
    },
    context_for_educator: state.context_for_educator.trim(),
  };

  try {
    const r = await apiFetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as AnalyzeResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.loading = false;
      renderSync();
      return;
    }
    state.result = data;
    state.view = "analyze";
  } catch (e) {
    state.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.loading = false;
  }
  renderSync();
}

function readTeamForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.team.project_name = g("team_project_name")?.value ?? "";
  state.team.project_description = g("team_project_description")?.value ?? "";
  state.team.evaluation_criteria = g("team_evaluation_criteria")?.value ?? "";
  state.team.members = state.team.members.map((_, i) => ({
    name: g(`tm_name_${i}`)?.value ?? "",
    role: g(`tm_role_${i}`)?.value ?? "",
    commits: g(`tm_commits_${i}`)?.value ?? "",
    pull_requests: g(`tm_pr_${i}`)?.value ?? "",
    lines_changed: g(`tm_lines_${i}`)?.value ?? "",
    tasks_completed: g(`tm_tasks_${i}`)?.value ?? "",
    meetings_attended: g(`tm_meet_${i}`)?.value ?? "",
    attendance: g(`tm_attendance_${i}`)?.value ?? "",
    outcome_score: g(`tm_outcome_${i}`)?.value ?? "",
    self_report: g(`tm_self_${i}`)?.value ?? "",
    peer_notes: g(`tm_peer_${i}`)?.value ?? "",
    timeline: [0, 1, 2, 3].map((j) => ({
      period_label: g(`tm_tl_p_${i}_${j}`)?.value ?? "",
      activity_score: g(`tm_tl_s_${i}_${j}`)?.value ?? "",
    })),
  }));
  state.team.collaboration_edges_json = g("team_collaboration_edges")?.value ?? "";
}

async function submitTeam(): Promise<void> {
  readTeamForm();
  state.team.error = null;
  state.team.report = null;
  state.team.loading = true;
  renderSync();

  const buildBody = (): { body: { project_name: string; teamData: Array<Record<string, unknown>> } | null; error?: string } => {
    const rows = state.team.members.filter((m) => m.name.trim());
    const teamData = rows.map((m) => {
      const att = parseOptFloat(m.attendance);
      const meet = parseOptInt(m.meetings_attended);
      const attendance =
        att != null
          ? Math.min(100, Math.max(0, att))
          : Math.min(100, Math.max(0, (meet ?? 0) * 12.5));
      return {
        name: m.name.trim(),
        commits: parseOptInt(m.commits) ?? 0,
        prs: parseOptInt(m.pull_requests) ?? 0,
        lines: parseOptInt(m.lines_changed) ?? 0,
        attendance,
        selfReport: m.self_report,
      };
    });
    if (!state.team.project_name.trim() || teamData.length === 0) {
      return { body: null, error: "프로젝트명과 최소 한 명의 이름을 입력하세요." };
    }
    return {
      body: {
        project_name: state.team.project_name.trim(),
        teamData,
      },
    };
  };

  const built = buildBody();
  if (!built.body) {
    state.team.error = built.error || "입력 오류";
    state.team.loading = false;
    renderSync();
    return;
  }

  try {
    const r = await apiFetch("/api/team/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(built.body),
    });
    const data = (await r.json()) as TeamUnifiedReport & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.team.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.team.loading = false;
      renderSync();
      return;
    }
    state.team.report = data;
    state.team.recheck_result = null;
    void fetchTeamTrends();
    void fetchTeamModelMonitor();
    clearTeamDraft();
  } catch (e) {
    state.team.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.team.loading = false;
  }
  renderSync();
}

async function runTeamRecheck(): Promise<void> {
  readTeamForm();
  state.team.error = null;
  state.team.recheck_loading = true;
  renderSync();

  const rows = state.team.members.filter((m) => m.name.trim());
  const teamData = rows.map((m) => {
    const att = parseOptFloat(m.attendance);
    const meet = parseOptInt(m.meetings_attended);
    const attendance =
      att != null
        ? Math.min(100, Math.max(0, att))
        : Math.min(100, Math.max(0, (meet ?? 0) * 12.5));
    return {
      name: m.name.trim(),
      commits: parseOptInt(m.commits) ?? 0,
      prs: parseOptInt(m.pull_requests) ?? 0,
      lines: parseOptInt(m.lines_changed) ?? 0,
      attendance,
      selfReport: m.self_report,
    };
  });

  if (!state.team.project_name.trim() || teamData.length === 0) {
    state.team.error = "프로젝트명과 최소 한 명의 이름을 입력하세요.";
    state.team.recheck_loading = false;
    renderSync();
    return;
  }

  const body = {
    project_name: state.team.project_name.trim(),
    teamData,
  };

  try {
    const runs: number[] = [];
    for (let i = 0; i < 3; i += 1) {
      const r = await apiFetch("/api/team/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = (await r.json()) as TeamUnifiedReport & { detail?: unknown };
      if (!r.ok) {
        const d = data.detail;
        state.team.error = typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
        state.team.recheck_loading = false;
        renderSync();
        return;
      }
      const mean = data.scores.length
        ? data.scores.reduce((a, s) => a + (s.blendedScore ?? s.normalizedScore), 0) / data.scores.length
        : 0;
      runs.push(mean);
      if (i === 2) state.team.report = data;
    }
    const mean_overall = runs.reduce((a, b) => a + b, 0) / runs.length;
    const spread_overall = Math.max(...runs) - Math.min(...runs);
    const stability_label: "HIGH" | "MEDIUM" | "LOW" =
      spread_overall <= 1.5 ? "HIGH" : spread_overall <= 3.5 ? "MEDIUM" : "LOW";
    const notes: string[] = [
      `3회 평균 점수 ${mean_overall.toFixed(2)}, 편차 ${spread_overall.toFixed(2)}`,
      stability_label === "HIGH"
        ? "평가 일관성이 높습니다."
        : stability_label === "MEDIUM"
          ? "평가 편차가 중간 수준입니다. 데이터 보강 시 안정성이 향상됩니다."
          : "평가 편차가 큽니다. 입력 데이터 품질/충분성을 점검하세요.",
    ];
    state.team.recheck_result = {
      runs: 3,
      mean_overall: Number(mean_overall.toFixed(2)),
      spread_overall: Number(spread_overall.toFixed(2)),
      stability_label,
      notes,
    };
  } catch (e) {
    state.team.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.team.recheck_loading = false;
  }
  renderSync();
}

async function fetchTeamModelMonitor(): Promise<void> {
  try {
    const r = await apiFetch("/api/team/model/monitor?days=30");
    const data = (await r.json()) as { status?: string; monitor?: TeamModelMonitorPayload };
    if (!r.ok || !data.monitor) {
      state.team.model_monitor = null;
      return;
    }
    state.team.model_monitor = data.monitor;
  } catch {
    state.team.model_monitor = null;
  } finally {
    renderSync();
  }
}

async function fetchTeamTrends(): Promise<void> {
  if (!state.team.report) return;
  state.team.trends_loading = true;
  renderSync();
  try {
    const q = new URLSearchParams();
    q.set("days", String(state.team.trends_days || 30));
    if (state.team.trends_member.trim()) q.set("member_name", state.team.trends_member.trim());
    const r = await apiFetch(`/api/team/data/trends?${q.toString()}`);
    const data = (await r.json()) as { status?: string; trends?: TeamTrendsPayload };
    if (!r.ok || !data?.trends) {
      state.team.trends = null;
      return;
    }
    state.team.trends = data.trends;
  } catch {
    state.team.trends = null;
  } finally {
    state.team.trends_loading = false;
    renderSync();
  }
}

function trendPath(points: TeamTrendPoint[], key: "rule_score" | "dl_score" | "blended_score"): string {
  if (!points.length) return "";
  const w = 760;
  const h = 220;
  const pad = 18;
  const step = points.length > 1 ? (w - pad * 2) / (points.length - 1) : 0;
  return points
    .map((p, i) => {
      const x = pad + i * step;
      const v = Math.max(0, Math.min(100, p[key]));
      const y = h - pad - (v / 100) * (h - pad * 2);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function trendInsight(points: TeamTrendPoint[]): string {
  if (points.length < 2) return "데이터가 더 쌓이면 추세 해석이 정확해집니다.";
  const first = points[0].blended_score;
  const last = points[points.length - 1].blended_score;
  const delta = last - first;
  if (delta > 3) return `최근 혼합 점수가 +${delta.toFixed(1)} 상승해 기여 안정성이 개선되는 흐름입니다.`;
  if (delta < -3) return `최근 혼합 점수가 ${delta.toFixed(1)} 하락해 활동/협업 지표 점검이 필요합니다.`;
  return "최근 혼합 점수 변동이 작아 현재 기여 패턴이 비교적 안정적으로 유지됩니다.";
}

function teamTrendHtml(): string {
  const t = state.team.trends;
  if (!state.team.report) return "";
  const memberOptions = [
    `<option value="">전체</option>`,
    ...((t?.members || state.team.report.scores.map((s) => s.member_name)).map(
      (m) => `<option value="${escapeHtml(m)}" ${state.team.trends_member === m ? "selected" : ""}>${escapeHtml(m)}</option>`
    )),
  ].join("");
  const pts =
    t && Object.keys(t.series).length
      ? state.team.trends_member
        ? t.series[state.team.trends_member] || []
        : t.series[Object.keys(t.series)[0]] || []
      : [];
  const pRule = trendPath(pts, "rule_score");
  const pDl = trendPath(pts, "dl_score");
  const pBlend = trendPath(pts, "blended_score");
  return `
  <section class="panel hud-panel trend-panel">
    <div class="row-actions no-print">
      <label class="lbl">기간
        <select id="trend-days" class="txt">
          <option value="30" ${state.team.trends_days === 30 ? "selected" : ""}>30일</option>
          <option value="60" ${state.team.trends_days === 60 ? "selected" : ""}>60일</option>
          <option value="90" ${state.team.trends_days === 90 ? "selected" : ""}>90일</option>
        </select>
      </label>
      <label class="lbl">멤버
        <select id="trend-member" class="txt">${memberOptions}</select>
      </label>
      <button type="button" class="btn btn-ghost btn-sm" id="btn-trend-refresh" ${state.team.trends_loading ? "disabled" : ""}>${state.team.trends_loading ? "로딩..." : "추세 갱신"}</button>
    </div>
    <h3 class="subh">개인 기여도 추세 (Rule / DL / Blended)</h3>
    ${
      pts.length
        ? `<svg class="trend-chart" viewBox="0 0 760 220" role="img" aria-label="기여도 추세 그래프">
        <path d="${pRule}" class="trend-line trend-line--rule"></path>
        <path d="${pDl}" class="trend-line trend-line--dl"></path>
        <path d="${pBlend}" class="trend-line trend-line--blend"></path>
      </svg>
      <p class="muted small">${escapeHtml(trendInsight(pts))}</p>`
        : `<p class="muted small">추세 데이터가 아직 충분하지 않습니다. 팀 평가를 누적 실행하면 그래프가 표시됩니다.</p>`
    }
  </section>`;
}

function modelMonitorHtml(): string {
  const m = state.team.model_monitor;
  if (!m) return "";
  const driftSign = m.rule_dl_drift >= 0 ? "+" : "";
  return `
  <section class="panel hud-panel">
    <h3 class="subh">모델 모니터링 (최근 ${m.window_days}일)</h3>
    <p class="muted small">샘플 ${m.samples}건 · Rule 평균 ${m.rule_avg.toFixed(1)} · DL 평균 ${m.dl_avg.toFixed(1)} · 혼합 평균 ${m.blended_avg.toFixed(1)}</p>
    <p class="muted small">Rule↔DL 드리프트: ${driftSign}${m.rule_dl_drift.toFixed(2)} · 자동 재학습 ${m.recent_auto_retrain_count}회</p>
  </section>`;
}

function recheckHtml(): string {
  const r = state.team.recheck_result;
  if (!r) return "";
  const pill = r.stability_label === "HIGH" ? "pill-on" : r.stability_label === "MEDIUM" ? "pill-muted" : "pill-warn";
  return `
  <section class="panel hud-panel">
    <h3 class="subh">Re-check 검증 결과</h3>
    <p><span class="pill ${pill}">${r.stability_label}</span> 평균 ${r.mean_overall.toFixed(2)} · 편차 ${r.spread_overall.toFixed(2)} (${r.runs}회)</p>
    <ul class="report-flags">${r.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
  </section>`;
}

function confidenceGateHtml(rep: TeamUnifiedReport): string {
  const trustVals = Object.values(rep.trust_scores || {});
  const avgTrust = trustVals.length ? trustVals.reduce((a, b) => a + b, 0) / trustVals.length : 0;
  const anomalyCount = (rep.anomalies || []).reduce((a, x) => a + (x.flags?.length || 0), 0);
  const dataPoints = rep.scores.length;
  const confidence = Math.max(0, Math.min(100, avgTrust - anomalyCount * 2 + Math.min(10, dataPoints * 1.5)));
  const level = confidence >= 75 ? "HIGH" : confidence >= 55 ? "MEDIUM" : "LOW";
  const guide =
    level === "HIGH"
      ? "신뢰도 양호: 현재 결과를 기준으로 면담/피드백을 진행할 수 있습니다."
      : level === "MEDIUM"
        ? "중간 신뢰도: 협업 근거(PR/리뷰/출석) 데이터를 추가하면 결과 안정성이 향상됩니다."
        : "낮은 신뢰도: 데이터가 부족하거나 이상 플래그가 많습니다. 입력 보강 후 재평가를 권장합니다.";
  const cls = level === "HIGH" ? "pill-on" : level === "MEDIUM" ? "pill-muted" : "pill-warn";
  return `
  <section class="panel hud-panel">
    <h3 class="subh">평가 신뢰도 게이트</h3>
    <p><span class="pill ${cls}">${level}</span> <strong class="score-num">${confidence.toFixed(1)}</strong> / 100</p>
    <p class="muted small">${escapeHtml(guide)}</p>
  </section>`;
}

function promotionGateHtml(rep: TeamUnifiedReport): string {
  const info = (rep.dl_model_info as { quality?: { promotion_gate?: PromotionGateInfo } } | undefined)
    ?.quality?.promotion_gate;
  if (!info) return "";
  const accepted = !!info.accepted;
  const reasons = (info.reasons || []).filter((x) => typeof x === "string");
  const pill = accepted ? "pill-on" : "pill-warn";
  const label = accepted ? "PROMOTED" : "REJECTED";
  return `
  <section class="panel hud-panel">
    <h3 class="subh">DL 승격 게이트</h3>
    <p><span class="pill ${pill}">${label}</span> <span class="muted small">신규 학습 모델 자동 승격 판정</span></p>
    ${
      reasons.length
        ? `<ul class="report-flags">${reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`
        : `<p class="muted small">사유 정보 없음</p>`
    }
  </section>`;
}

function dlExplainOpsPanelHtml(rep: TeamUnifiedReport): string {
  const dl = rep.dl_model_info as
    | {
        quality?: {
          semantic_encoder?: { enabled?: boolean; model_id?: string | null; note_ko?: string };
          operations_playbook?: {
            recommendation_level?: string;
            summary_ko?: string;
            checklist_ko?: string[];
          };
          explainability_snapshot?: {
            permutation_importance_top?: { feature?: string; delta_mae?: number }[];
            input_noise_std_training?: number;
            note_ko?: string;
          };
        };
      }
    | undefined;
  const pb = dl?.quality?.operations_playbook;
  const ex = dl?.quality?.explainability_snapshot;
  const sem = dl?.quality?.semantic_encoder;
  if (!pb && !ex && !sem) return "";
  const lvl = (pb?.recommendation_level || "ok").toLowerCase();
  const pill =
    lvl === "action" ? "pill-warn" : lvl === "watch" ? "pill-muted" : "pill-on";
  const label = lvl === "action" ? "조치 권고" : lvl === "watch" ? "주의" : "양호";
  const summary = pb?.summary_ko ? `<p class="muted small">${escapeHtml(pb.summary_ko)}</p>` : "";
  const checklist = (pb?.checklist_ko || []).filter((x) => typeof x === "string");
  const listHtml = checklist.length
    ? `<ul class="report-flags">${checklist.map((c) => `<li>${escapeHtml(c)}</li>`).join("")}</ul>`
    : "";
  const perm = ex?.permutation_importance_top || [];
  const permHtml =
    perm.length > 0
      ? `<p class="muted small"><strong>치환 중요도(상위)</strong></p><ul class="report-flags">${perm
          .slice(0, 5)
          .map(
            (p) =>
              `<li>${escapeHtml(String(p.feature ?? "?"))} · ΔMAE ${typeof p.delta_mae === "number" ? p.delta_mae.toFixed(4) : "?"}</li>`,
          )
          .join("")}</ul>`
      : "";
  const noise =
    typeof ex?.input_noise_std_training === "number" && ex.input_noise_std_training > 0
      ? `<p class="muted small">학습 입력 노이즈 σ=${ex.input_noise_std_training}</p>`
      : "";
  const semLine =
    sem && typeof sem === "object"
      ? `<p class="muted small">문장 임베딩(거대 인코더): <strong>${sem.enabled ? "사용" : "미사용"}</strong>${
          sem.model_id ? ` · ${escapeHtml(String(sem.model_id))}` : ""
        }</p>`
      : "";
  return `
  <section class="panel hud-panel">
    <h3 class="subh">DL 설명 · 운영 권고</h3>
    ${semLine}
    ${
      pb
        ? `<p><span class="pill ${pill}">${escapeHtml(label)}</span> <span class="muted small">recommendation_level=${escapeHtml(lvl)}</span></p>
    ${summary}
    ${listHtml}`
        : ""
    }
    ${noise}
    ${permHtml}
    ${ex?.note_ko ? `<p class="muted small">${escapeHtml(ex.note_ko)}</p>` : ""}
  </section>`;
}

function promotionGateBannerHtml(rep: TeamUnifiedReport): string {
  const info = (rep.dl_model_info as { quality?: { promotion_gate?: PromotionGateInfo } } | undefined)
    ?.quality?.promotion_gate;
  if (!info || info.accepted) return "";
  const reasons = (info.reasons || []).filter((x) => typeof x === "string");
  const reasonText = reasons.length ? reasons.join(", ") : "metrics_regressed";
  return `
  <section class="panel hud-panel">
    <h3 class="subh">승격 차단 알림</h3>
    <p><span class="pill pill-warn">DEPLOY BLOCKED</span> <strong>신규 DL 모델 승격이 차단되었습니다.</strong></p>
    <p class="muted small">사유: ${escapeHtml(reasonText)}</p>
    <div class="row-actions no-print">
      <button type="button" class="btn btn-ghost btn-sm" id="btn-copy-promote-hint">재학습 권고 문구 복사</button>
    </div>
  </section>`;
}

function whatIfSimulatorHtml(rep: TeamUnifiedReport): string {
  const names = rep.scores.map((s) => s.member_name);
  const selected = state.team.what_if.member_name || names[0] || "";
  const member = rep.scores.find((s) => s.member_name === selected) || rep.scores[0];
  if (!member) return "";
  const cDelta = Math.max(0, state.team.what_if.add_commits);
  const pDelta = Math.max(0, state.team.what_if.add_prs);
  const lDelta = Math.max(0, state.team.what_if.add_lines);
  const uplift = cDelta * 0.9 + pDelta * 1.3 + Math.min(20, lDelta / 180);
  const after = Math.max(0, Math.min(100, member.normalizedScore + uplift));
  const blend = member.blendedScore ?? member.normalizedScore;
  const blendAfter = Math.max(0, Math.min(100, blend + uplift * 0.85));
  const options = names
    .map((n) => `<option value="${escapeHtml(n)}" ${n === selected ? "selected" : ""}>${escapeHtml(n)}</option>`)
    .join("");
  return `
  <section class="panel hud-panel whatif-panel">
    <h3 class="subh">What-if 시뮬레이터 (가산 활동 시 예상 변화)</h3>
    <div class="grid-2 no-print">
      <div><label class="lbl">멤버</label><select id="whatif-member" class="txt">${options}</select></div>
      <div><label class="lbl">커밋 추가</label><input id="whatif-commits" class="txt" type="number" min="0" value="${cDelta}" /></div>
      <div><label class="lbl">PR 추가</label><input id="whatif-prs" class="txt" type="number" min="0" value="${pDelta}" /></div>
      <div><label class="lbl">라인 추가</label><input id="whatif-lines" class="txt" type="number" min="0" value="${lDelta}" /></div>
    </div>
    <p class="muted small">${escapeHtml(selected)} 기준: 정규화 <strong>${member.normalizedScore.toFixed(1)}</strong> → <strong>${after.toFixed(1)}</strong> / 혼합 <strong>${blend.toFixed(1)}</strong> → <strong>${blendAfter.toFixed(1)}</strong></p>
  </section>`;
}

function readAtRiskForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.atRisk.course_name = g("ar_course")?.value ?? "";
  state.atRisk.student_label = g("ar_student")?.value ?? "";
  state.atRisk.attendance_ratio = g("ar_attendance")?.value ?? "";
  state.atRisk.assignment_ratio = g("ar_assignment")?.value ?? "";
  state.atRisk.notes = g("ar_notes")?.value ?? "";
}

async function submitAtRisk(): Promise<void> {
  readAtRiskForm();
  state.atRisk.error = null;
  state.atRisk.result = null;
  state.atRisk.loading = true;
  renderSync();

  const att = parseOptFloat(state.atRisk.attendance_ratio);
  const asg = parseOptFloat(state.atRisk.assignment_ratio);
  if (att === null || asg === null) {
    state.atRisk.error = "출석 비율과 과제 제출 비율(0–100)을 모두 입력하세요.";
    state.atRisk.loading = false;
    renderSync();
    return;
  }
  const a = Math.min(100, Math.max(0, att));
  const b = Math.min(100, Math.max(0, asg));
  const engagement = (a + b) / 2;
  const weeks = [{ week_label: "출석·과제", engagement }];

  const body = {
    course_name: state.atRisk.course_name.trim(),
    student_label: state.atRisk.student_label.trim(),
    weeks,
    notes: state.atRisk.notes,
  };

  try {
    const r = await apiFetch("/api/at-risk/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as AtRiskResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.atRisk.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.atRisk.loading = false;
      renderSync();
      return;
    }
    state.atRisk.result = data;
  } catch (e) {
    state.atRisk.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.atRisk.loading = false;
  }
  renderSync();
}

function readFeedbackForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLTextAreaElement | null;
  state.feedback.rubric = g("fb_rubric")?.value ?? "";
  state.feedback.assignment_prompt = g("fb_prompt")?.value ?? "";
  state.feedback.submission = g("fb_submission")?.value ?? "";
}

async function submitFeedback(): Promise<void> {
  readFeedbackForm();
  state.feedback.error = null;
  state.feedback.result = null;
  state.feedback.loading = true;
  renderSync();

  if (!state.feedback.rubric.trim() || !state.feedback.submission.trim()) {
    state.feedback.error = "루브릭과 제출물을 입력하세요.";
    state.feedback.loading = false;
    renderSync();
    return;
  }

  const body = {
    rubric: state.feedback.rubric.trim(),
    assignment_prompt: state.feedback.assignment_prompt,
    submission: state.feedback.submission,
  };

  try {
    const r = await apiFetch("/api/feedback/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as FeedbackResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.feedback.error =
        typeof d === "string"
          ? d
          : d && typeof d === "object" && "detail" in d
            ? String((d as { detail: string }).detail)
            : "요청 실패";
      state.feedback.loading = false;
      renderSync();
      return;
    }
    state.feedback.result = data;
  } catch (e) {
    state.feedback.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.feedback.loading = false;
  }
  renderSync();
}

function readRubricAlignForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLTextAreaElement | null;
  state.rubricAlign.rubric = g("ra_rubric")?.value ?? "";
  state.rubricAlign.grader_rationale = g("ra_rationale")?.value ?? "";
  state.rubricAlign.student_work = g("ra_student")?.value ?? "";
}

function readRubricGenForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.rubricGen.course_name = g("rg_course")?.value ?? "";
  state.rubricGen.learning_objectives = g("rg_objectives")?.value ?? "";
  state.rubricGen.assignment_type = g("rg_assignment")?.value ?? "";
  state.rubricGen.max_criteria = g("rg_max_criteria")?.value ?? "5";
}

async function submitRubricGenerate(): Promise<void> {
  readRubricGenForm();
  state.rubricGen.error = null;
  state.rubricGen.result = null;
  state.rubricGen.loading = true;
  renderSync();

  const obj = state.rubricGen.learning_objectives.trim();
  if (obj.length < 10) {
    state.rubricGen.error = "학습 목표를 10자 이상 입력하세요.";
    state.rubricGen.loading = false;
    renderSync();
    return;
  }

  let maxC = parseInt(state.rubricGen.max_criteria.trim(), 10);
  if (!Number.isFinite(maxC)) maxC = 5;
  maxC = Math.min(8, Math.max(3, maxC));

  const body = {
    course_name: state.rubricGen.course_name.trim(),
    assignment_type: state.rubricGen.assignment_type.trim(),
    learning_objectives: obj,
    max_criteria: maxC,
  };

  try {
    const r = await apiFetch("/api/rubric/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as RubricGenerateResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.rubricGen.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.rubricGen.loading = false;
      renderSync();
      return;
    }
    state.rubricGen.result = data;
  } catch (e) {
    state.rubricGen.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.rubricGen.loading = false;
  }
  renderSync();
}

async function submitRubricAlign(): Promise<void> {
  readRubricAlignForm();
  state.rubricAlign.error = null;
  state.rubricAlign.result = null;
  state.rubricAlign.loading = true;
  renderSync();

  if (!state.rubricAlign.rubric.trim() || !state.rubricAlign.grader_rationale.trim()) {
    state.rubricAlign.error = "루브릭과 채점 근거를 입력하세요.";
    state.rubricAlign.loading = false;
    renderSync();
    return;
  }

  const body = {
    rubric: state.rubricAlign.rubric.trim(),
    grader_rationale: state.rubricAlign.grader_rationale.trim(),
    student_work_excerpt: state.rubricAlign.student_work.trim(),
  };

  try {
    const r = await apiFetch("/api/rubric/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await r.json()) as RubricAlignResponse & { detail?: unknown };
    if (!r.ok) {
      const d = data.detail;
      state.rubricAlign.error =
        typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : "요청 실패";
      state.rubricAlign.loading = false;
      renderSync();
      return;
    }
    state.rubricAlign.result = data;
  } catch (e) {
    state.rubricAlign.error = e instanceof Error ? e.message : String(e);
  } finally {
    state.rubricAlign.loading = false;
  }
  renderSync();
}

function connectionStripHtml(): string {
  const connected = !!state.health;
  const unstable = connected && connectionUnstable;
  const latency = lastHealthLatencyMs;
  const isDegraded = connected && !unstable && latency != null && latency >= 1200;
  const status = unstable ? "RECONNECT" : connected ? (isDegraded ? "DEGRADED" : "LIVE") : "OFFLINE";
  const stripClass = unstable
    ? "conn-strip conn-strip--partial"
    : connected
      ? isDegraded
        ? "conn-strip conn-strip--degraded"
        : "conn-strip conn-strip--live"
      : "conn-strip conn-strip--offline";
  const dotClass = unstable
    ? "api-dot api-dot--degraded"
    : connected
      ? isDegraded
        ? "api-dot api-dot--degraded"
        : "api-dot api-dot--on"
      : "api-dot api-dot--off";
  const mainMsg =
    status === "RECONNECT"
      ? "재연결 중 — 마지막 정상 연결 정보 유지(자동 재시도)"
      : status === "LIVE"
        ? "프론트 ↔ 백엔드 정상 연결"
        : status === "DEGRADED"
          ? "프론트 ↔ 백엔드 연결(지연)"
          : "프론트 ↔ 백엔드 미연결";
  const checked = lastHealthCheckedAt ? new Date(lastHealthCheckedAt).toLocaleTimeString() : "-";
  const latencyTxt = latency != null ? `${latency}ms` : "-";
  return `
  <div class="${stripClass}" role="status" aria-live="polite">
    <div class="conn-strip-inner">
      <span class="${dotClass}" aria-hidden="true"></span>
      <span class="conn-msg"><strong>${status}</strong> · ${mainMsg}</span>
      <span class="conn-code">응답 ${latencyTxt} · 확인 ${checked}</span>
    </div>
  </div>`;
}

function navHtml(): string {
  const cur = (v: SiteView) => (state.view === v ? "nav-link active" : "nav-link");
  return `
  <header class="site-header site-header--glass">
    <div class="nav-inner">
      <a href="#/" class="brand brand-mark" data-view="home" aria-label="팀 프로젝트 기여도 평가 — 홈">
        <span class="brand-glow" aria-hidden="true"></span>
        <span class="brand-title">팀 공정 평가</span>
        <span class="brand-tag">기여도 자동 산출</span>
      </a>
      <nav class="nav" aria-label="주요 메뉴">
        <button type="button" class="${cur("home")}" data-view="home">홈</button>
        <button type="button" class="${cur("team")}" data-view="team">팀 평가</button>
        <button type="button" class="${cur("hub")}" data-view="hub">부가 분석</button>
      </nav>
    </div>
  </header>`;
}

function footerHtml(): string {
  const docsHref = API_BASE ? `${API_BASE}/docs` : "/docs";
  const openApiLink =
    state.health?.version != null
      ? `<a href="${escapeHtml(docsHref)}" target="_blank" rel="noopener noreferrer">OpenAPI ${escapeHtml(state.health.version)}</a>`
      : `<a href="${escapeHtml(docsHref)}" target="_blank" rel="noopener noreferrer">OpenAPI</a>`;
  return `
  <footer class="site-footer">
    <div class="footer-inner">
      <p class="footer-line">
        ${openApiLink}
        · 교육 보조 도구이며 징계·단정에 사용할 수 없습니다.
      </p>
      <p class="footer-muted">팀 프로젝트 기여도 자동 평가 시스템 — 교육 보조·참고용</p>
    </div>
  </footer>`;
}

function homeHtml(): string {
  return `
  <div class="page page-animate home-page">
    <section class="hero-block home-hero-main home-hero-visual">
      <div class="hero-orb hero-orb--a" aria-hidden="true"></div>
      <div class="hero-orb hero-orb--b" aria-hidden="true"></div>
      <p class="eyebrow eyebrow--shine">교육 보조 · 참고용</p>
      <h1 class="home-headline home-headline--fx">팀 프로젝트 기여도 자동 평가</h1>
      <p class="hero-text home-lead">
        Git·활동 지표를 바탕으로 기여를 정리하고, 무임승차 의심·협업 네트워크 등을 참고용으로 제시합니다.
        최종 성적·징계 판단을 대체하지 않습니다.
      </p>
      <div class="pill-row hero-pills">
        <span class="pill pill-on">기여 지수</span>
        <span class="pill pill-on">이상·무임승차 탐지</span>
        <span class="pill pill-on">선택적 LLM 해설</span>
        <span class="pill pill-muted">과정–시험·이탈 등 부가 분석</span>
      </div>
      <p class="row-actions" style="margin-top:1.25rem;">
        <button type="button" class="btn btn-primary" data-view="team">팀 평가 시작</button>
        <button type="button" class="btn btn-ghost" data-view="hub">부가 분석 도구</button>
      </p>
      <p class="muted small home-lead">OpenAPI 연동 · 로컬 또는 Docker로 실행</p>
    </section>
  </div>`;
}

function hubHtml(): string {
  return `
  <div class="page page-animate home-page">
    <section class="section-block hud-section home-section">
      <h2 class="section-title">부가 분석 (평가 보조)</h2>
      <p class="muted small" style="margin:-0.5rem 0 1rem;">팀 평가와 별도로, 과정·시험 불일치·이탈 신호 등을 점검합니다.</p>
      <div class="analysis-flow">
        <article class="solution-tile solution-tile--live hud-card analysis-step">
          <span class="solution-badge solution-badge--on">1. 과정 vs 시험 불일치</span>
          <h3>과제 점수와 시험 점수를 비교해 평가의 불균형을 탐지합니다</h3>
          <p>과정 데이터와 시험 결과를 비교해 왜 불일치가 발생했는지 확인하고, 평가 이상 징후를 조기에 발견하기 위해 필요합니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="analyze">분석 열기</button>
        </article>
        <article class="solution-tile solution-tile--live hud-card analysis-step">
          <span class="solution-badge solution-badge--on">2. 학습 이탈 신호</span>
          <h3>출석과 과제 데이터를 기반으로 참여 저하 위험을 감지합니다</h3>
          <p>학습 참여가 낮아지는 시점을 조기에 식별해 개입 타이밍을 잡고, 팀 기여도 하락을 예방하기 위해 필요합니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="at-risk">분석 열기</button>
        </article>
        <article class="solution-tile solution-tile--live hud-card analysis-step">
          <span class="solution-badge solution-badge--on">3. 과제 피드백</span>
          <h3>루브릭과 제출물을 분석해 교수 피드백 초안을 생성합니다</h3>
          <p>반복되는 피드백 작성 시간을 줄이고, 기준에 맞는 일관된 코멘트를 제공하기 위해 필요합니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="feedback">분석 열기</button>
        </article>
        <article class="solution-tile solution-tile--live hud-card analysis-step">
          <span class="solution-badge solution-badge--on">4. 루브릭 정합성</span>
          <h3>루브릭 기준과 채점 근거의 일치 여부를 점검합니다</h3>
          <p>채점 기준의 일관성과 공정성을 유지하고, 평가 결과에 대한 설명 가능성을 높이기 위해 필요합니다.</p>
          <button type="button" class="btn btn-primary btn-block" data-view="rubric">분석 열기</button>
        </article>
      </div>
    </section>
  </div>`;
}

function teamMemberBlock(i: number, m: TeamMemberRow): string {
  const tl = m.timeline?.length ? m.timeline : [0, 1, 2, 3].map(() => emptyTimelineRow());
  const timelineRows = tl.slice(0, 4).map((row, j) => {
    const p = row.period_label ?? "";
    const s = row.activity_score ?? "";
    return `
    <div class="grid-2">
      <div><label class="lbl">기간 ${j + 1}</label><input class="txt" id="tm_tl_p_${i}_${j}" placeholder="예: 3주차" value="${escapeHtml(p)}" /></div>
      <div><label class="lbl">활동 0–100</label><input class="txt" id="tm_tl_s_${i}_${j}" type="number" min="0" max="100" step="0.1" value="${escapeHtml(s)}" /></div>
    </div>`;
  });
  return `
  <div class="member-block hud-panel" data-member-idx="${i}">
    <h4 class="subh">멤버 ${i + 1}</h4>
    <div class="grid-2">
      <div><label class="lbl">이름</label><input class="txt" id="tm_name_${i}" value="${escapeHtml(m.name)}" /></div>
      <div><label class="lbl">역할</label><input class="txt" id="tm_role_${i}" value="${escapeHtml(m.role)}" /></div>
    </div>
    <div class="grid-3">
      <div><label class="lbl">커밋 수</label><input class="txt" id="tm_commits_${i}" type="number" min="0" value="${escapeHtml(m.commits)}" /></div>
      <div><label class="lbl">PR 수</label><input class="txt" id="tm_pr_${i}" type="number" min="0" value="${escapeHtml(m.pull_requests)}" /></div>
      <div><label class="lbl">변경 라인</label><input class="txt" id="tm_lines_${i}" type="number" min="0" value="${escapeHtml(m.lines_changed)}" /></div>
      <div><label class="lbl">완료 태스크</label><input class="txt" id="tm_tasks_${i}" type="number" min="0" value="${escapeHtml(m.tasks_completed)}" /></div>
      <div><label class="lbl">회의 출석</label><input class="txt" id="tm_meet_${i}" type="number" min="0" value="${escapeHtml(m.meetings_attended)}" /></div>
      <div><label class="lbl">출석·참여 % (0–100)</label><input class="txt" id="tm_attendance_${i}" type="number" min="0" max="100" step="0.1" placeholder="비우면 회의 횟수×12.5%" value="${escapeHtml(m.attendance)}" /></div>
      <div><label class="lbl">결과 점수 (선택, 0–100)</label><input class="txt" id="tm_outcome_${i}" type="number" min="0" max="100" step="0.1" placeholder="발표·동료평가 등" value="${escapeHtml(m.outcome_score)}" /></div>
    </div>
    <details class="timeline-details">
      <summary>주차별 활동 점수 (선택, 무임승차·타임라인에 반영)</summary>
      <p class="muted small">같은 주차 라벨을 팀원 간에 맞추면 비교가 쉽습니다. 비우면 서버가 가상 시계열을 생성합니다.</p>
      ${timelineRows.join("")}
    </details>
    <label class="lbl">자기 서술</label>
    <textarea class="txt" id="tm_self_${i}" rows="2">${escapeHtml(m.self_report)}</textarea>
    <label class="lbl">동료 메모</label>
    <textarea class="txt" id="tm_peer_${i}" rows="2">${escapeHtml(m.peer_notes)}</textarea>
  </div>`;
}

function teamReportHtml(): string {
  const rep = state.team.report;
  if (!rep) return "";
  const n = rep.scores.length || 1;
  const log = rep.evaluation_log as { request_id?: string; timestamp?: string };
  const meta = [log.request_id ? `요청 ${escapeHtml(log.request_id)}` : "", log.timestamp ? escapeHtml(log.timestamp) : ""]
    .filter(Boolean)
    .join(" · ");
  const edgeCases = (rep.edge_cases || []).length
    ? `<div class="hud-panel"><h3 class="subh">엣지 케이스</h3><ul>${rep.edge_cases.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul></div>`
    : "";
  const scoreSections = rep.scores
    .map((s) => {
      const an = rep.analysis.find((x) => x.member_name === s.member_name);
      const am = rep.anomalies.find((x) => x.member_name === s.member_name);
      const trust = rep.trust_scores[s.member_name];
      const b = s.breakdown;
      const p = s.weighted_points;
      const rel = s.data_reliability;
      const flags = (am?.flags || []).length
        ? `<ul class="report-flags">${(am?.flags || []).map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>`
        : `<p class="muted small">없음</p>`;
      const strengths = (an?.strengths || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
      const weaknesses = (an?.weaknesses || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
      const rec = (an?.recommended_actions || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
      return `
  <article class="report-member-flow hud-panel">
    <header class="report-member-head">
      <h3 class="subh">${escapeHtml(s.member_name)}</h3>
      <p class="report-summary-line"><strong>기여도(정규화)</strong> ${s.normalizedScore.toFixed(0)}점 · <strong>순위</strong> ${s.rank}/${n} (상위 ${s.top_percent.toFixed(1)}%)</p>
      ${
        s.dl_score != null && s.blendedScore != null
          ? `<p class="muted small"><strong>DL 보강</strong> ${s.dl_score.toFixed(1)} · <strong>최종 혼합</strong> ${s.blendedScore.toFixed(1)} ${s.dl_confidence != null ? `(신뢰도 ${s.dl_confidence.toFixed(1)})` : ""}</p>`
          : ""
      }
      ${
        s.dl_top_factors?.length
          ? `<p class="muted small">DL 영향 요인: ${escapeHtml(s.dl_top_factors.join(", "))}</p>`
          : ""
      }
      <p class="muted small">팀 평균 대비 ${s.pct_vs_team_mean >= 0 ? "+" : ""}${s.pct_vs_team_mean.toFixed(1)}%</p>
      ${trust != null ? `<p class="report-trust"><strong>데이터 신뢰도</strong> <span class="score-num">${trust.toFixed(0)}</span>% <span class="muted small">(Git ${rel.git_ratio_percent.toFixed(0)}% / 자기서술 ${rel.self_report_ratio_percent.toFixed(0)}%)</span></p>` : ""}
      ${rel.note ? `<p class="muted small">⚠️ ${escapeHtml(rel.note)}</p>` : ""}
    </header>
    <section class="report-section">
      <h4 class="report-h4">점수 근거 (Score Engine)</h4>
      <ul class="report-breakdown">
        <li>커밋: ${p.commits.toFixed(2)}점 (기준값 ${b.commits.toFixed(1)})</li>
        <li>PR: ${p.prs.toFixed(2)}점 (기준값 ${b.prs.toFixed(1)})</li>
        <li>코드라인: ${p.lines.toFixed(2)}점 (기준값 ${b.lines.toFixed(1)})</li>
        <li>출석: ${p.attendance.toFixed(2)}점 (기준값 ${b.attendance.toFixed(1)})</li>
        <li>자기서술: ${p.self_report.toFixed(2)}점 (기준값 ${b.self_report.toFixed(1)})</li>
      </ul>
      <p class="muted small">환산 raw ${s.rawScore.toFixed(2)}</p>
    </section>
    <section class="report-section">
      <h4 class="report-h4">이상 감지 (규칙)</h4>
      ${flags}
    </section>
    <section class="report-section report-ai">
      <h4 class="report-h4">AI 분석 (설명)</h4>
      <p class="prose">${escapeHtml(an?.summary || "")}</p>
      ${
        an?.metric_dl_note?.trim()
          ? `<p class="muted small"><strong>규칙·DL 정합</strong> ${escapeHtml(an.metric_dl_note)}</p>`
          : ""
      }
      <p class="muted small">${escapeHtml(an?.position_in_team || "")}</p>
      <div class="grid-2 report-strength-weak">
        <div><strong>강점</strong><ul>${strengths}</ul></div>
        <div><strong>개선</strong><ul>${weaknesses}</ul></div>
      </div>
    </section>
    <section class="report-section">
      <h4 class="report-h4">추천 행동</h4>
      <ul>${rec}</ul>
    </section>
  </article>`;
    })
    .join("");
  return `
  <section class="panel panel-result hud-panel team-report-root team-print-root" id="team-printable-root">
    <div class="result-toolbar no-print" role="toolbar">
      <button type="button" class="btn btn-ghost btn-sm" id="btn-team-export-json">JSON 내보내기</button>
      <button type="button" class="btn btn-ghost btn-sm" id="btn-team-export-evidence">심사 제출 패키지</button>
      <button type="button" class="btn btn-ghost btn-sm" id="btn-team-recheck" ${state.team.recheck_loading ? "disabled" : ""}>${state.team.recheck_loading ? "검증 중..." : "Re-check(3회)"} </button>
      <button type="button" class="btn btn-ghost btn-sm" id="btn-team-copy-summary">요약 복사</button>
      <button type="button" class="btn btn-ghost btn-sm" id="btn-team-print">인쇄·PDF</button>
    </div>
    <h2>팀 기여도 평가 리포트</h2>
    <p class="muted small">데이터 → Score Engine → Anomaly → AI 설명 (점수는 AI가 매기지 않음)</p>
    ${
      rep.dl_model_info
        ? `<p class="muted small">DL 보강 모델: ${escapeHtml(String(rep.dl_model_info.model_name ?? "enabled"))} · 혼합식 ${escapeHtml(String(rep.dl_model_info.blend_formula ?? "dynamic blend"))}</p>`
        : ""
    }
    ${meta ? `<p class="result-meta muted small no-print">${meta}</p>` : ""}
    ${
      rep.team_narrative?.trim()
        ? `<div class="hud-panel report-team-narrative"><h3 class="subh">팀 총평 (LLM)</h3><p class="prose">${escapeHtml(rep.team_narrative)}</p>${
            rep.ai_meta && typeof rep.ai_meta === "object" && rep.ai_meta.mode
              ? `<p class="muted small">엔진: ${escapeHtml(String(rep.ai_meta.mode))}</p>`
              : ""
          }</div>`
        : ""
    }
    ${promotionGateBannerHtml(rep)}
    ${dlExplainOpsPanelHtml(rep)}
    ${confidenceGateHtml(rep)}
    ${promotionGateHtml(rep)}
    ${edgeCases}
    <div class="report-flow">${scoreSections}</div>
    ${whatIfSimulatorHtml(rep)}
    ${teamTrendHtml()}
    ${recheckHtml()}
    ${modelMonitorHtml()}
    <p class="footer-note muted small">${escapeHtml(rep.disclaimer)}</p>
  </section>`;
}


function teamHtml(): string {
  const blocks = state.team.members.map((m, i) => teamMemberBlock(i, m)).join("");
  return `
  <div class="page page-analyze page-animate analyze-page">
    <p class="eyebrow eyebrow--shine">Team fairness · 기여도 평가</p>
    <h1 class="page-title page-title--hero">팀 프로젝트 공정성 — 자동 기여도 평가</h1>
    <p class="lead analyze-lead">
      Git 기반 팀 데이터로 <strong>공정한 기여도</strong>를 정량·정성으로 평가하고, <strong>설명 가능한 리포트</strong>를 만듭니다.
      <strong>계산(Score Engine) · 판단(Anomaly) · 설명(AI)</strong>을 분리했습니다. AI는 점수를 매기지 않습니다.
    </p>
    <section class="panel hud-panel team-arch-panel">
      <h3 class="subh">아키텍처 (계산 / 판단 / 설명 분리)</h3>
      <div class="team-arch-grid">
        <article class="team-arch-step"><strong>Score Engine</strong><p class="muted small">커밋·PR·라인·출석·자기서술을 팀 중앙값 대비 정규화해 가중 합산합니다. AI 미사용.</p></article>
        <article class="team-arch-step"><strong>Anomaly Detector</strong><p class="muted small">3~4개 규칙으로 이상 패턴만 표시합니다. AI 미사용.</p></article>
        <article class="team-arch-step"><strong>AI Analyzer</strong><p class="muted small">정량 결과·팀 비교·플래그를 바탕으로 요약·강점·개선·행동만 서술합니다.</p></article>
      </div>
    </section>
    <p class="muted small team-practice-hint">입력 내용은 브라우저에 <strong>자동 임시 저장</strong>됩니다(성공적으로 평가하면 삭제). 조교·기록용으로 JSON 내보내기·요약 복사를 활용하세요.</p>

    <section class="panel hud-panel team-form-panel">
      <div class="grid-2">
        <div>
          <label class="lbl">프로젝트명</label>
          <input class="txt" id="team_project_name" value="${escapeHtml(state.team.project_name)}" />
        </div>
      </div>
      <label class="lbl">프로젝트 설명</label>
      <textarea class="txt" id="team_project_description" rows="2">${escapeHtml(state.team.project_description)}</textarea>
      <label class="lbl">평가 기준 (선택)</label>
      <textarea class="txt" id="team_evaluation_criteria" rows="6">${escapeHtml(state.team.evaluation_criteria)}</textarea>
      <label class="lbl">협업 네트워크 (선택, JSON 배열)</label>
      <textarea class="txt mono-input" id="team_collaboration_edges" rows="3">${escapeHtml(state.team.collaboration_edges_json)}</textarea>
      ${blocks}
      <div class="row-actions">
        <button type="button" class="btn btn-ghost" id="btn-team-demo">데모 데이터 입력</button>
        <button type="button" class="btn btn-ghost" id="btn-team-add">멤버 추가</button>
        <button type="button" class="btn btn-ghost" id="btn-team-remove" ${state.team.members.length <= 1 ? "disabled" : ""}>마지막 멤버 제거</button>
        <button type="button" class="btn btn-primary" id="btn-team-run" ${state.team.loading ? "disabled" : ""}>
          ${state.team.loading ? "자동 평가 중…" : "자동 평가 실행"}
        </button>
      </div>
      ${state.team.error ? `<p class="err">${escapeHtml(state.team.error)}</p>` : ""}
    </section>
    ${
      state.team.loading
        ? `<div class="team-loading hud-panel" aria-live="polite" aria-busy="true">
      <div class="skeleton-line skeleton-line--long"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line skeleton-line--med"></div>
      <p class="muted small">다중 분석·피드백 생성 중입니다. 팀 규모·API에 따라 수십 초 걸릴 수 있습니다.</p>
    </div>`
        : ""
    }
    ${teamReportHtml()}
  </div>`;
}

function atRiskResultHtml(): string {
  const r = state.atRisk.result;
  if (!r) return "";
  const sig = r.signals.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  const verdict = r.dropout_risk >= 50 ? "위험" : "정상";
  const verdictPill =
    verdict === "위험"
      ? `<span class="pill pill-warn" style="font-size:1.1rem;padding:0.35rem 0.75rem;">${verdict}</span>`
      : `<span class="pill pill-on" style="font-size:1.1rem;padding:0.35rem 0.75rem;">${verdict}</span>`;
  return `
  <section class="panel panel-result hud-panel">
    <h2 style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">판정 ${verdictPill}
      <span class="muted small">참고 지수 ${r.dropout_risk.toFixed(1)} / 100 · ${escapeHtml(r.mode)}</span>
    </h2>
    <p class="prose">${escapeHtml(r.trend_summary)}</p>
    <h4>신호</h4>
    <ul class="prose">${sig}</ul>
    <details class="at-risk-details">
      <summary>상세 (개입 제안 초안)</summary>
      <p class="prose">${escapeHtml(r.intervention_suggestions)}</p>
    </details>
    <p class="footer-note muted small">${escapeHtml(r.disclaimer || "")}</p>
  </section>`;
}

function atRiskHtml(): string {
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">조기 경보</p>
    <h1 class="page-title">학습 이탈 신호</h1>
    <p class="lead analyze-lead">LMS에서 가져올 수 있는 <strong>출석 비율</strong>과 <strong>과제 제출 비율</strong>만 넣으면 결과는 <strong>정상 / 위험</strong> 두 단계로 표시됩니다.</p>
    <section class="panel hud-panel">
      <div class="grid-2">
        <div><label class="lbl">과목 (선택)</label><input class="txt" id="ar_course" value="${escapeHtml(state.atRisk.course_name)}" /></div>
        <div><label class="lbl">학습자 라벨 (선택)</label><input class="txt" id="ar_student" value="${escapeHtml(state.atRisk.student_label)}" /></div>
      </div>
      <div class="grid-2">
        <div>
          <label class="lbl">출석·출첵 비율 %</label>
          <input class="txt" id="ar_attendance" type="number" min="0" max="100" step="0.1" placeholder="0–100" value="${escapeHtml(state.atRisk.attendance_ratio)}" />
        </div>
        <div>
          <label class="lbl">과제 제출 비율 %</label>
          <input class="txt" id="ar_assignment" type="number" min="0" max="100" step="0.1" placeholder="0–100" value="${escapeHtml(state.atRisk.assignment_ratio)}" />
        </div>
      </div>
      <label class="lbl">추가 메모 (선택)</label>
      <textarea class="txt" id="ar_notes" rows="2">${escapeHtml(state.atRisk.notes)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-ar-run" ${state.atRisk.loading ? "disabled" : ""}>
          ${state.atRisk.loading ? "분석 중…" : "판정 실행"}
        </button>
      </div>
      ${state.atRisk.error ? `<p class="err">${escapeHtml(state.atRisk.error)}</p>` : ""}
    </section>
    ${atRiskResultHtml()}
  </div>`;
}

function feedbackHtml(): string {
  const r = state.feedback.result;
  const resultBlock = r
    ? `
  <section class="panel panel-result hud-panel">
    <h2>피드백 초안</h2>
    <p class="prose">${escapeHtml(r.draft_feedback)}</p>
    <h4>강점</h4>
    <ul>${r.strengths.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    <h4>개선</h4>
    <ul>${r.improvements.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    <p class="footer-note muted small">${escapeHtml(r.disclaimer || "")}</p>
  </section>`
    : "";
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">과제</p>
    <h1 class="page-title">피드백 초안 생성</h1>
    <p class="lead analyze-lead"><code>OPENAI_API_KEY</code>가 필요합니다.</p>
    <section class="panel hud-panel">
      <label class="lbl">루브릭·채점 기준</label>
      <textarea class="txt" id="fb_rubric" rows="4">${escapeHtml(state.feedback.rubric)}</textarea>
      <label class="lbl">과제 설명 (선택)</label>
      <textarea class="txt" id="fb_prompt" rows="3">${escapeHtml(state.feedback.assignment_prompt)}</textarea>
      <label class="lbl">학생 제출물</label>
      <textarea class="txt" id="fb_submission" rows="8">${escapeHtml(state.feedback.submission)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-fb-run" ${state.feedback.loading ? "disabled" : ""}>
          ${state.feedback.loading ? "생성 중…" : "초안 생성"}
        </button>
      </div>
      ${state.feedback.error ? `<p class="err">${escapeHtml(state.feedback.error)}</p>` : ""}
    </section>
    ${resultBlock}
  </div>`;
}

function rubricGenResultHtml(): string {
  const r = state.rubricGen.result;
  if (!r) return "";
  const rows = r.criteria
    .map(
      (c) =>
        `<tr><td>${escapeHtml(c.name)}</td><td class="muted small">${escapeHtml(c.description)}</td><td>${c.weight_percent.toFixed(1)}%</td></tr>`
    )
    .join("");
  return `
  <section class="panel panel-result hud-panel">
    <div class="result-toolbar no-print" role="toolbar" aria-label="루브릭 초안 내보내기">
      <button type="button" class="btn btn-ghost btn-sm" id="btn-rg-export-json">JSON 내보내기</button>
      <button type="button" class="btn btn-ghost btn-sm" id="btn-rg-copy-md">마크다운 복사</button>
    </div>
    <h2>루브릭 초안 <span class="pill ${r.mode === "ai" ? "pill-on" : "pill-muted"}">${escapeHtml(r.mode)}</span></h2>
    <div class="table-scroll-wrap">
      <table class="data-table"><thead><tr><th>항목</th><th>기준</th><th>배점</th></tr></thead><tbody>${rows}</tbody></table>
    </div>
    <h3 class="subh">마크다운 전문</h3>
    <pre class="rubric-md-preview mono-input">${escapeHtml(r.rubric_markdown)}</pre>
    <p class="footer-note muted small">${escapeHtml(r.disclaimer || "")}</p>
  </section>`;
}

function rubricAlignHtml(): string {
  const r = state.rubricAlign.result;
  const resultBlock = r
    ? `
  <section class="panel panel-result hud-panel">
    <h2>정합성 <span class="score-num">${r.alignment_score.toFixed(1)}</span> / 100
      <span class="pill ${r.mode === "ai" ? "pill-on" : "pill-muted"}">${escapeHtml(r.mode)}</span></h2>
    <h4>잘 맞는 루브릭 요소</h4>
    <ul>${r.matched_rubric_points.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    <h4>격차·주의</h4>
    <ul>${r.gaps.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    <h4>제안</h4>
    <p class="prose">${escapeHtml(r.suggestions)}</p>
    <p class="footer-note muted small">${escapeHtml(r.disclaimer || "")}</p>
  </section>`
    : "";
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">채점 품질</p>
    <h1 class="page-title">루브릭·채점 근거 일치 점검</h1>
    <p class="lead analyze-lead">학습 목표에서 <strong>루브릭 초안</strong>을 만들거나, 기존 루브릭과 채점 코멘트의 <strong>정합성</strong>을 점검할 수 있습니다. OpenAI가 있으면 서술 품질이 올라갑니다.</p>
    <section class="panel hud-panel">
      <h4 class="subh">1) 학습 목표 → 루브릭 초안</h4>
      <div class="grid-2">
        <div><label class="lbl">과목명 (선택)</label><input class="txt" id="rg_course" value="${escapeHtml(state.rubricGen.course_name)}" placeholder="예: 소프트웨어공학" /></div>
        <div><label class="lbl">과제 유형 (선택)</label><input class="txt" id="rg_assignment" value="${escapeHtml(state.rubricGen.assignment_type)}" placeholder="팀 프로젝트, 보고서…" /></div>
      </div>
      <div class="grid-2">
        <div>
          <label class="lbl">항목 수 (3–8)</label>
          <input class="txt" id="rg_max_criteria" type="number" min="3" max="8" step="1" value="${escapeHtml(state.rubricGen.max_criteria)}" />
        </div>
      </div>
      <label class="lbl">학습 목표·성과 목표 (10자 이상)</label>
      <textarea class="txt" id="rg_objectives" rows="5" placeholder="예: REST API 설계·문서화, 보안·오류 처리 등">${escapeHtml(state.rubricGen.learning_objectives)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-rg-run" ${state.rubricGen.loading ? "disabled" : ""}>
          ${state.rubricGen.loading ? "생성 중…" : "루브릭 초안 생성"}
        </button>
      </div>
      ${state.rubricGen.error ? `<p class="err">${escapeHtml(state.rubricGen.error)}</p>` : ""}
    </section>
    ${rubricGenResultHtml()}
    <section class="panel hud-panel">
      <h4 class="subh">2) 루브릭·채점 근거 정합성 점검</h4>
      <label class="lbl">루브릭 전문</label>
      <textarea class="txt" id="ra_rubric" rows="6">${escapeHtml(state.rubricAlign.rubric)}</textarea>
      <label class="lbl">채점 근거·코멘트</label>
      <textarea class="txt" id="ra_rationale" rows="5">${escapeHtml(state.rubricAlign.grader_rationale)}</textarea>
      <label class="lbl">학생 답안 발췌 (선택)</label>
      <textarea class="txt" id="ra_student" rows="4">${escapeHtml(state.rubricAlign.student_work)}</textarea>
      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-ra-run" ${state.rubricAlign.loading ? "disabled" : ""}>
          ${state.rubricAlign.loading ? "점검 중…" : "정합성 점검"}
        </button>
      </div>
      ${state.rubricAlign.error ? `<p class="err">${escapeHtml(state.rubricAlign.error)}</p>` : ""}
    </section>
    ${resultBlock}
  </div>`;
}

/** 과정·시험 불일치 방향을 한 줄로 (모델 서술 기반, 참고용). */
function inferProcessExamTriage(res: AnalyzeResponse): "정상" | "과제 과대" | "시험 과대" {
  const blob = [
    res.consensus_summary,
    ...res.judgments.filter((j) => j.ok).map((j) => j.mismatch_analysis),
  ].join("\n");
  if (/시험.*(높|우수|양호|과대|만점|상위)|과정.*(낮|부족|미흡|하락)/.test(blob)) return "시험 과대";
  if (/과제.*(높|우수|양호|과대)|과정.*(높|우수).*시험.*(낮|부족|미흡)/.test(blob)) return "과제 과대";
  return "정상";
}

function resultSectionHtml(): string {
  if (!state.result) return "";
  const res = state.result;
  const triage = inferProcessExamTriage(res);
  const triPill =
    triage === "정상"
      ? "pill-on"
      : triage === "과제 과대"
        ? "pill-muted"
        : "pill-warn";
  const triageLine = `<p class="analyze-triage-line"><span class="pill ${triPill}">요약: ${triage}</span> <span class="muted small">과정 vs 시험 불일치 방향(참고)</span></p>`;
  const avg =
    res.consensus_cheating_avg != null
      ? `<p class="consensus"><strong>합의 평균 부정행위 의심도:</strong> ${res.consensus_cheating_avg.toFixed(1)} / 100</p>`
      : "";
  const cards = res.judgments
    .map((j) => {
      if (!j.ok) {
        return `<div class="card card-err"><h3>${escapeHtml(j.provider)}</h3><p class="err">${escapeHtml(j.error || "오류")}</p></div>`;
      }
      const score =
        j.cheating_likelihood != null
          ? `<div class="score-line">부정행위 의심도 <span class="score-num">${j.cheating_likelihood.toFixed(1)}</span> / 100</div>`
          : "";
      return `<div class="card">
        <div class="card-head">
          <h3>${escapeHtml(j.provider)}</h3>
          <span class="model-tag">${escapeHtml(j.model_label || "")}</span>
        </div>
        ${score}
        <h4>학습 상태</h4>
        <p class="prose">${escapeHtml(j.learning_state_summary)}</p>
        <h4>과정–시험 불일치</h4>
        <p class="prose">${escapeHtml(j.mismatch_analysis)}</p>
        <h4>미래 예측·위험</h4>
        <p class="prose future">${escapeHtml(j.future_prediction)}</p>
        <p class="muted small">${escapeHtml(j.confidence_note)}</p>
      </div>`;
    })
    .join("");
  const skippedHtml =
    res.providers_skipped.length > 0
      ? `<p class="skipped muted small">건너뜀: ${escapeHtml(res.providers_skipped.join(" · "))}</p>`
      : "";
  return `
  <section class="panel panel-result hud-panel" id="results">
    <div class="result-toolbar no-print" role="toolbar" aria-label="결과 내보내기">
      <button type="button" class="btn btn-ghost btn-sm" id="btn-analyze-export-json">JSON 내보내기</button>
    </div>
    <h2>분석 결과</h2>
    ${triageLine}
    <p class="prose">${escapeHtml(res.consensus_summary)}</p>
    ${avg}
    ${skippedHtml}
    <div class="card-grid">${cards}</div>
    <p class="footer-note">${escapeHtml(res.disclaimer)}</p>
  </section>`;
}

function analyzeHtml(): string {
  return `
  <div class="page page-animate analyze-page">
    <p class="eyebrow">보조 · 과정 vs 시험</p>
    <h1 class="page-title">과정 지표와 시험 점수</h1>
    <p class="lead analyze-lead">LMS·과제 지표와 시험 점수를 넣으면 <strong>불일치 방향</strong>을 참고용으로 요약합니다.</p>
    <section class="panel hud-panel">
      <div class="grid-2">
        <div>
          <label class="lbl">과목명 (선택)</label>
          <input class="txt" id="course_name" value="${escapeHtml(state.course_name)}" />
        </div>
        <div>
          <label class="lbl">식별 라벨 (익명, 선택)</label>
          <input class="txt" id="student_or_group_label" value="${escapeHtml(state.student_or_group_label)}" placeholder="예: 학습자-042" />
        </div>
      </div>
      <h3 class="subh">학습 과정</h3>
      <div class="grid-3">
        <div>
          <label class="lbl">주간 학습시간 (자기보고)</label>
          <input class="txt" id="weekly_study_hours" type="number" step="0.1" min="0" value="${escapeHtml(state.weekly_study_hours)}" />
        </div>
        <div>
          <label class="lbl">강의 시청률 %</label>
          <input class="txt" id="lms_video_watch_ratio" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.lms_video_watch_ratio)}" />
        </div>
        <div>
          <label class="lbl">캐퀴즈·소평균 %</label>
          <input class="txt" id="quiz_average" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.quiz_average)}" />
        </div>
        <div>
          <label class="lbl">과제 기한 내 제출 비율 %</label>
          <input class="txt" id="assignment_on_time_ratio" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.assignment_on_time_ratio)}" />
        </div>
        <div>
          <label class="lbl">토론·포럼 참여 횟수</label>
          <input class="txt" id="discussion_count" type="number" min="0" value="${escapeHtml(state.discussion_count)}" />
        </div>
        <div>
          <label class="lbl">출석·출첵 비율 %</label>
          <input class="txt" id="attendance_ratio" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.attendance_ratio)}" />
        </div>
      </div>
      <label class="lbl">학습 과정 메모</label>
      <textarea class="txt" id="learning_notes" rows="2">${escapeHtml(state.learning_notes)}</textarea>

      <h3 class="subh">시험·평가</h3>
      <div class="grid-2">
        <div>
          <label class="lbl">중간고사 %</label>
          <input class="txt" id="midterm_score" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.midterm_score)}" />
        </div>
        <div>
          <label class="lbl">기말/최근 시험 %</label>
          <input class="txt" id="final_score" type="number" step="0.1" min="0" max="100" value="${escapeHtml(state.final_score)}" />
        </div>
      </div>
      <label class="lbl">시험 시간·제출 이상 (선택)</label>
      <input class="txt" id="exam_time_anomaly_note" value="${escapeHtml(state.exam_time_anomaly_note)}" placeholder="예: 매우 짧은 시간에 고득점" />
      <label class="lbl">시험 메모</label>
      <textarea class="txt" id="exam_notes" rows="2">${escapeHtml(state.exam_notes)}</textarea>

      <label class="lbl">교수·조교 맥락 (선택)</label>
      <textarea class="txt" id="context_for_educator" rows="2" placeholder="난이도, 오픈북 여부 등">${escapeHtml(state.context_for_educator)}</textarea>

      <div class="row-actions">
        <button type="button" class="btn btn-primary" id="btn-analyze" ${state.loading ? "disabled" : ""}>
          ${state.loading ? "분석 중…" : "불일치 분석 실행"}
        </button>
      </div>
      ${state.error ? `<p class="err">${escapeHtml(state.error)}</p>` : ""}
    </section>
    ${resultSectionHtml()}
  </div>`;
}

function mainContentHtml(): string {
  switch (state.view) {
    case "home":
      return homeHtml();
    case "hub":
      return hubHtml();
    case "team":
      return teamHtml();
    case "at-risk":
      return atRiskHtml();
    case "feedback":
      return feedbackHtml();
    case "rubric":
      return rubricAlignHtml();
    case "analyze":
      return analyzeHtml();
    default:
      return homeHtml();
  }
}

let renderRafId = 0;

/** 즉시 DOM 반영 — 로딩 표시·비동기 완료 후에는 이쪽(다음 프레임 지연 없음) */
function renderSync(): void {
  if (renderRafId) {
    cancelAnimationFrame(renderRafId);
    renderRafId = 0;
  }
  const app = document.getElementById("app");
  if (!app) return;
  app.innerHTML = `
  <div class="app-shell app-shell--fx">
    ${navHtml()}
    ${connectionStripHtml()}
    <main class="site-main site-main--fx">${mainContentHtml()}</main>
    ${footerHtml()}
  </div>`;
  wire();
}

/** 동기 스택에서 연속 호출 시 한 프레임으로 합쳐 reflow 비용 절감 */
function render(): void {
  if (renderRafId) return;
  renderRafId = requestAnimationFrame(() => {
    renderRafId = 0;
    renderSync();
  });
}

function readForm(): void {
  const g = (id: string) => document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | null;
  state.course_name = g("course_name")?.value ?? "";
  state.student_or_group_label = g("student_or_group_label")?.value ?? "";
  state.weekly_study_hours = g("weekly_study_hours")?.value ?? "";
  state.lms_video_watch_ratio = g("lms_video_watch_ratio")?.value ?? "";
  state.quiz_average = g("quiz_average")?.value ?? "";
  state.assignment_on_time_ratio = g("assignment_on_time_ratio")?.value ?? "";
  state.discussion_count = g("discussion_count")?.value ?? "";
  state.attendance_ratio = g("attendance_ratio")?.value ?? "";
  state.learning_notes = g("learning_notes")?.value ?? "";
  state.midterm_score = g("midterm_score")?.value ?? "";
  state.final_score = g("final_score")?.value ?? "";
  state.exam_time_anomaly_note = g("exam_time_anomaly_note")?.value ?? "";
  state.exam_notes = g("exam_notes")?.value ?? "";
  state.context_for_educator = g("context_for_educator")?.value ?? "";
}

function setView(v: SiteView): void {
  if (state.view === "analyze") {
    readForm();
  }
  if (state.view === "team") {
    readTeamForm();
  }
  if (state.view === "at-risk") {
    readAtRiskForm();
  }
  if (state.view === "feedback") {
    readFeedbackForm();
  }
  if (state.view === "rubric") {
    readRubricAlignForm();
    readRubricGenForm();
  }
  state.view = v;
  if (v === "team") {
    hydrateTeamDraftIfEmpty();
  }
  const nextHash = v === "home" ? "#/" : `#/${v}`;
  if (location.hash !== nextHash) {
    history.replaceState(null, "", `${location.pathname}${location.search}${nextHash}`);
  }
  void refreshHealth(true).then(() => {
    renderSync();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function wire(): void {
  document.querySelectorAll("[data-view]").forEach((el) => {
    el.addEventListener("click", (e) => {
      const v = (el as HTMLElement).dataset.view as SiteView | undefined;
      if (!v || v === state.view) {
        if ((el as HTMLElement).classList.contains("brand")) e.preventDefault();
        return;
      }
      e.preventDefault();
      setView(v);
    });
  });

  document.getElementById("btn-analyze")?.addEventListener("click", () => {
    readForm();
    void submitAnalyze();
  });

  document.getElementById("btn-team-demo")?.addEventListener("click", () => {
    applyDemoTeamData();
    state.view = "team";
    render();
    scheduleTeamDraftSave();
  });

  document.getElementById("btn-team-add")?.addEventListener("click", () => {
    readTeamForm();
    state.team.members.push(emptyTeamMember());
    state.view = "team";
    render();
  });

  document.getElementById("btn-team-remove")?.addEventListener("click", () => {
    readTeamForm();
    if (state.team.members.length > 1) state.team.members.pop();
    state.view = "team";
    render();
  });

  document.getElementById("btn-team-run")?.addEventListener("click", () => {
    void submitTeam();
  });

  document.querySelector(".team-form-panel")?.addEventListener("input", scheduleTeamDraftSave);
  document.querySelector(".team-form-panel")?.addEventListener("change", scheduleTeamDraftSave);

  document.getElementById("btn-team-export-json")?.addEventListener("click", () => {
    const r = state.team.report;
    if (!r) return;
    const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json;charset=utf-8" });
    const safe = (state.team.project_name || "team-eval").replace(/[\\/:*?"<>|]/g, "_").slice(0, 80);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${safe}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  });

  document.getElementById("btn-team-export-evidence")?.addEventListener("click", () => {
    const r = state.team.report;
    if (!r) return;
    const payload = buildTeamEvidencePackage(r);
    const safe = (state.team.project_name || "team-evidence").replace(/[\\/:*?"<>|]/g, "_").slice(0, 80);
    downloadJsonExport(`${safe}-judge-evidence`, payload);
  });

  document.getElementById("btn-team-copy-summary")?.addEventListener("click", async () => {
    const r = state.team.report;
    if (!r) return;
    const text = teamUnifiedExportSummaryText(r);
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      /* ignore */
    }
  });

  document.getElementById("btn-team-print")?.addEventListener("click", () => {
    window.print();
  });

  document.getElementById("btn-copy-promote-hint")?.addEventListener("click", async () => {
    const rep = state.team.report;
    if (!rep) return;
    const pg = (rep.dl_model_info as { quality?: { promotion_gate?: PromotionGateInfo } } | undefined)
      ?.quality?.promotion_gate;
    if (!pg || pg.accepted) return;
    const rs = (pg.reasons || []).join(", ");
    const msg = [
      "[DL 재학습 권고]",
      "신규 모델 승격이 차단되었습니다.",
      `사유: ${rs || "metrics_regressed"}`,
      "권고: 학습 데이터 품질(결측/이상치), 홀드아웃 분포, 하이퍼파라미터 범위를 재점검 후 재학습하세요.",
    ].join("\n");
    try {
      await navigator.clipboard.writeText(msg);
    } catch {
      /* ignore */
    }
  });

  document.getElementById("btn-team-recheck")?.addEventListener("click", () => {
    void runTeamRecheck();
  });

  document.getElementById("btn-trend-refresh")?.addEventListener("click", () => {
    const d = document.getElementById("trend-days") as HTMLSelectElement | null;
    const m = document.getElementById("trend-member") as HTMLSelectElement | null;
    if (d) state.team.trends_days = parseInt(d.value, 10) || 30;
    if (m) state.team.trends_member = m.value;
    void fetchTeamTrends();
  });

  document.getElementById("whatif-member")?.addEventListener("change", () => {
    const v = (document.getElementById("whatif-member") as HTMLSelectElement | null)?.value ?? "";
    state.team.what_if.member_name = v;
    renderSync();
  });
  document.getElementById("whatif-commits")?.addEventListener("input", () => {
    const v = parseInt((document.getElementById("whatif-commits") as HTMLInputElement | null)?.value ?? "0", 10);
    state.team.what_if.add_commits = Number.isFinite(v) ? Math.max(0, v) : 0;
    renderSync();
  });
  document.getElementById("whatif-prs")?.addEventListener("input", () => {
    const v = parseInt((document.getElementById("whatif-prs") as HTMLInputElement | null)?.value ?? "0", 10);
    state.team.what_if.add_prs = Number.isFinite(v) ? Math.max(0, v) : 0;
    renderSync();
  });
  document.getElementById("whatif-lines")?.addEventListener("input", () => {
    const v = parseInt((document.getElementById("whatif-lines") as HTMLInputElement | null)?.value ?? "0", 10);
    state.team.what_if.add_lines = Number.isFinite(v) ? Math.max(0, v) : 0;
    renderSync();
  });

  document.getElementById("btn-ar-run")?.addEventListener("click", () => {
    void submitAtRisk();
  });

  document.getElementById("btn-fb-run")?.addEventListener("click", () => {
    void submitFeedback();
  });

  document.getElementById("btn-ra-run")?.addEventListener("click", () => {
    void submitRubricAlign();
  });

  document.getElementById("btn-rg-run")?.addEventListener("click", () => {
    void submitRubricGenerate();
  });

  document.getElementById("btn-rg-export-json")?.addEventListener("click", () => {
    const x = state.rubricGen.result;
    if (!x) return;
    downloadJsonExport("rubric-draft", x);
  });

  document.getElementById("btn-rg-copy-md")?.addEventListener("click", async () => {
    const x = state.rubricGen.result;
    if (!x) return;
    try {
      await navigator.clipboard.writeText(x.rubric_markdown);
    } catch {
      /* ignore */
    }
  });

  document.getElementById("btn-analyze-export-json")?.addEventListener("click", () => {
    if (!state.result) return;
    downloadJsonExport("analyze-learning-exam", state.result);
  });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      void refreshHealth(true).then(() => renderSync());
    }
  });

  window.addEventListener("online", () => {
    connectionUnstable = false;
    void refreshHealth(true).then(() => renderSync());
  });
}

/** URL 해시로 뷰 복원  해시 없으면 홈. */
function applyHashRouteOnce(): void {
  const h = (location.hash || "").replace(/^#\/?/, "").split(/[/?]/)[0]?.toLowerCase() ?? "";
  if (!h) return;
  const map: Record<string, SiteView> = {
    home: "home",
    team: "team",
    hub: "hub",
    analyze: "analyze",
    "at-risk": "at-risk",
    feedback: "feedback",
    rubric: "rubric",
  };
  const v = map[h];
  if (v) state.view = v;
}

/** 백엔드 health 전에도 셸을 먼저 그려 빈 화면을 막음 */
function boot(): void {
  applyHashRouteOnce();
  renderSync();
  startHealthKeepAlive();
  void refreshHealth(true).then(() => renderSync());
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}

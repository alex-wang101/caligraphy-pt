export type Task = {
  task_id: string;
  character: string;
  pinyin: string;
  definition: string;
  stroke_count: number;
  style_difficulty: number;
  calligrapher: string;
  calligrapher_label: string;
};

export type SearchHit = { character: string; pinyin: string[]; definition: string };

export type StrokeMetrics = {
  inside_frac: number;
  iou: number;
  frechet: number;
  frechet_reversed: number;
  direction_ok: boolean;
};

export type Levels = {
  placement_offset: number | null;
  placement_ratio: number;
  placement_ok: boolean;
  structure: number;
  shape: number;
  structure_strokes: number[];
  shape_strokes: number[];
  n_agent: number;
  n_target: number;
  align_scale: number;
  align_offset: [number, number];
  align_inliers: boolean[];
};

export type CharData = {
  key: string;
  character: string;
  task_id: string | null;
  meta: Record<string, unknown>;
  n_strokes: number;
  has_reference: boolean;
  outlines: string[];
  medians: [number, number][][];
  drawing: string;
  drawings: { name: string; label: string }[];
  clamped_points: number;
  strokes: { points: [number, number, number][]; width: number }[];
  stroke_metrics: StrokeMetrics[];
  levels: Levels;
  score_ms: number;
  rollout: Rollout | null;
};

export type Rollout = {
  run_id: string;
  index: number;
  model: string;
  provider: string;
  response: string;
  reward: number;
  parse_ok: boolean;
  error: string | null;
  latency_s: number;
  stop_reason: string | null;
  mode: "single" | "tools";
  trace_summary: TraceSummary | null;
};

/** One multi-turn episode in numbers (tools mode). */
export type TraceSummary = {
  end_reason: string;
  n_turns: number;
  n_tool_calls: number;
  calls: Record<string, number>;
  n_failed_calls: number;
  n_skipped_calls: number;
  n_captures: number;
  n_deletes: number;
  n_redraws: number;
  n_nudges: number;
  n_strokes: number;
  checked_final: boolean;
  final_score: number | null;
  peak_score: number | null;
  draft_score: number | null;
  duration_s: number;
};

export type StepScore = { structure: number; shape: number; strokes: Record<string, number> };

export type ModelEvent = {
  i: number;
  type: "model";
  t_s: number;
  turn: number;
  text: string;
  reasoning: string;
  stop_reason: string | null;
  latency_s: number;
  usage: Record<string, unknown>;
  n_tool_calls: number;
  error: string | null;
};

export type ToolEvent = {
  i: number;
  type: "tool";
  t_s: number;
  turn: number;
  call_id: string | null;
  name: string;
  arguments: Record<string, unknown> | null;
  arguments_raw?: string;
  ok: boolean;
  skipped?: boolean;
  result: string;
  stroke_id?: number | null;
  stroke_ids?: number[];
  redraw?: boolean;
  deleted_stroke_score?: number | null;
  score?: StepScore | null;
  image?: string;
  image_size?: number;
};

export type TraceEvent =
  | ModelEvent
  | ToolEvent
  | { i: number; type: "nudge"; t_s: number; turn: number; text: string }
  | { i: number; type: "end"; t_s: number; reason: string };

export type Trace = {
  index: number;
  character: string;
  model: string;
  system: string;
  user: string;
  events: TraceEvent[];
  summary: TraceSummary;
  reward: number;
};

export type NormCase = {
  key: string;
  character: string;
  drawing: string;
  label: string;
  category: string;
  reward: number;
  shape: number;
  placement_ok: boolean;
  clamped_points: number;
};

export type Provider = {
  id: string;
  label: string;
  kind: "openai" | "anthropic";
  base_url: string;
  key_env: string | null;
  key_env_set: boolean;
  model_hint: string;
  local: boolean;
  steps: string[];
};

export type DatasetOption = { id: string; label: string };

export type RunSummary = {
  id: string;
  created_at: string;
  finished_at?: string;
  status: "running" | "done" | "cancelled" | "error" | "interrupted" | "rate_limited";
  provider: string;
  model: string;
  base_url: string;
  api_key_source: string;
  dataset: string;
  dataset_label: string;
  n_total: number;
  n_done: number;
  n_parse_ok: number;
  n_errors: number;
  mean_reward: number | null;
  elapsed_s: number;
  error?: string;
  mode?: "single" | "tools";
};

export type RunRow = {
  index: number;
  key: string;
  character: string;
  reward: number;
  parse_ok: boolean;
  error: string | null;
  latency_s: number;
  trace_summary: TraceSummary | null;
};

export type RunDetail = { summary: RunSummary; rollouts: RunRow[] };

export type ModelRequest = {
  provider: string;
  model: string;
  base_url?: string;
  api_key?: string;
  temperature?: number | null;
  max_tokens?: number;
};

export type RunRequest = ModelRequest & {
  dataset: string;
  characters?: string;
  limit?: number | null;
  concurrency?: number;
  mode?: "single" | "tools";
};

export type TestResult = { ok: boolean; text: string; error: string | null; latency_s: number; key_source: string };

/** Judge experiments: a model scores drawings of known quality (scripts/run_judge.py). */
export type JudgeKind = "correct" | "flawed" | "wrong character" | "model answer";

export type Judgement = {
  score: number | null;
  recognized_as: string;
  is_correct_character: boolean | null;
  problems: string;
  parse_ok: boolean;
};

export type JudgeItem = {
  index: number;
  key: string;
  character: string;
  variant: string;
  label: string;
  kind: JudgeKind;
  source: string | null;
  reward: number;
  shape: number;
  n_strokes: number;
  n_target: number;
  reply: string;
  error: string | null;
  latency_s: number;
  judge: Judgement | null;
};

export type JudgeVariantRow = {
  variant: string;
  label: string;
  kind: JudgeKind;
  n: number;
  reward_mean: number;
  judge_mean: number;
  says_correct: number | null;
};

export type JudgeStats = {
  n: number;
  n_scored: number;
  spearman: number | null;
  pearson: number | null;
  mae: number | null;
  auc_good_bad: number | null;
  n_good: number;
  n_bad: number;
  judge_mean_good: number | null;
  judge_mean_bad: number | null;
  reads_correct: number | null;
  accepts_correct: number | null;
  rejects_wrong: number | null;
  by_variant: JudgeVariantRow[];
};

export type JudgeSummary = {
  id: string;
  created_at: string;
  finished_at?: string;
  status: "running" | "done" | "error" | "interrupted";
  provider: string;
  model: string;
  temperature: number | null;
  characters: string;
  obs_size: number;
  n_total: number;
  n_done: number;
  n_errors: number;
  elapsed_s?: number;
  error?: string;
  prompt: string;
  stats?: JudgeStats;
};

export type JudgeDetail = { summary: JudgeSummary; items: JudgeItem[] };

export type RenderOpts = { drawing: string; brush: string; size: number; grid: boolean };

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    const body = await r.text();
    let detail = body;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {
      /* not JSON */
    }
    if (r.status === 404 && url.startsWith("/api/")) detail = STALE_SERVER;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  // An API server started before this route existed answers with the page's HTML.
  if (!(r.headers.get("content-type") ?? "").includes("application/json")) throw new Error(STALE_SERVER);
  return r.json() as Promise<T>;
}

const STALE_SERVER =
  "The API server is running older code than this page. Restart it: press Ctrl+C in its terminal, then run `uv run uvicorn server.app:app --port 8000`.";

const post = <T,>(url: string, body: unknown) =>
  json<T>(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  tasks: () => json<Task[]>("/api/tasks"),
  search: (q: string) => json<SearchHit[]>(`/api/search?q=${encodeURIComponent(q)}`),
  char: (key: string, drawing: string, brush: string) =>
    json<CharData>(`/api/char/${encodeURIComponent(key)}?drawing=${encodeURIComponent(drawing)}&brush=${brush}`),
  referenceUrl: (key: string) => `/api/char/${encodeURIComponent(key)}/reference.jpg`,
  renderUrl: (key: string, upto: number, o: RenderOpts) =>
    `/api/char/${encodeURIComponent(key)}/render.png?upto=${upto}&size=${o.size}&brush=${o.brush}&grid=${o.grid}&drawing=${encodeURIComponent(o.drawing)}`,
  normalization: () => json<NormCase[]>("/api/normalization"),
  providers: () => json<Provider[]>("/api/agent/providers"),
  datasets: () => json<DatasetOption[]>("/api/agent/datasets"),
  testModel: (req: ModelRequest) => post<TestResult>("/api/agent/test", req),
  startRun: (req: RunRequest) => post<{ run_id: string }>("/api/agent/runs", req),
  runs: () => json<RunSummary[]>("/api/agent/runs"),
  run: (id: string) => json<RunDetail>(`/api/agent/runs/${encodeURIComponent(id)}`),
  cancelRun: (id: string) => post<{ ok: boolean }>(`/api/agent/runs/${encodeURIComponent(id)}/cancel`, {}),
  trace: (runId: string, index: number) => json<Trace>(`/api/agent/runs/${encodeURIComponent(runId)}/trace/${index}`),
  /** The canvas as it stood after event `step` (-1: empty). `mark` draws one stroke in colour. */
  traceStateUrl: (runId: string, index: number, step: number, mark?: { stroke: number; tone: "added" | "removed" }) =>
    `/api/agent/runs/${encodeURIComponent(runId)}/trace/${index}/state?step=${step}&size=320` +
    (mark ? `&mark=${mark.stroke}&tone=${mark.tone}` : ""),
  /** An image exactly as the model was shown it. */
  traceImageUrl: (runId: string, index: number, name: string) =>
    `/api/agent/runs/${encodeURIComponent(runId)}/trace/${index}/image/${encodeURIComponent(name)}`,
  judgeList: () => json<JudgeSummary[]>("/api/judge"),
  judge: (id: string) => json<JudgeDetail>(`/api/judge/${encodeURIComponent(id)}`),
  judgeImageUrl: (id: string, index: number) => `/api/judge/${encodeURIComponent(id)}/image/${index}.png`,
};

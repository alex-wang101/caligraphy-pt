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
  status: "running" | "done" | "cancelled" | "error" | "interrupted";
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
};

export type RunRow = {
  index: number;
  key: string;
  character: string;
  reward: number;
  parse_ok: boolean;
  error: string | null;
  latency_s: number;
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

export type RunRequest = ModelRequest & { dataset: string; characters?: string; limit?: number | null; concurrency?: number };

export type TestResult = { ok: boolean; text: string; error: string | null; latency_s: number; key_source: string };

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
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json() as Promise<T>;
}

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
};

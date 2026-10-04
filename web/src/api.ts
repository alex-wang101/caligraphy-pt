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
  placement_offset: number;
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
  drawings: string[];
  strokes: { points: [number, number, number][]; width: number }[];
  stroke_metrics: StrokeMetrics[];
  levels: Levels;
  score_ms: number;
};

export type RenderOpts = { drawing: string; brush: string; size: number; grid: boolean };

async function json<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  tasks: () => json<Task[]>("/api/tasks"),
  search: (q: string) => json<SearchHit[]>(`/api/search?q=${encodeURIComponent(q)}`),
  char: (key: string, drawing: string, brush: string) =>
    json<CharData>(`/api/char/${encodeURIComponent(key)}?drawing=${drawing}&brush=${brush}`),
  referenceUrl: (key: string) => `/api/char/${encodeURIComponent(key)}/reference.jpg`,
  renderUrl: (key: string, upto: number, o: RenderOpts) =>
    `/api/char/${encodeURIComponent(key)}/render.png?upto=${upto}&size=${o.size}&brush=${o.brush}&grid=${o.grid}&drawing=${o.drawing}`,
};

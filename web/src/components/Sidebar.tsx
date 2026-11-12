import { useEffect, useMemo, useState } from "react";
import { api, JudgeSummary, NormCase, RunDetail, RunSummary, SearchHit, Task } from "../api";

export type Source = "bench" | "normalization" | "runs" | "judge";

type Props = {
  tasks: Task[];
  source: Source;
  onSource: (s: Source) => void;
  selectedKey: string;
  selectedDrawing: string;
  onSelect: (key: string, drawing: string) => void;
  onNewRun: () => void;
  activeRun: string | null;
  onActiveRun: (id: string | null) => void;
  activeJudge: string | null;
  onActiveJudge: (id: string | null) => void;
};

export function Sidebar(props: Props) {
  const { source, onSource } = props;
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-glyph">永</span>
        <div>
          <div className="brand-title">Calligraphy viewer</div>
          <div className="muted">calligraphy-bench · Make Me a Hanzi</div>
        </div>
      </div>
      <label className="source">
        <span>Dataset</span>
        <select value={source} onChange={(e) => onSource(e.target.value as Source)} aria-label="Dataset">
          <option value="bench">calligraphy-bench (88)</option>
          <option value="normalization">Normalization tests</option>
          <option value="runs">Agent runs</option>
          <option value="judge">Judge experiments</option>
        </select>
      </label>
      {source === "bench" && <BenchList {...props} />}
      {source === "normalization" && <NormalizationList {...props} />}
      {source === "runs" && <RunsList {...props} />}
      {source === "judge" && <JudgeList {...props} />}
    </aside>
  );
}

export function RewardPill({ value }: { value: number }) {
  const tone = value > 0.8 ? "good" : value > 0.5 ? "mid" : "bad";
  return <span className={`pill ${tone}`}>{value.toFixed(2)}</span>;
}

function BenchList({ tasks, selectedKey, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [calligrapher, setCalligrapher] = useState("");
  const [difficulty, setDifficulty] = useState(0);
  const [hits, setHits] = useState<SearchHit[]>([]);

  const calligraphers = useMemo(() => [...new Set(tasks.map((t) => t.calligrapher))].sort(), [tasks]);
  const filtered = tasks.filter(
    (t) =>
      (!calligrapher || t.calligrapher === calligrapher) &&
      (!difficulty || t.style_difficulty === difficulty) &&
      (!query || t.character.includes(query) || query.includes(t.character) || t.pinyin.includes(query.toLowerCase())),
  );

  useEffect(() => {
    if (!/\p{Script=Han}/u.test(query)) return setHits([]);
    const id = setTimeout(() => api.search(query).then(setHits).catch(() => setHits([])), 150);
    return () => clearTimeout(id);
  }, [query]);

  return (
    <>
      <input className="search" placeholder="Search character or pinyin" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search" />
      <div className="filters">
        <select value={calligrapher} onChange={(e) => setCalligrapher(e.target.value)} aria-label="Calligrapher">
          <option value="">All calligraphers</option>
          {calligraphers.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
        <select value={difficulty} onChange={(e) => setDifficulty(Number(e.target.value))} aria-label="Difficulty">
          <option value={0}>Any difficulty</option>
          {[1, 2, 3, 4, 5].map((d) => (
            <option key={d} value={d}>
              Difficulty {d}
            </option>
          ))}
        </select>
      </div>
      <div className="list">
        <div className="list-head">calligraphy-bench · {filtered.length}</div>
        {filtered.map((t) => (
          <button key={t.task_id} className={`task${selectedKey === t.task_id ? " active" : ""}`} onClick={() => onSelect(t.task_id, "converter")}>
            <span className="task-glyph">{t.character}</span>
            <span className="task-meta">
              <span>{t.calligrapher}</span>
              <span className="muted">
                {t.pinyin} · {t.stroke_count} strokes
              </span>
            </span>
            <span className="dots" aria-label={`difficulty ${t.style_difficulty}`}>
              {[1, 2, 3, 4, 5].map((d) => (
                <i key={d} className={d <= t.style_difficulty ? "on" : ""} />
              ))}
            </span>
          </button>
        ))}
        {hits.length > 0 && <div className="list-head">Make Me a Hanzi</div>}
        {hits.map((h) => (
          <button key={h.character} className={`task${selectedKey === h.character ? " active" : ""}`} onClick={() => onSelect(h.character, "converter")}>
            <span className="task-glyph">{h.character}</span>
            <span className="task-meta">
              <span>{h.pinyin.join(", ")}</span>
              <span className="muted ellipsis">{h.definition}</span>
            </span>
          </button>
        ))}
      </div>
    </>
  );
}

const CATEGORIES = ["baseline", "size", "position", "shape"];

function NormalizationList({ selectedKey, selectedDrawing, onSelect }: Props) {
  const [cases, setCases] = useState<NormCase[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [char, setChar] = useState("");
  const [category, setCategory] = useState("");

  useEffect(() => {
    api.normalization().then(setCases).catch((e) => setError(String(e)));
  }, []);

  const chars = useMemo(() => [...new Set((cases ?? []).map((c) => c.character))], [cases]);
  const shown = (cases ?? []).filter((c) => (!char || c.character === char) && (!category || c.category === category));

  return (
    <>
      <div className="filters">
        <select value={char} onChange={(e) => setChar(e.target.value)} aria-label="Character">
          <option value="">All characters</option>
          {chars.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
        <select value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Category">
          <option value="">All kinds</option>
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {c[0].toUpperCase() + c.slice(1)}
            </option>
          ))}
        </select>
      </div>
      <div className="list">
        {error && <div className="list-note bad">{error}</div>}
        {!cases && !error && <div className="list-note">Scoring scenarios…</div>}
        {cases && <div className="list-head">Normalization tests · {shown.length}</div>}
        {shown.map((c) => (
          <button
            key={`${c.character}-${c.drawing}`}
            className={`task case${selectedKey === c.key && selectedDrawing === c.drawing ? " active" : ""}`}
            onClick={() => onSelect(c.key, c.drawing)}
          >
            <span className="task-glyph">{c.character}</span>
            <span className="task-meta">
              <span className="ellipsis">{c.label}</span>
              <span className="muted">
                {c.category}
                {!c.placement_ok && " · placement flagged"}
                {c.clamped_points > 0 && ` · ${c.clamped_points} clamped`}
              </span>
            </span>
            <RewardPill value={c.reward} />
          </button>
        ))}
      </div>
    </>
  );
}

function RunsList({ selectedDrawing, onSelect, onNewRun, activeRun, onActiveRun }: Props) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The run list refreshes while any run is still going.
  useEffect(() => {
    let alive = true;
    const load = () => api.runs().then((r) => alive && setRuns(r)).catch((e) => alive && setError(String(e)));
    load();
    const id = setInterval(load, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    if (!activeRun) return setDetail(null);
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = () =>
      api
        .run(activeRun)
        .then((d) => {
          if (!alive) return;
          setDetail(d);
          if (d.summary.status === "running") timer = setTimeout(load, 1200);
        })
        .catch((e) => alive && setError(String(e)));
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [activeRun]);

  if (activeRun && detail) {
    const s = detail.summary;
    return (
      <>
        <button className="back" onClick={() => onActiveRun(null)}>
          ‹ All runs
        </button>
        <RunCard s={s} />
        {s.status === "running" && (
          <button className="secondary" onClick={() => api.cancelRun(s.id)}>
            Cancel run
          </button>
        )}
        <div className="list">
          <div className="list-head">
            Answers · {detail.rollouts.length}/{s.n_total}
          </div>
          {detail.rollouts.map((r) => {
            const drawing = `run:${s.id}:${r.index}`;
            return (
              <button key={r.index} className={`task case${selectedDrawing === drawing ? " active" : ""}`} onClick={() => onSelect(r.key, drawing)}>
                <span className="task-glyph">{r.character}</span>
                <span className="task-meta">
                  <span className={r.error ? "bad-text ellipsis" : "ellipsis"}>{r.error ? r.error : r.parse_ok ? "parsed" : "did not parse"}</span>
                  <span className="muted">
                    {r.latency_s.toFixed(1)} s{r.trace_summary ? ` · ${r.trace_summary.n_tool_calls} calls` : ""}
                  </span>
                </span>
                <RewardPill value={r.reward} />
              </button>
            );
          })}
        </div>
      </>
    );
  }

  return (
    <>
      <button className="primary" onClick={onNewRun}>
        + New run
      </button>
      <div className="list">
        {error && <div className="list-note bad">{error}</div>}
        <div className="list-head">Runs · {runs.length}</div>
        {runs.length === 0 && <div className="list-note">No runs yet. Start one to see a model's answers here.</div>}
        {runs.map((r) => (
          <button key={r.id} className="run-item" onClick={() => onActiveRun(r.id)}>
            <RunCard s={r} />
          </button>
        ))}
      </div>
    </>
  );
}

function JudgeList({ activeJudge, onActiveJudge }: Props) {
  const [list, setList] = useState<JudgeSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Refreshes while an experiment is still going.
  useEffect(() => {
    let alive = true;
    const load = () => api.judgeList().then((r) => alive && setList(r)).catch((e) => alive && setError(String(e)));
    load();
    const id = setInterval(load, 4000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="list">
      {error && <div className="list-note bad">{error}</div>}
      <div className="list-head">Judge experiments · {list.length}</div>
      {list.length === 0 && !error && (
        <div className="list-note">
          None yet. Run <code>uv run python scripts/run_judge.py --provider openrouter --model qwen/qwen3.8-27b</code>.
        </div>
      )}
      {list.map((s) => {
        const rho = s.stats?.spearman;
        const pct = s.n_total ? (100 * s.n_done) / s.n_total : 0;
        return (
          <button key={s.id} className={`run-item${activeJudge === s.id ? " active" : ""}`} onClick={() => onActiveJudge(s.id)}>
            <div className="run-card">
              <div className="run-top">
                <span className="run-model ellipsis">{s.model}</span>
                <span className={`status ${s.status}`}>{s.status}</span>
              </div>
              <div className="muted run-sub ellipsis">
                {s.provider} · {s.characters}
              </div>
              <div className="muted run-sub ellipsis">started {formatTime(s.created_at)}</div>
              <div className="progress" aria-label={`${s.n_done} of ${s.n_total}`}>
                <i style={{ width: `${pct}%` }} />
              </div>
              <div className="run-stats">
                <span>
                  {s.n_done}/{s.n_total}
                </span>
                <span>ρ {rho == null ? "–" : rho.toFixed(2)}</span>
                {s.n_errors > 0 && <span className="bad-text">{s.n_errors} errors</span>}
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

/** "2026-10-05T09:15:13" -> "Oct 5, 09:15" */
function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
}

export function RunCard({ s }: { s: RunSummary }) {
  const pct = s.n_total ? (100 * s.n_done) / s.n_total : 0;
  return (
    <div className="run-card">
      <div className="run-top">
        <span className="run-model ellipsis">{s.model}</span>
        <span className={`status ${s.status}`}>{s.status}</span>
      </div>
      <div className="muted run-sub ellipsis">
        {s.provider} · {s.dataset_label}
        {s.mode === "tools" ? " · tools" : ""}
      </div>
      <div className="muted run-sub ellipsis" title={s.finished_at ? `finished ${formatTime(s.finished_at)}` : undefined}>
        started {formatTime(s.created_at)}
      </div>
      <div className="progress" aria-label={`${s.n_done} of ${s.n_total}`}>
        <i style={{ width: `${pct}%` }} />
      </div>
      <div className="run-stats">
        <span>
          {s.n_done}/{s.n_total}
        </span>
        <span>reward {s.mean_reward == null ? "–" : s.mean_reward.toFixed(3)}</span>
        <span>parsed {s.n_done ? Math.round((100 * s.n_parse_ok) / s.n_done) : 0}%</span>
        {s.n_errors > 0 && <span className="bad-text">{s.n_errors} errors</span>}
      </div>
    </div>
  );
}

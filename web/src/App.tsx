import { useCallback, useEffect, useState } from "react";
import { api, CharData, RenderOpts, Task } from "./api";
import { JudgePanel } from "./components/Judge";
import { EnginePanel, ReferencePanel, Show, StrokesPanel } from "./components/Panels";
import { RunForm } from "./components/RunForm";
import { ScoreSummary, StrokeTable } from "./components/Scores";
import { RewardPill, Sidebar, Source } from "./components/Sidebar";
import { TracePanel } from "./components/Trace";

const IDS_NAMES: Record<string, string> = {
  "⿰": "left–right",
  "⿱": "top–bottom",
  "⿲": "left–middle–right",
  "⿳": "top–middle–bottom",
  "⿴": "surround",
  "⿵": "open below",
  "⿶": "open above",
  "⿷": "open right",
  "⿸": "upper-left wrap",
  "⿹": "upper-right wrap",
  "⿺": "lower-left wrap",
  "⿻": "overlaid",
};

/** "⿰雚見" -> "雚 + 見 · left–right" (IDS operator glyphs are missing from most fonts). */
function describeIds(ids: string): string {
  const op = ids[0];
  if (!(op in IDS_NAMES)) return ids;
  const parts = [...ids.slice(1)].map((c) => (c in IDS_NAMES ? "" : c)).filter(Boolean);
  return `${parts.join(" + ")} · ${IDS_NAMES[op]}`;
}

/** What the page is showing, kept in the URL hash so a refresh (or a shared link) lands on the same place. */
type Place = { key: string; drawing: string; source: Source; run: string | null; judge: string | null };

const SOURCES: Source[] = ["bench", "normalization", "runs", "judge"];

/** "#<key>?drawing=…&source=…&run=…&judge=…"; a bare "#<key>" (older links) still works. */
function readPlace(): Place {
  const [rawKey, query = ""] = window.location.hash.slice(1).split("?");
  const q = new URLSearchParams(query);
  let key = "wxz_6420";
  try {
    key = decodeURIComponent(rawKey) || key;
  } catch {
    /* malformed hash: fall back to the default character */
  }
  const source = q.get("source") as Source;
  return { key, drawing: q.get("drawing") || "converter", source: SOURCES.includes(source) ? source : "bench", run: q.get("run"), judge: q.get("judge") };
}

function writePlace(p: Place): void {
  const q = new URLSearchParams();
  if (p.drawing !== "converter") q.set("drawing", p.drawing);
  if (p.source !== "bench") q.set("source", p.source);
  if (p.run) q.set("run", p.run);
  if (p.judge) q.set("judge", p.judge);
  const query = q.toString();
  const hash = `#${encodeURIComponent(p.key)}${query ? `?${query}` : ""}`;
  if (hash !== window.location.hash) window.history.replaceState(null, "", hash);
}

export default function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [start] = useState(readPlace);
  const [key, setKey] = useState<string>(start.key);
  const [data, setData] = useState<CharData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);
  const [opts, setOpts] = useState<RenderOpts>({ drawing: start.drawing, brush: "disc", size: 512, grid: true });
  const [show, setShow] = useState<Show>({ outlines: false, medians: true, numbers: true, ghost: true, pixelated: false });
  const [source, setSource] = useState<Source>(start.source);
  const [activeRun, setActiveRun] = useState<string | null>(start.run);
  const [activeJudge, setActiveJudge] = useState<string | null>(start.judge);
  const [view, setView] = useState<"character" | "newRun" | "judge">(start.source === "judge" && start.judge ? "judge" : "character");

  const select = (nextKey: string, drawing: string) => {
    setKey(nextKey);
    setOpts((o) => ({ ...o, drawing }));
    setView("character");
  };

  useEffect(() => {
    api.tasks().then(setTasks).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    writePlace({ key, drawing: opts.drawing, source, run: activeRun, judge: activeJudge });
  }, [key, opts.drawing, source, activeRun, activeJudge]);

  useEffect(() => {
    setError(null);
    api
      .char(key, opts.drawing, opts.brush)
      .then((d) => {
        setData(d);
        setK(Math.max(d.n_strokes, d.strokes.length));
      })
      .catch((e) => setError(String(e)));
  }, [key, opts.drawing, opts.brush]);

  // Prefetch every step so scrubbing is instant.
  useEffect(() => {
    if (!data) return;
    for (let i = 0; i <= data.strokes.length; i++) new Image().src = api.renderUrl(data.key, i, opts);
  }, [data, opts]);

  const n = data ? Math.max(data.n_strokes, data.strokes.length) : 0;
  const step = useCallback((d: number) => setK((v) => Math.max(0, Math.min(n, v + d))), [n]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setK((v) => {
        if (v >= n) {
          setPlaying(false);
          return v;
        }
        return v + 1;
      });
    }, 450);
    return () => clearInterval(id);
  }, [playing, n]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === "ArrowRight") step(1);
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const togglePlay = () => {
    if (!playing && k >= n) setK(0);
    setPlaying((p) => !p);
  };

  const toggle = (name: keyof Show) => setShow((s) => ({ ...s, [name]: !s[name] }));
  // The page keeps the selected character in the URL hash, so scroll instead of following the link.
  const jumpToTrace = (e: React.MouseEvent) => {
    e.preventDefault();
    document.getElementById("trace")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const meta = data?.meta ?? {};
  const pinyin = Array.isArray(meta.pinyin) ? (meta.pinyin as string[]).join(", ") : (meta.pinyin as string | undefined);

  return (
    <div className="app">
      <Sidebar
        tasks={tasks}
        source={source}
        onSource={setSource}
        selectedKey={key}
        selectedDrawing={opts.drawing}
        onSelect={select}
        onNewRun={() => setView("newRun")}
        activeRun={activeRun}
        onActiveRun={setActiveRun}
        activeJudge={activeJudge}
        onActiveJudge={(id) => {
          setActiveJudge(id);
          setView(id ? "judge" : "character");
        }}
      />
      <main className="main">
        {view === "newRun" && (
          <RunForm
            onStarted={(id) => {
              setSource("runs");
              setActiveRun(id);
            }}
          />
        )}
        {view === "judge" && activeJudge && <JudgePanel expId={activeJudge} onOpen={select} />}
        {view === "character" && error && <div className="error">{error}</div>}
        {view === "character" && data && (
          <>
            <header className="char-head">
              <div className="char-big">{data.character}</div>
              <div className="char-info">
                <div className="char-title">
                  <span>{pinyin}</span>
                  <span className="muted">{meta.definition as string}</span>
                </div>
                <div className="chips">
                  <span className="chip">{data.n_strokes} strokes</span>
                  {data.task_id && <span className="chip">{data.task_id}</span>}
                  {typeof meta.style_difficulty === "number" && <span className="chip">difficulty {meta.style_difficulty}/5</span>}
                  {typeof meta.decomposition === "string" && (
                    <span className="chip" title={meta.decomposition}>
                      {describeIds(meta.decomposition)}
                    </span>
                  )}
                </div>
              </div>
            </header>

            {data.rollout && (
              <section className="answer">
                <div className="answer-head">
                  <span>
                    Model answer · <b>{data.rollout.model}</b> · {data.rollout.latency_s.toFixed(1)} s
                  </span>
                  <RewardPill value={data.rollout.reward} />
                </div>
                {data.rollout.error ? (
                  <div className="bad-text">{data.rollout.error}</div>
                ) : data.rollout.trace_summary ? (
                  <div className="muted small">
                    {data.strokes.length} strokes left on the canvas for a {data.n_strokes}-stroke character ·{" "}
                    {data.rollout.trace_summary.n_tool_calls} tool calls in {data.rollout.trace_summary.n_turns} turns · ended by{" "}
                    {data.rollout.trace_summary.end_reason.replace(/_/g, " ")} · <a href="#trace" onClick={jumpToTrace}>see every call ↓</a>
                  </div>
                ) : (
                  <div className="muted small">
                    Parsed {data.strokes.length} strokes for a {data.n_strokes}-stroke character
                    {data.rollout.stop_reason && data.rollout.stop_reason !== "stop" && data.rollout.stop_reason !== "end_turn"
                      ? ` · stopped: ${data.rollout.stop_reason}`
                      : ""}
                  </div>
                )}
                <details>
                  <summary>
                    {data.rollout.trace_summary ? "Final canvas as strokes" : "Raw response"} ({data.rollout.response.length.toLocaleString()}{" "}
                    characters)
                  </summary>
                  <pre>{data.rollout.response || "(empty)"}</pre>
                </details>
              </section>
            )}

            <div className="controls">
              <label>
                Drawing
                <select value={opts.drawing} onChange={(e) => setOpts({ ...opts, drawing: e.target.value })}>
                  {data.drawings.map((d) => (
                    <option key={d.name} value={d.name}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Brush
                <select value={opts.brush} onChange={(e) => setOpts({ ...opts, brush: e.target.value })}>
                  <option value="disc">disc (default)</option>
                  <option value="freehand">freehand (tldraw pen)</option>
                </select>
              </label>
              <label>
                Size
                <select value={opts.size} onChange={(e) => setOpts({ ...opts, size: Number(e.target.value) })}>
                  {[224, 512, 1024].map((s) => (
                    <option key={s} value={s}>
                      {s}px
                    </option>
                  ))}
                </select>
              </label>
              <div className="toggles">
                <Toggle on={opts.grid} onClick={() => setOpts({ ...opts, grid: !opts.grid })} label="Grid" />
                <Toggle on={show.medians} onClick={() => toggle("medians")} label="Medians" />
                <Toggle on={show.numbers} onClick={() => toggle("numbers")} label="Order" />
                <Toggle on={show.ghost} onClick={() => toggle("ghost")} label="Ghost" />
                <Toggle on={show.outlines} onClick={() => toggle("outlines")} label="Contours" />
                <Toggle on={show.pixelated} onClick={() => toggle("pixelated")} label="Pixels" />
              </div>
            </div>

            <div className="panels">
              <ReferencePanel data={data} show={show} />
              <StrokesPanel data={data} k={k} hovered={hovered} onHover={setHovered} show={show} />
              <EnginePanel data={data} k={k} hovered={hovered} onHover={setHovered} show={show} opts={opts} />
            </div>

            <div className="scrubber">
              <button onClick={() => setK(0)} aria-label="First">⏮</button>
              <button onClick={() => step(-1)} aria-label="Previous">◀</button>
              <button className="play" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
                {playing ? "❚❚" : "▶"}
              </button>
              <button onClick={() => step(1)} aria-label="Next">▶▏</button>
              <input type="range" min={0} max={n} value={k} onChange={(e) => setK(Number(e.target.value))} aria-label="Stroke" />
              <span className="step-label">
                stroke <b>{k}</b> / {n}
              </span>
            </div>

            <ScoreSummary data={data} />
            {data.rollout?.trace_summary && <TracePanel rollout={data.rollout} />}
            <StrokeTable data={data} k={k} hovered={hovered} onHover={setHovered} onPick={setK} />
          </>
        )}
      </main>
    </div>
  );
}

function Toggle({ on, onClick, label }: { on: boolean; onClick: () => void; label: string }) {
  return (
    <button className={`toggle${on ? " on" : ""}`} onClick={onClick} aria-pressed={on}>
      {label}
    </button>
  );
}

import { useCallback, useEffect, useState } from "react";
import { api, CharData, RenderOpts, Task } from "./api";
import { EnginePanel, ReferencePanel, Show, StrokesPanel } from "./components/Panels";
import { ScoreSummary, StrokeTable } from "./components/Scores";
import { TaskList } from "./components/TaskList";

const DRAWING_LABELS: Record<string, string> = {
  converter: "Converter output (SFT label)",
  A_small_corner: "Scene A: small, in a corner",
  B_slid_component: "Scene B: component slid",
  C_straight_flat: "Scene C: straight, flat pressure",
};

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

const initialKey = () => decodeURIComponent(window.location.hash.slice(1)) || "wxz_6420";

export default function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [key, setKey] = useState<string>(initialKey);
  const [data, setData] = useState<CharData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);
  const [opts, setOpts] = useState<RenderOpts>({ drawing: "converter", brush: "disc", size: 512, grid: true });
  const [show, setShow] = useState<Show>({ outlines: false, medians: true, numbers: true, ghost: true, pixelated: false });

  useEffect(() => {
    api.tasks().then(setTasks).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    window.location.hash = encodeURIComponent(key);
    setError(null);
    api
      .char(key, opts.drawing, opts.brush)
      .then((d) => {
        setData(d);
        setK(d.n_strokes);
      })
      .catch((e) => setError(String(e)));
  }, [key, opts.drawing, opts.brush]);

  // Prefetch every step so scrubbing is instant.
  useEffect(() => {
    if (!data) return;
    for (let i = 0; i <= data.strokes.length; i++) new Image().src = api.renderUrl(data.key, i, opts);
  }, [data, opts]);

  const n = data?.n_strokes ?? 0;
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
  const meta = data?.meta ?? {};
  const pinyin = Array.isArray(meta.pinyin) ? (meta.pinyin as string[]).join(", ") : (meta.pinyin as string | undefined);

  return (
    <div className="app">
      <TaskList tasks={tasks} selected={key} onSelect={(k) => setKey(k)} />
      <main className="main">
        {error && <div className="error">{error}</div>}
        {data && (
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

            <div className="controls">
              <label>
                Drawing
                <select value={opts.drawing} onChange={(e) => setOpts({ ...opts, drawing: e.target.value })}>
                  {data.drawings.map((d) => (
                    <option key={d} value={d}>
                      {DRAWING_LABELS[d] ?? d}
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

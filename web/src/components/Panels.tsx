import { api, CharData, RenderOpts } from "../api";
import { CanvasSvg, RiceGrid } from "./RiceGrid";

export type Show = { outlines: boolean; medians: boolean; numbers: boolean; ghost: boolean; pixelated: boolean };

type PanelProps = { data: CharData; k: number; hovered: number | null; onHover: (i: number | null) => void; show: Show };

function Panel({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-head">
        <span className="panel-title">{title}</span>
        {note && <span className="muted">{note}</span>}
      </header>
      <div className="square">{children}</div>
    </section>
  );
}

export function ReferencePanel({ data, show }: { data: CharData; show: Show }) {
  const calligrapher = data.meta.calligrapher as string | undefined;
  return (
    <Panel title="Master reference" note={data.has_reference ? `${calligrapher ?? ""} · 64×64` : "none"}>
      {data.has_reference ? (
        <>
          <img
            src={api.referenceUrl(data.key)}
            className={`fill ${show.pixelated ? "pixelated" : ""}`}
            alt={`${data.character} by ${calligrapher}`}
          />
          <CanvasSvg>
            <RiceGrid />
          </CanvasSvg>
        </>
      ) : (
        <div className="empty">Make Me a Hanzi entries have no reference image</div>
      )}
    </Panel>
  );
}

const MEDIAN_COLORS = ["var(--accent)", "var(--accent-2)"];

export function StrokesPanel({ data, k, hovered, onHover, show }: PanelProps) {
  return (
    <Panel title="Dataset strokes" note="canonical Kaiti, not the master's">
      <CanvasSvg className="overlay paper">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" className="arrow-head" />
          </marker>
        </defs>
        <RiceGrid />
        {data.outlines.map((d, i) => {
          const state = i < k - 1 ? "done" : i === k - 1 ? "current" : "future";
          if (state === "future" && !show.ghost) return null;
          return (
            <path
              key={i}
              d={d}
              className={`outline ${state}${hovered === i ? " hover" : ""}`}
              onMouseEnter={() => onHover(i)}
              onMouseLeave={() => onHover(null)}
            />
          );
        })}
        {show.medians &&
          data.medians.slice(0, k).map((m, i) => (
            <g key={i} className="median" style={{ color: MEDIAN_COLORS[i % 2] }}>
              <polyline points={m.map((p) => p.join(",")).join(" ")} markerEnd="url(#arrow)" />
              <circle cx={m[0][0]} cy={m[0][1]} r={9} />
            </g>
          ))}
        {show.numbers &&
          data.medians.slice(0, k).map((m, i) => (
            <text key={i} x={m[0][0] - 14} y={m[0][1] - 14} className="stroke-num">
              {i + 1}
            </text>
          ))}
      </CanvasSvg>
    </Panel>
  );
}

export function EnginePanel({ data, k, hovered, show, opts }: PanelProps & { opts: RenderOpts }) {
  const drawn = data.strokes.slice(0, k);
  return (
    <Panel title="Engine render" note={`${opts.brush} · ${opts.size}px · what the model would see`}>
      <img src={api.renderUrl(data.key, k, opts)} className="fill" alt={`engine render after ${k} strokes`} />
      <CanvasSvg>
        {show.outlines &&
          data.outlines.slice(0, k).map((d, i) => <path key={i} d={d} className={`contour${hovered === i ? " hover" : ""}`} />)}
        {drawn.map((s, i) => (
          <g key={i} className={`agent-points${i === k - 1 ? " current" : ""}${hovered === i ? " hover" : ""}`}>
            {s.points.map(([x, y], j) => (
              <circle key={j} cx={x} cy={y} r={j === 0 ? 8 : 5} />
            ))}
          </g>
        ))}
      </CanvasSvg>
    </Panel>
  );
}

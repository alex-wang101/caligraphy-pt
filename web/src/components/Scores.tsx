import { CharData } from "../api";

const pct = (v: number) => v.toFixed(3);

export function ScoreSummary({ data }: { data: CharData }) {
  const l = data.levels;
  return (
    <div className="scores">
      <Stat label="Structure (reward v1)" value={pct(l.structure)} hint="one robust scale + translation, per-stroke IoU" />
      <Stat label="Stroke shape" value={pct(l.shape)} hint="per-stroke scale + translation, IoU" />
      <Stat
        label="Placement"
        value={`${l.placement_offset.toFixed(0)} · ${l.placement_ratio.toFixed(2)}×`}
        hint="bbox centre offset · size ratio"
        flag={!l.placement_ok}
      />
      <Stat label="Alignment" value={`s ${l.align_scale.toFixed(2)}`} hint={`t (${l.align_offset.map((v) => v.toFixed(0)).join(", ")})`} />
      <Stat label="Strokes" value={`${l.n_agent} / ${l.n_target}`} hint={`scored in ${data.score_ms.toFixed(0)} ms`} />
    </div>
  );
}

function Stat({ label, value, hint, flag }: { label: string; value: string; hint: string; flag?: boolean }) {
  return (
    <div className={`stat${flag ? " flag" : ""}`} title={hint}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-hint">{hint}</div>
    </div>
  );
}

type TableProps = { data: CharData; k: number; hovered: number | null; onHover: (i: number | null) => void; onPick: (i: number) => void };

export function StrokeTable({ data, k, hovered, onHover, onPick }: TableProps) {
  const l = data.levels;
  return (
    <div className="table-wrap">
      <table className="strokes">
        <thead>
          <tr>
            <th>#</th>
            <th title="rendered ink vs filled outline, no alignment">IoU</th>
            <th title="IoU after the whole-character robust alignment">structure</th>
            <th title="IoU after a per-stroke alignment">shape</th>
            <th title="discrete Fréchet, agent path vs median (canvas units)">Fréchet</th>
            <th>direction</th>
            <th title="fraction of the path inside the outline">inside</th>
            <th>points</th>
            <th title="used in the robust fit">fit</th>
          </tr>
        </thead>
        <tbody>
          {data.stroke_metrics.map((m, i) => (
            <tr
              key={i}
              className={`${i === k - 1 ? "current" : ""}${i >= k ? " later" : ""}${hovered === i ? " hover" : ""}`}
              onMouseEnter={() => onHover(i)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onPick(i + 1)}
            >
              <td>{i + 1}</td>
              <td>{m.iou.toFixed(2)}</td>
              <td>{(l.structure_strokes[i] ?? 0).toFixed(2)}</td>
              <td>{(l.shape_strokes[i] ?? 0).toFixed(2)}</td>
              <td>{m.frechet.toFixed(1)}</td>
              <td className={m.direction_ok ? "ok" : "bad"}>{m.direction_ok ? "✓" : "reversed"}</td>
              <td>{m.inside_frac.toFixed(2)}</td>
              <td>{data.strokes[i]?.points.length ?? "–"}</td>
              <td>{l.align_inliers[i] ? "●" : "○"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

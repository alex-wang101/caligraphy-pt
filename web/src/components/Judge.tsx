import { useEffect, useMemo, useState } from "react";
import { api, JudgeDetail, JudgeItem, JudgeKind, JudgeStats } from "../api";
import { RewardPill } from "./Sidebar";

const KINDS: JudgeKind[] = ["correct", "flawed", "wrong character", "model answer"];
const KIND_CLASS: Record<JudgeKind, string> = {
  correct: "k-correct",
  flawed: "k-flawed",
  "wrong character": "k-wrong",
  "model answer": "k-model",
};

/** Judge variants that are also viewer drawings, so the item can be opened in the character view. */
const VIEWER_DRAWING: Record<string, string> = {
  perfect: "converter",
  small_corner: "A_small_corner",
  slid_component: "B_slid_component",
  straight_flat: "C_straight_flat",
  "squash_x_0.6": "squash_x_0.6",
  rotate_8: "rotate_8",
  rotate_20: "rotate_20",
  "tiny_0.2": "tiny_0.2",
};

const fmt = (v: number | null | undefined, digits = 2) => (v == null ? "–" : v.toFixed(digits));
const pct = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(100 * v)}%`);

type Props = { expId: string; onOpen: (key: string, drawing: string) => void };

export function JudgePanel({ expId, onOpen }: Props) {
  const [detail, setDetail] = useState<JudgeDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [char, setChar] = useState("");
  const [kind, setKind] = useState("");
  const [sort, setSort] = useState<"order" | "gap">("order");
  const [hovered, setHovered] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    setDetail(null);
    const load = () =>
      api
        .judge(expId)
        .then((d) => {
          if (!alive) return;
          setDetail(d);
          setError(null);
          if (d.summary.status === "running") timer = setTimeout(load, 2000);
        })
        .catch((e) => alive && setError(String(e)));
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [expId]);

  const chars = useMemo(() => [...new Set((detail?.items ?? []).map((i) => i.character))], [detail]);
  if (error) return <div className="error">{error}</div>;
  if (!detail) return <div className="muted">Loading experiment…</div>;

  const s = detail.summary;
  const st = s.stats;
  const gap = (i: JudgeItem) => (i.judge?.score == null ? -1 : Math.abs(i.judge.score - i.reward));
  const shown = detail.items
    .filter((i) => (!char || i.character === char) && (!kind || i.kind === kind))
    .sort((a, b) => (sort === "gap" ? gap(b) - gap(a) : a.index - b.index));

  return (
    <div className="judge">
      <header className="judge-head">
        <div>
          <h2>Can the model grade a drawing?</h2>
          <p className="muted">
            Each drawing was shown to <b>{s.model}</b> as <code>capture_canvas</code> shows it ({s.obs_size} px, with the grid) and the model
            gave a 0–100 score. The reward is the eval score for the same drawing. Characters: {s.characters} · temperature {s.temperature ?? "default"}.
          </p>
        </div>
        <span className={`status ${s.status}`}>
          {s.status} · {s.n_done}/{s.n_total}
        </span>
      </header>
      {s.error && <div className="error">{s.error}</div>}
      {st && <JudgeStatsRow st={st} errors={s.n_errors} />}

      {st && st.n_scored > 0 && (
        <div className="judge-grid">
          <section className="panel judge-plot">
            <div className="panel-head">
              <span className="panel-title">Model score vs reward</span>
              <span className="muted">one dot per drawing · diagonal = agreement</span>
            </div>
            <Scatter items={detail.items} hovered={hovered} onHover={setHovered} />
            <div className="legend">
              {KINDS.map((k) => (
                <span key={k}>
                  <i className={KIND_CLASS[k]} />
                  {k}
                </span>
              ))}
            </div>
          </section>
          <section className="panel">
            <div className="panel-head">
              <span className="panel-title">By kind of drawing</span>
              <span className="muted">means over characters</span>
            </div>
            <div className="table-wrap">
              <table className="strokes judge-table">
                <thead>
                  <tr>
                    <th>Drawing</th>
                    <th className="num">n</th>
                    <th className="num" title="mean eval reward (structure)">reward</th>
                    <th className="num" title="mean model score / 100">model</th>
                    <th className="num" title="share of drawings the model called the correct character">says right char</th>
                  </tr>
                </thead>
                <tbody>
                  {st.by_variant.map((r) => (
                    <tr key={r.variant}>
                      <td>
                        <i className={`dot ${KIND_CLASS[r.kind]}`} /> {r.label}
                      </td>
                      <td className="num">{r.n}</td>
                      <td className="num">{fmt(r.reward_mean)}</td>
                      <td className={`num${Math.abs(r.judge_mean - r.reward_mean) > 0.3 ? " flagged" : ""}`}>{fmt(r.judge_mean)}</td>
                      <td className="num">{pct(r.says_correct)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}

      <div className="controls">
        <label>
          Character
          <select value={char} onChange={(e) => setChar(e.target.value)}>
            <option value="">All</option>
            {chars.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
        <label>
          Kind
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="">All</option>
            {KINDS.map((k) => (
              <option key={k}>{k}</option>
            ))}
          </select>
        </label>
        <label>
          Order
          <select value={sort} onChange={(e) => setSort(e.target.value as "order" | "gap")}>
            <option value="order">As run</option>
            <option value="gap">Biggest disagreement first</option>
          </select>
        </label>
        <span className="muted small">{shown.length} drawings</span>
      </div>

      <div className="judge-cards">
        {shown.map((i) => (
          <JudgeCard key={i.index} expId={s.id} item={i} active={hovered === i.index} onHover={setHovered} onOpen={onOpen} />
        ))}
      </div>

      <details className="answer">
        <summary>Prompt shown with each image</summary>
        <pre>{s.prompt}</pre>
      </details>
    </div>
  );
}

function JudgeStatsRow({ st, errors }: { st: JudgeStats; errors: number }) {
  return (
    <div className="scores">
      <Tile label="Rank agreement" value={fmt(st.spearman)} hint="Spearman ρ between model score and reward (1 = same order)" />
      <Tile
        label="Good vs bad"
        value={fmt(st.auc_good_bad)}
        hint={`chance a good drawing (reward ≥ 0.8, n=${st.n_good}) outscores a bad one (< 0.3, n=${st.n_bad}); 0.5 = coin flip`}
      />
      <Tile label="Mean gap" value={fmt(st.mae)} hint="mean |model score − reward|" />
      <Tile label="Reads correct drawings" value={pct(st.reads_correct)} hint="correct drawings it named as the right character" />
      <Tile label="Catches wrong character" value={pct(st.rejects_wrong)} hint="a different character drawn: share it called not the right character" />
      <Tile label="Scored" value={`${st.n_scored} / ${st.n}`} hint={errors ? `${errors} request errors` : "replies with a usable score"} flag={st.n_scored < st.n} />
    </div>
  );
}

function Tile({ label, value, hint, flag }: { label: string; value: string; hint: string; flag?: boolean }) {
  return (
    <div className={`stat${flag ? " flag" : ""}`} title={hint}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-hint">{hint}</div>
    </div>
  );
}

const W = 440;
const H = 320;
const M = { l: 44, r: 14, t: 12, b: 38 };

function Scatter({ items, hovered, onHover }: { items: JudgeItem[]; hovered: number | null; onHover: (i: number | null) => void }) {
  const x = (v: number) => M.l + v * (W - M.l - M.r);
  const y = (v: number) => H - M.b - v * (H - M.t - M.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const pts = items.filter((i) => i.judge?.score != null);
  return (
    <svg className="scatter" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Model score against reward">
      {ticks.map((t) => (
        <g key={t}>
          <line className="grid-line" x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} />
          <line className="grid-line" x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} />
          <text className="tick" x={x(t)} y={H - M.b + 16} textAnchor="middle">
            {t}
          </text>
          <text className="tick" x={M.l - 8} y={y(t) + 4} textAnchor="end">
            {t}
          </text>
        </g>
      ))}
      <line className="diag" x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
      <text className="axis" x={(x(0) + x(1)) / 2} y={H - 4} textAnchor="middle">
        reward (eval score)
      </text>
      <text className="axis" transform={`translate(12 ${(y(0) + y(1)) / 2}) rotate(-90)`} textAnchor="middle">
        model score
      </text>
      {pts.map((i) => (
        <circle
          key={i.index}
          className={`pt ${KIND_CLASS[i.kind]}${hovered === i.index ? " hot" : ""}`}
          cx={x(Math.min(1, i.reward))}
          cy={y(i.judge!.score!)}
          r={hovered === i.index ? 7 : 4.5}
          onMouseEnter={() => onHover(i.index)}
          onMouseLeave={() => onHover(null)}
        >
          <title>
            {i.character} · {i.label} · reward {i.reward.toFixed(2)} · model {i.judge!.score!.toFixed(2)}
          </title>
        </circle>
      ))}
    </svg>
  );
}

function JudgeCard({
  expId,
  item: i,
  active,
  onHover,
  onOpen,
}: {
  expId: string;
  item: JudgeItem;
  active: boolean;
  onHover: (i: number | null) => void;
  onOpen: (key: string, drawing: string) => void;
}) {
  const j = i.judge;
  const drawing = i.source ?? VIEWER_DRAWING[i.variant];
  const read = j?.recognized_as?.trim();
  return (
    <div className={`judge-card${active ? " active" : ""}`} onMouseEnter={() => onHover(i.index)} onMouseLeave={() => onHover(null)}>
      <img src={api.judgeImageUrl(expId, i.index)} alt={`${i.character}: ${i.label}`} loading="lazy" width={448} height={448} />
      <div className="judge-card-body">
        <div className="judge-card-title">
          <span className="task-glyph">{i.character}</span>
          <span className="ellipsis" title={i.label}>
            <i className={`dot ${KIND_CLASS[i.kind]}`} /> {i.label}
          </span>
        </div>
        <div className="judge-scores">
          <span title="eval reward (structure)">
            reward <RewardPill value={i.reward} />
          </span>
          <span title="model score / 100">model {j?.score != null ? <RewardPill value={j.score} /> : <span className="bad-text">–</span>}</span>
        </div>
        {i.error ? (
          <div className="bad-text small">{i.error}</div>
        ) : j ? (
          <>
            <div className="small">
              reads it as <b className={read === i.character ? "" : "bad-text"}>{read || "?"}</b>
              {j.is_correct_character != null && <span className="muted"> · says {j.is_correct_character ? "right" : "wrong"} character</span>}
            </div>
            {j.problems && <div className="muted small judge-problems">{j.problems}</div>}
          </>
        ) : null}
        <div className="judge-card-foot">
          <span className="muted small">
            {i.n_strokes}/{i.n_target} strokes
          </span>
          {drawing && (
            <button className="link" onClick={() => onOpen(i.key, drawing)}>
              Open in viewer
            </button>
          )}
        </div>
        <details>
          <summary className="small">Raw reply</summary>
          <pre>{i.reply || "(empty)"}</pre>
        </details>
      </div>
    </div>
  );
}

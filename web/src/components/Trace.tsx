import { useEffect, useState } from "react";
import { api, ModelEvent, Rollout, ToolEvent, Trace, TraceEvent } from "../api";

const END_LABELS: Record<string, string> = {
  submit: "The model submitted.",
  max_turns: "Stopped at the turn limit.",
  max_tool_calls: "Stopped at the tool-call limit.",
  no_tool_call: "Stopped: the model replied without calling a tool.",
  error: "Stopped by an error.",
  rate_limited: "Stopped by the provider's rate limit.",
};

/** [1, 2, 3, 5] -> "1–3, 5" */
function formatIds(ids: number[]): string {
  if (ids.length === 0) return "none";
  const runs: string[] = [];
  let start = ids[0];
  let prev = ids[0];
  for (const id of [...ids.slice(1), null]) {
    if (id !== null && id === prev + 1) {
      prev = id;
      continue;
    }
    runs.push(start === prev ? `${start}` : `${start}–${prev}`);
    if (id !== null) start = prev = id;
  }
  return runs.join(", ");
}

/** One tool call with the canvas state on either side of it. */
type Step = {
  e: ToolEvent;
  idsBefore: number[];
  idsAfter: number[];
  scoreBefore: number | null;
  scoreAfter: number | null;
  changed: boolean;
};

function buildSteps(events: TraceEvent[]): Map<number, Step> {
  const steps = new Map<number, Step>();
  let ids: number[] = [];
  let score: number | null = null;
  for (const e of events) {
    if (e.type !== "tool") continue;
    const changed = e.ok && (e.name === "create_stroke" || e.name === "delete_stroke");
    const idsAfter: number[] = e.stroke_ids ?? ids;
    const scoreAfter: number | null = changed && e.score ? e.score.structure : score;
    steps.set(e.i, { e, idsBefore: ids, idsAfter, scoreBefore: score, scoreAfter, changed });
    ids = idsAfter;
    score = scoreAfter;
  }
  return steps;
}

function describeCall(e: ToolEvent): string {
  const a = e.arguments;
  if (!a) return e.arguments_raw ? `unreadable arguments: ${e.arguments_raw.slice(0, 80)}` : "";
  if (e.name === "create_stroke" && Array.isArray(a.points) && a.points.length > 0) {
    const pts = a.points as unknown[];
    const xy = (p: unknown) => (Array.isArray(p) ? `(${Number(p[0]).toFixed(0)}, ${Number(p[1]).toFixed(0)})` : "?");
    return `${pts.length} points, ${xy(pts[0])} → ${xy(pts[pts.length - 1])}, width ${a.width}`;
  }
  if (e.name === "delete_stroke") {
    const was = e.deleted_stroke_score;
    return `stroke ${a.stroke_id}` + (was != null ? ` · it scored ${was.toFixed(2)}` : "");
  }
  return "";
}

const fmt = (v: number | null) => (v == null ? "–" : v.toFixed(3));

function Call({ step, rollout }: { step: Step; rollout: Rollout }) {
  const { e, idsBefore, idsAfter, scoreBefore, scoreAfter, changed } = step;
  const { run_id, index } = rollout;
  const drew = e.ok && e.name === "create_stroke" && e.stroke_id != null;
  const removed = e.ok && e.name === "delete_stroke" && e.stroke_id != null;
  const before = api.traceStateUrl(run_id, index, e.i - 1, removed ? { stroke: e.stroke_id!, tone: "removed" } : undefined);
  const after = e.image
    ? api.traceImageUrl(run_id, index, e.image)
    : api.traceStateUrl(run_id, index, e.i, drew ? { stroke: e.stroke_id!, tone: "added" } : undefined);
  const delta = scoreBefore != null && scoreAfter != null ? scoreAfter - scoreBefore : null;
  const status = e.skipped ? "not run" : !e.ok ? "rejected" : e.redraw ? "redraw" : null;
  return (
    <li className={`call${e.ok ? "" : " rejected"}`}>
      <div className="call-text">
        <div className="call-head">
          <span className="call-no">#{e.i}</span>
          <b>{e.name}</b>
          {status && <span className={`call-tag${e.ok ? "" : " bad"}`}>{status}</span>}
          <span className="muted">{describeCall(e)}</span>
        </div>
        <div className={e.ok ? "call-result" : "call-result bad-text"}>→ {e.result}</div>
        <div className="call-state muted small">
          strokes on the canvas: {formatIds(idsBefore)} → {formatIds(idsAfter)}
          {changed && (
            <>
              {" "}
              · score {fmt(scoreBefore)} → <b>{fmt(scoreAfter)}</b>
              {delta != null && <span className={delta >= 0 ? "ok-text" : "bad-text"}> ({delta >= 0 ? "+" : ""}{delta.toFixed(3)})</span>}
            </>
          )}
        </div>
        {e.arguments && Object.keys(e.arguments).length > 0 && (
          <details>
            <summary>Arguments</summary>
            <pre>{JSON.stringify(e.arguments)}</pre>
          </details>
        )}
      </div>
      <div className="call-shots">
        <figure className="call-shot">
          <img src={before} alt={`canvas before call ${e.i}`} loading="lazy" />
          <figcaption>before{removed ? " · red: to be deleted" : ""}</figcaption>
        </figure>
        <span className="call-arrow" aria-hidden="true">
          →
        </span>
        <figure className="call-shot">
          <img src={after} alt={`canvas after call ${e.i}`} loading="lazy" />
          <figcaption>
            after
            {e.image ? " · as shown to the model" : drew ? " · blue: just drawn" : changed ? "" : " · unchanged"}
          </figcaption>
        </figure>
      </div>
    </li>
  );
}

function Turn({ e }: { e: ModelEvent }) {
  const tokens = (e.usage.completion_tokens ?? e.usage.output_tokens) as number | undefined;
  return (
    <li className="turn">
      <div className="turn-head">
        Turn {e.turn} · {e.latency_s.toFixed(1)} s · {e.n_tool_calls} tool call{e.n_tool_calls === 1 ? "" : "s"}
        {tokens ? ` · ${tokens.toLocaleString()} tokens out` : ""}
        {e.stop_reason ? ` · stop: ${e.stop_reason}` : ""}
      </div>
      {e.error && <div className="bad-text">{e.error}</div>}
      {e.text && <p className="said">“{e.text}”</p>}
      {e.reasoning && (
        <details>
          <summary>Thinking ({e.reasoning.length.toLocaleString()} characters)</summary>
          <pre>{e.reasoning}</pre>
        </details>
      )}
    </li>
  );
}

/** Every model reply and tool call of one multi-turn answer, with the canvas before and after each call. */
export function TracePanel({ rollout }: { rollout: Rollout }) {
  const [trace, setTrace] = useState<Trace | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setTrace(null);
    setError(null);
    api
      .trace(rollout.run_id, rollout.index)
      .then((t) => alive && setTrace(t))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [rollout.run_id, rollout.index]);

  if (error) return <section className="trace error">{error}</section>;
  if (!trace) return <section className="trace muted">Loading tool calls…</section>;

  const s = trace.summary;
  const steps = buildSteps(trace.events);
  return (
    <section className="trace" id="trace">
      <header className="trace-head">
        <span className="trace-title">Tool calls</span>
        <span className="muted">
          {s.n_tool_calls} calls in {s.n_turns} turn{s.n_turns === 1 ? "" : "s"} · {s.n_captures} canvas image{s.n_captures === 1 ? "" : "s"} ·{" "}
          {s.n_deletes} delete{s.n_deletes === 1 ? "" : "s"} · {s.n_failed_calls} rejected
          {s.draft_score != null && s.final_score != null ? ` · draft ${s.draft_score.toFixed(3)} → final ${s.final_score.toFixed(3)}` : ""}
        </span>
      </header>
      <p className="muted small">
        Each row is one call, with the canvas before and after it. The score is the structure score the canvas would get if submitted at that
        point; the model never sees it.
      </p>
      <details>
        <summary>What the model was given</summary>
        <pre>{`${trace.system}\n\n${trace.user}`}</pre>
      </details>
      <ol className="trace-list">
        {trace.events.map((e) => {
          if (e.type === "model") return <Turn key={e.i} e={e} />;
          if (e.type === "tool") return <Call key={e.i} step={steps.get(e.i)!} rollout={rollout} />;
          if (e.type === "nudge")
            return (
              <li key={e.i} className="turn">
                <div className="turn-head">Reminder from the harness: “{e.text}”</div>
              </li>
            );
          return (
            <li key={e.i} className="turn">
              <div className="turn-head">{END_LABELS[e.reason] ?? `Ended: ${e.reason}`}</div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

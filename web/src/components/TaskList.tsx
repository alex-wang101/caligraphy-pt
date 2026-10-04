import { useEffect, useMemo, useState } from "react";
import { api, SearchHit, Task } from "../api";

type Props = { tasks: Task[]; selected: string | null; onSelect: (key: string) => void };

export function TaskList({ tasks, selected, onSelect }: Props) {
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
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-glyph">永</span>
        <div>
          <div className="brand-title">Calligraphy viewer</div>
          <div className="muted">calligraphy-bench · Make Me a Hanzi</div>
        </div>
      </div>
      <input
        className="search"
        placeholder="Search character or pinyin"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Search"
      />
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
          <button
            key={t.task_id}
            className={`task${selected === t.task_id ? " active" : ""}`}
            onClick={() => onSelect(t.task_id)}
          >
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
          <button
            key={h.character}
            className={`task${selected === h.character ? " active" : ""}`}
            onClick={() => onSelect(h.character)}
          >
            <span className="task-glyph">{h.character}</span>
            <span className="task-meta">
              <span>{h.pinyin.join(", ")}</span>
              <span className="muted ellipsis">{h.definition}</span>
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}

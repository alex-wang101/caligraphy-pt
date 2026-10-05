import { useEffect, useState } from "react";
import { api, DatasetOption, Provider, RunSummary, TestResult } from "../api";
import { RunCard } from "./Sidebar";

type Props = { onStarted: (runId: string) => void };

export function RunForm({ onStarted }: Props) {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [providerId, setProviderId] = useState("ollama");
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [dataset, setDataset] = useState("bench");
  const [mode, setMode] = useState<"single" | "tools">("single");
  const [characters, setCharacters] = useState("永國心");
  const [limit, setLimit] = useState(10);
  const [concurrency, setConcurrency] = useState(2);
  const [temperature, setTemperature] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(16000);
  const [test, setTest] = useState<TestResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState<RunSummary | null>(null);

  useEffect(() => {
    api.providers().then(setProviders).catch((e) => setError(String(e)));
    api.datasets().then(setDatasets).catch((e) => setError(String(e)));
  }, []);

  const provider = providers.find((p) => p.id === providerId);

  // Switching provider resets the endpoint and suggests a model name.
  useEffect(() => {
    if (!provider) return;
    setBaseUrl(provider.base_url);
    setModel(["stub", "ollama", "anthropic", "vllm_tunnel"].includes(provider.id) ? provider.model_hint : "");
    setApiKey("");
    setTest(null);
  }, [provider]);

  // Follow the started run until it finishes.
  useEffect(() => {
    if (!started || started.status !== "running") return;
    const id = setTimeout(() => api.run(started.id).then((d) => setStarted(d.summary)), 1000);
    return () => clearTimeout(id);
  }, [started]);

  const request = () => ({
    provider: providerId,
    model,
    base_url: baseUrl,
    api_key: apiKey || undefined,
    temperature: provider?.kind === "anthropic" ? null : temperature,
    max_tokens: maxTokens,
  });

  const runTest = async () => {
    setBusy(true);
    setError(null);
    setTest(null);
    try {
      setTest(await api.testModel(request()));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const { run_id } = await api.startRun({ ...request(), dataset, characters, limit: limit || null, concurrency, mode });
      setApiKey(""); // the server holds it for this run; don't keep it in the page
      onStarted(run_id);
      setStarted((await api.run(run_id)).summary);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!provider) return <div className="form-card">{error ?? "Loading providers…"}</div>;
  const needsKey = Boolean(provider.key_env);

  return (
    <div className="run-form">
      <header className="form-head">
        <h2>Run a model on the dataset</h2>
        <p className="muted">
          Each character is sent as the single-turn prompt the model will be trained on (the character's name only, no image). The answer is parsed,
          rendered and scored with reward v1, exactly as in training. Answers appear under Agent runs as they arrive.
        </p>
      </header>

      <section className="form-card">
        <h3>1. Model</h3>
        <label>
          Provider
          <select value={providerId} onChange={(e) => setProviderId(e.target.value)}>
            {providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <Steps steps={provider.steps} />
        <div className="form-grid">
          {provider.kind === "openai" && (
            <label>
              Base URL
              <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} spellCheck={false} />
            </label>
          )}
          <label>
            Model
            <input value={model} onChange={(e) => setModel(e.target.value)} placeholder={provider.model_hint} spellCheck={false} />
          </label>
        </div>
        {needsKey && (
          <div className="key-box">
            <div>
              <code>{provider.key_env}</code>{" "}
              {provider.key_env_set ? (
                <span className="ok-text">is set in the API server's environment ✓</span>
              ) : (
                <span className="muted">
                  is not set in the API server's environment{provider.kind === "anthropic" ? " (an `ant auth login` profile also works)" : ""}
                </span>
              )}
            </div>
            <label>
              Or paste a key
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={provider.key_env_set ? "optional: overrides the environment variable" : "paste key"}
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            <p className="muted small">
              A pasted key goes only to this local server, is held in memory for this run, and is never written to disk or sent back to the page.
            </p>
          </div>
        )}
        <div className="row">
          <button className="secondary" onClick={runTest} disabled={busy || !model}>
            Test connection
          </button>
          {test && (
            <span className={test.ok ? "ok-text" : "bad-text"}>
              {test.ok ? `Connected in ${test.latency_s}s · replied “${test.text.slice(0, 60)}”` : test.error}
            </span>
          )}
        </div>
      </section>

      <section className="form-card">
        <h3>2. Dataset</h3>
        <div className="form-grid">
          <label>
            Characters
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </label>
          {dataset === "custom" ? (
            <label>
              Custom characters
              <input value={characters} onChange={(e) => setCharacters(e.target.value)} />
            </label>
          ) : (
            <label>
              Limit (first N, 0 = all)
              <input type="number" min={0} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
            </label>
          )}
          <label>
            How the model answers
            <select value={mode} onChange={(e) => setMode(e.target.value as "single" | "tools")}>
              <option value="single">One reply with every stroke</option>
              <option value="tools">Step by step with tools (draw, delete, look)</option>
            </select>
          </label>
          <label>
            Parallel requests
            <input type="number" min={1} max={16} value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))} />
          </label>
          <label>
            Max output tokens
            <input type="number" min={64} max={128000} step={1000} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
          </label>
          {provider.kind === "openai" && (
            <label>
              Temperature
              <input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
            </label>
          )}
        </div>
        {mode === "tools" && (
          <p className="muted small">
            The model draws one stroke at a time, can ask for an image of its canvas and can delete a stroke and redraw it. Every tool call is
            recorded, with the canvas before and after it. A character takes several requests instead of one.
          </p>
        )}
        {!provider.local && (
          <p className="muted small">
            Hosted APIs bill per request: one request per character{mode === "tools" ? " and reply" : ""}, and a full answer is a few thousand output
            tokens. Start with a small limit.
          </p>
        )}
        <div className="row">
          <button className="primary" onClick={start} disabled={busy || !model}>
            Start run
          </button>
          {error && <span className="bad-text">{error}</span>}
        </div>
      </section>

      {started && (
        <section className="form-card">
          <h3>Run</h3>
          <RunCard s={started} />
          {started.error && <p className="bad-text">{started.error}</p>}
          <p className="muted small">Open answers from the sidebar as they arrive.</p>
        </section>
      )}
    </div>
  );
}

function Steps({ steps }: { steps: string[] }) {
  return (
    <div className="steps">
      {steps.map((s, i) => (s.startsWith("$ ") ? <Command key={i} cmd={s.slice(2)} /> : <p key={i}>{s}</p>))}
    </div>
  );
}

function Command({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="cmd">
      <code>{cmd}</code>
      <button
        onClick={() => {
          navigator.clipboard?.writeText(cmd).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
          });
        }}
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

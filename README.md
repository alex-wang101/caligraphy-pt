# Calligraphy agent: core

This repo holds the framework-free core for post-training a VLM to write Chinese characters stroke by stroke through tool calls, given only the character's name. It contains:
- data loaders;
- the stroke renderer;
- the SFT converter;
- the geometric scorer and reward;
- frozen data splits;
- a local viewer.

Training (SFT and GRPO on NeMo RL) runs in the cloud and imports this package. The design, the measured results and the remaining milestones are in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Setup

```bash
uv sync --all-extras
```

```bash
uv run python scripts/fetch_data.py
```

```bash
uv run pytest
```

`fetch_data.py` downloads Make Me a Hanzi (`graphics.txt` and `dictionary.txt`, about 33 MB) and calligraphy-bench (about 0.6 MB) into `data/`, which is gitignored.

## Using the core from a trainer

```python
from calligraphy.data.mmah import get_entry
from calligraphy.splits import load_splits
from calligraphy.tools import phase1_messages, format_strokes
from calligraphy.convert import convert
from calligraphy.score.reward import compute_reward

splits = load_splits()                     # train / val / test_compositional / viz16 / bench_excluded
entry = get_entry("永")
messages = phase1_messages("永", "yǒng")    # single-turn prompt: the name only, no image
sft_target = format_strokes(convert(entry))  # SFT label in the phase-1 JSON format
reward, info = compute_reward(sft_target, entry)  # reward v1 = structure; info logs every level
```

## Viewer

```bash
uv run uvicorn server.app:app --port 8000
```

```bash
pnpm --dir web install && pnpm --dir web dev
```

Open http://localhost:5173. The page shows each calligraphy-bench task in the 米字格 grid in three panels:
- the master's 64×64 reference;
- the dataset's canonical strokes, with order and direction;
- the engine's render of the converter output.

A shared stroke scrubber drives the panels. The page also has per-stroke metrics, and a drawing switcher that shows the scorer's golden scenes. Search finds any Make Me a Hanzi character. `.claude/launch.json` defines both servers.

## Running a model on the dataset

In the viewer, set **Dataset → Agent runs → New run**, pick a provider and follow its setup steps, then click **Test connection** and **Start run**. Each answer is scored with the training reward and opens in the panels.

The providers are:
- the scripted stub, which needs no model;
- local Ollama, LM Studio or llama.cpp;
- vLLM on a cloud GPU over an SSH tunnel;
- OpenAI;
- OpenRouter;
- Anthropic.

Keys come from the provider's environment variable in the API server's terminal, or can be pasted in the form. A pasted key is held in memory for the run only and is never saved. The same runner works from the command line:

```bash
uv run python scripts/run_agent.py --list-providers
```

```bash
uv run python scripts/run_agent.py --provider ollama --model qwen3:8b --dataset bench --limit 10
```

To check the pipeline without a model, start the stub (`uv run python scripts/stub_model.py`) and use `--provider stub --model converter`.

## Scripts

| Script | What it does |
|---|---|
| `scripts/flip_check.py` | Checks the y-flip against bench references and writes `out/flip_check.png`. |
| `scripts/bench_render.py` | Measures renderer throughput (stroke steps/s, characters/s, scoring masks). |
| `scripts/convert_report.py` | Gives converter IoU over the full dataset (naive vs fixed variants), using every core, in about 10 minutes. |
| `scripts/normalization_report.py` | Shows how the reward treats characters that are too small, too big, off-centre, squashed or rotated. Writes `out/normalization_<char>.png`. |
| `scripts/run_agent.py` | Runs a model on a dataset from the command line (the same runner as the viewer). |
| `scripts/stub_model.py` | A fake OpenAI-compatible model on port 8100 that answers with the converter output. |
| `scripts/make_splits.py` | Regenerates `splits/`. The splits are frozen and committed, so only rerun this deliberately. |

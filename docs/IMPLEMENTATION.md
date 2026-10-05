# Calligraphy Agent: Implementation Doc

This is a compressed version of "Implementation Plan v2" with the concrete engineering choices filled in. Code in this repo follows this doc. If the code has to deviate, update this doc in the same change.

**Goal:** post-train a VLM (Qwen3-VL family) to write a Chinese character stroke by stroke through tool calls, given only the character's name. It never sees a reference image. Training runs on NeMo RL in the cloud. This repo provides the framework-free core: data, renderer, converter, scorer and splits, plus a local viewer.

## 1. Fixed decisions

- **Frame:**
  - The canvas is 1024 × 1024, with the origin top-left and y pointing down.
  - Data in Make Me a Hanzi (MMAH) or bench format is converted once at load time with `(x, y) → (x, 900 − y)`.
  - The flip is applied to **every** coordinate: M/L endpoints, Q and C control points, and medians.
- **Action frame:** `CanvasConfig.action_frame` (default 1024) scales the agent's coordinates **and** widths into the canvas frame. 64 is a reasonable early setting. Widths are 20–88 canvas units, and a 64-grid cell is 16.
- **SVG:** outlines use absolute `M L Q C Z` with explicit parameters. This was verified over all 9,574 characters; `C` appears about 78k times.
- **Brushes:** both take `[[x, y, p], …]` plus a width.
  - **`disc` (default):**
    - A swept disc with radius `width/2 · p`.
    - Points are interpolated with centripetal Catmull-Rom at about 8-unit spacing, and pressure is interpolated linearly.
    - It is drawn as discs plus the tangent trapezoids between them. Every primitive is convex.
  - **`freehand` (secondary):** the PyPI `perfect-freehand` package with tldraw's pen preset. Even-odd fill artifacts are accepted.
- **Raster:**
  - Pillow at 2× supersampling, painted incrementally.
  - The 米字格 grid is composited under the ink, and the ink mask is kept separate from the display composite.
- **Episodes:**
  - **Phase 1:** single turn, text only. The model emits every stroke in one JSON response, and the environment renders and scores once.
  - **Phase 2:** multi-turn, with four tools: `create_stroke`, `delete_stroke`, `capture_canvas`, `submit` (section 3.7). The final canvas is scored exactly like a phase-1 answer.
  - `create_stroke` and `delete_stroke` return text only. Images come only from `capture_canvas`. `CanvasConfig.obs_size` defaults to 224 px (about 49 tokens); tools-mode runs use 448 px (about 200 tokens) so a misplaced stroke is visible.
  - There is no `update_stroke`: a stroke is changed by deleting it and drawing it again. No tool shows a reference image.
- **Core:**
  - `Canvas` and `tools.py` import no trainer code.
  - Python ≥ 3.12. The NeMo Gym requirement of 3.13.14 is gone now that Gym is out of scope.
- **Scoring inputs:**
  - Order, direction and shape metrics use the agent's **raw points** against the medians.
  - IoU uses masks. The target ink is rasterized from the reference outlines by the **same function at the same resolution** as the agent's ink: binary masks at 512 px.

## 2. Phase-1 action format (`tools.py`)

```json
{"strokes": [{"points": [[x, y, p], ...], "width": w}, ...]}
```

- Values are in `action_frame` units. `p` is in [0, 1].
- The parser accepts the object bare or inside a `` ```json `` fence.
- **Reward 0 (parse failure)** for any of:
  - JSON that won't parse or a missing `strokes` key;
  - a stroke with fewer than 1 or more than 16 points;
  - a point that isn't 3 finite numbers;
  - a width that isn't positive;
  - more than 64 strokes.
- Out-of-range coordinates, pressures and widths are **clamped**: x and y to [0, frame], p to [0, 1], width to [1, 256] canvas units.

## 3. Components

### 3.1 Data (`data/`)
- `CharacterEntry` holds `character`, `strokes_d` (canvas-frame path strings), `outlines` (canvas-frame polygons), `medians` (canvas frame), `meta` and an optional `reference_jpeg`.
- **MMAH** (`graphics.txt` + `dictionary.txt`, about 9.5k characters) is for training.
- **calligraphy-bench** (88 rows, 82 unique characters) is **eval only**. Its stroke data is byte-identical to MMAH for all 88 rows. It stores the canonical font skeleton, not the calligrapher's brushwork, and its reference images are 64 × 64.
- **Tests:**
  - every `graphics.txt` entry parses with `len(strokes) == len(medians)`;
  - every coordinate, including Q and C control points, maps to (x, 900 − y);
  - after both are cropped to their ink, the flipped render correlates with the reference image better than the raw-y render on at least 75% of bench rows (measured: 73/88). The misses are expressive scripts that don't match the font skeleton;
  - across MMAH, the first stroke starts above where the last stroke ends in at least 90% of characters (measured: 99%);
  - `scripts/flip_check.py` writes `out/flip_check.png` (reference | raw | flipped) for a visual check. 觀 is upright.

### 3.2 SFT converter (`convert.py`): dataset → `create_stroke` calls
1. **Trim** median endpoints that lie outside the outline. Then resample at **fixed spacing**, `spacing = 8`.
2. **Centerline:** recentre each sample on the midpoint of the outline chord along the local normal, which is a local medial axis. This runs for 2 iterations, with re-resampling.
   - A move is accepted only if half the chord is at most 1.25 × the inscribed radius at the new point.
   - At a corner, such as the turn of a 横折, the normal runs diagonally across the bend and the chord is far too long. Those samples stay where they are; moving them would cut the corner.
   - The positions are not smoothed, because smoothing rounds corners.
   - v2 suggested skeletonizing; this approach avoids skeleton spurs.
3. **End extension:** cast a ray along the end tangent out to the outline boundary.
   - The tangent is measured over the last ~24 units, so the small hooks that MMAH medians often have at their ends don't steer it sideways.
   - Extension is capped at 100 units.
4. Radius = the inscribed distance to the outline edge, multiplied by `radius_scale` (default 1.0). Samples outside the outline get 0.
5. Take `r_max` as the largest radius in the **whole character**. Then `p = r / r_max` and `width = 2 · r_max`. For 觀, `width ≈ 87`.
6. **Downsample:** Ramer–Douglas–Peucker over (x, y, r) with `rdp_eps = 2.0` (canvas units), raised until no more than 16 points remain. Endpoints are always kept. Then quantize: integer coordinates, `p` to 2 decimal places.
   - `rdp_eps` trades label fidelity against output length. Each point costs the model about 9 tokens.
   - The remaining gap to dense samples comes from Catmull-Rom re-interpolation, not from the point budget.

The `naive` preset reproduces v2's validation converter: 40 fixed points, no centerline, no extension. On 觀 it gives mean stroke IoU 0.88 with stroke 9 at 0.76, matching v2.

**Full-dataset report** (`scripts/convert_report.py`, 9,574 characters, 112,617 strokes; mask IoU at 512 px):

| Variant | Stroke mean | Stroke p10 | Character mean | Points/stroke |
|---|---|---|---|---|
| naive (v2 validation) | 0.906 | 0.837 | 0.935 | 40.0 |
| fixed spacing | 0.906 | 0.835 | 0.937 | 44.5 |
| + end extension | 0.909 | 0.841 | 0.937 | 48.4 |
| + centerline (dense) | **0.939** | 0.879 | **0.959** | 48.7 |
| **default** (≤ 16 points, `rdp_eps` 2.0, quantized) | 0.922 | 0.854 | 0.941 | **9.8** |
| high fidelity (`rdp_eps` 0.5) | 0.932 | 0.870 | 0.948 | 13.9 |
| default, radius × 1.05 | 0.906 | 0.849 | 0.922 | 9.9 |
| default, radius × 1.1 | 0.877 | 0.830 | 0.892 | 10.0 |

- The default beats naive on mean IoU, and on character IoU for 84.7% of characters. ✓
- The centerline is the fix that matters. End extension adds little.
- Once strokes are recentred they are no longer systematically thin, so scaling the radius up hurts. Keep `radius_scale = 1.0`.
- This IoU measures label and brush quality only; it does not cap the agent.

**Known systematic error:** corners come out round, because a disc can't form the squared shoulder of a pressed-and-turned brush. A later fix is an angled elliptical tip, for aesthetics only.

### 3.3 Scorer (`score/`)
- **Per-stroke metrics (`stroke.py`):**
  - `inside_frac`: the fraction of the agent's resampled points that fall inside the outline;
  - `iou`: mask IoU;
  - `frechet`: discrete Fréchet distance between the agent path and the median, both resampled to 32 points;
  - `direction_ok`: Fréchet against the median beats Fréchet against the reversed median.
- **Golden test (觀 stroke 1):**
  - good: inside ≈ 1, high IoU, low Fréchet, direction correct;
  - reversed: the same IoU but Fréchet above 100, direction wrong;
  - shifted by (+120, +80): inside 0, IoU 0;
  - width 60: much lower IoU, same Fréchet.
- **Alignment (`align.py`):**
  - Scale + translation (no rotation) on index-matched **per-stroke ink moments**: the centroid and RMS spread of each stroke's ink, measured on the same masks as IoU.
  - Least squares over `s·c_a + t = c_t` and `s·spread_a = spread_t`.
  - Why not path points: arc-length-paired path points were tried first and are biased. The converter's paths run to the stroke tips while the medians stop short, so the perfect drawing scored 0.86 on structure and 0.77 on shape instead of 0.90. Ink moments don't depend on parameterization.
  - **Robust version:** fit, keep the 60% of strokes with the lowest residual, refit, 3 iterations. Clamp `s` to [0.25, 4].
  - **Per-stroke fit (shape level):** match the spread, then the centroid.
- **Levels (`levels.py`):**
  - **Placement:** the offset between the ink bounding-box centres and the ratio of their longest sides, computed on the raw drawing. It is flagged when the offset is over 128 units (two cells of a 64-grid) or the ratio is outside [0.8, 1.25]. This is logged only and is not part of reward v1.
  - **Structure:** one robust transform for the whole character, then per-stroke IoU. Score = Σ IoU / max(n_agent, n_target).
  - **Stroke shape:** a separate transform per stroke, then IoU, with the same normalisation.
- **Golden scenes (觀),** with measured values:

  | Scene | Placement | Structure | Shape | Expected and v2 values |
  |---|---|---|---|---|
  | Perfect (converter output) | 2 · 0.99× | 0.899 | 0.897 | both ≈ the converter's IoU (0.90) ✓ |
  | A: 0.6× scale, in a corner | **258 · 0.60×** | 0.893 | 0.904 | placement only ✓ |
  | B: 見 slid 180 units left (strokes found through `dictionary.txt` `matches`) | 92 · 0.96× | **0.644** | 0.897 | structure only ✓ (v2: 93 offset, 0.64) |
  | C: straight strokes, flat pressure | 15 · 1.00× | 0.433 | **0.618** | shape ✓ (v2: 0.49 / 0.63) |

  - In scene B, the 18 unmoved strokes keep 0.78–0.94 and the 7 moved strokes score 0 (v2: 0.76–0.96 and about 0).
  - Plain least squares on B drifts by (108, 11) and scores structure 0.04.

- **Reward v1 (`reward.py`):**
  - `reward = structure`, matching agent stroke *i* to target stroke *i* by index. A parse failure gives 0.
  - Every level is returned in `info` for logging.
  - Add these only once the failure they fix is observed: a placement term, a direction check, Hungarian matching.
  - `info["clamped_points"]` counts the points the canvas pinned to its edge (logging only).
  - **Cost:** about 50 ms for a 25-stroke character on one core. Geometry is built once per stroke and transformed under the fit; target moments are cached on the entry.

### 3.3.1 Normalization: size, position, aspect, rotation
Run with `scripts/normalization_report.py`; the behavior is pinned in `tests/test_normalization.py`. Every scenario goes through the full reward path (JSON → parser → Canvas clamping → scorer). The viewer scores drawings the same way.

**How an off-centre or mis-sized character is handled today:**
1. **The canvas never moves anything.** Coordinates outside [0, `action_frame`] are clamped to the border, which flattens those strokes against the edge. Nothing is recentred. In phase 2, the model's own observation will show exactly what it drew.
2. **Placement** measures the raw drawing (bounding-box centre offset and size ratio) and flags it. It is logged only and costs no reward. It is measured after clamping, so a drawing far too big can look in-bounds; `info["clamped_points"]` is the tell.
3. **Structure (reward v1)** first fits one robust scale + translation from the agent's strokes to the target. Position and size are therefore normalized away completely, as long as the drawing stays on the canvas and the needed scale is within [0.25, 4].
4. **The alignment does no rotation or aspect correction**, so a tilted or squashed character is scored as wrong.

**Measured on 觀 (perfect drawing = 0.899):**

| Scenario | Reward | Placement | Clamped points |
|---|---|---|---|
| 0.5× centred | 0.892 | flagged (0.50×) | 0 |
| 0.6×, shifted right and down by 190 and 170 | 0.893 | flagged (offset 254) | 0 |
| 0.4×, top-left corner | 0.890 | flagged (offset 425) | 0 |
| 1.15× centred | 0.895 | ok | 5 |
| **0.2× centred** | **0.156** | flagged (0.20×) | 0 |
| **1.6× centred (off canvas)** | **0.701** | ok (reads 1.18× after clamping) | 46 |
| full size, shifted right 250 | 0.853 | flagged | 18 |
| squashed to 0.6× wide | 0.195 | ok | 0 |
| rotated 8° | 0.304 | ok | 0 |
| rotated 20° | 0.063 | ok | 1 |

永, 国 and 心 follow the same pattern.

**Known limits** (not changed; they're reward decisions):
- **A cliff below 0.25× size.** The scale clamp is the cause: with the clamp widened to [0.1, 10], 0.2× scores 0.87 and 0.1× scores 0.79.
- **Rotation is punished steeply.** At 8°, strokes far from the centre move more than their own half-width, so IoU drops by about half. The shape level is more forgiving (0.54–0.72).
- **Off-canvas strokes** cost reward only through clamping distortion. There is no explicit term for them.
- **Placement isn't rewarded.** A model could learn to draw small or off-centre at no cost. Add a small placement term only if training shows this (v2's rule).

### 3.4 Splits (`splits.py`, frozen under `splits/`)
Generated by `scripts/make_splits.py` with seed 0. The result is committed; don't regenerate it casually. `tests/test_splits.py` fails if the files, or the source data hashes in `meta.json`, drift.

| Set | Size | Rule |
|---|---|---|
| `bench_excluded` | 82 | Every calligraphy-bench character. Never trained on. |
| `val` | 300 | A uniform sample of non-bench CJK Unified Ideographs (U+4E00–U+9FFF), in-distribution. |
| `test_compositional` | 500 | A seeded greedy pick (detailed below). |
| `viz16` | 16 | `val` characters evenly spaced by stroke count, 丁 (2 strokes) to 髖 (23). Used for the validation image grid. |
| `train` | 8,692 | Everything else, including radical forms and 永, which is kept for the single-character sanity check. |

**Rules for `test_compositional`:**
- a CJK Unified Ideograph with a first-level IDS decomposition into two or more components, with no `？`;
- every component still appears in at least 2 training characters;
- no training character has the same combination;
- the character never appears inside any training character's decomposition. Without this rule, training on 清 would teach 青 directly.

When a child in the decomposition is an unnamed nested subtree, its leaves count as the components.

### 3.5 Viewer (`server/`, `web/`)
- **FastAPI** (`server/app.py`). Characters are addressed by bench `task_id` or by the character itself.
  - `GET /api/tasks`, `/api/search?q=`
  - `GET /api/char/{key}?drawing=&brush=` returns outlines, medians, strokes, per-stroke metrics, levels and, for a model answer, the rollout.
  - `GET /api/char/{key}/reference.jpg`
  - `GET /api/char/{key}/render.png?upto=&size=&brush=&grid=&drawing=`
  - `POST /api/render` renders arbitrary phase-1 strokes.
  - `GET /api/normalization` returns the 48 normalization cases with their rewards.
  - `/api/agent/*` serves agent runs (section 3.6).
  - A `drawing` is one of: the converter output, a scene, or a model answer (`run:<run_id>:<index>`). Every drawing is scored through the same parse-and-clamp path as the reward.
- **React/Vite** (`web/`). A **Dataset** dropdown at the top of the sidebar switches between:
  - **calligraphy-bench:** the 88 tasks, with filters and MMAH search;
  - **Normalization tests:** 4 characters × 12 scenarios, filterable by character and kind (baseline, size, position, shape). Each case shows its reward;
  - **Agent runs:** past runs with progress and mean reward, a **New run** form, and each run's answers.
- **Main view:**
  - three 米字格 panels: the 64 × 64 reference, the canonical strokes with order and direction, and the engine render with optional contours and agent points;
  - a shared stroke scrubber;
  - score tiles: structure, shape, placement, alignment, strokes, clamped points;
  - a per-stroke table;
  - for a model answer, a panel with the reward, parse status and raw response.
- Works down to phone width.

### 3.6 Agent runs (`agent/`, `scripts/run_agent.py`, `scripts/stub_model.py`)
- **What a run does:** a model answers the phase-1 prompt (`phase1_messages`: the character's name only) for every character in a dataset, and each answer is scored with `compute_reward`. This is the zero-shot baseline from the training plan, and the same harness will evaluate trained checkpoints.
- **Two backends** (`agent/client.py`):
  - **OpenAI-compatible HTTP** (httpx): local Ollama, LM Studio, llama.cpp, vLLM (on a cloud GPU over an SSH tunnel), OpenAI, OpenRouter. If a hosted reasoning model rejects `max_tokens` or `temperature`, the request is retried with `max_completion_tokens` or without temperature.
  - **Anthropic** through the official SDK. Credentials come from `ANTHROPIC_API_KEY` or an `ant auth login` profile. No temperature is sent (current models reject it), and the server-side refusal fallback (`fallbacks: "default"`) is on for the models that support it.
- **Presets** (`agent/providers.py`): each carries its setup steps, which the viewer and `run_agent.py --list-providers` both show. `stub` (`scripts/stub_model.py`, port 8100) is a fake model that answers with the converter output; use it to test the pipeline (mean reward about 0.9).
- **Keys:** a key is read from the preset's environment variable in the API server's process (the server and `run_agent.py` also load the repo's gitignored `.env`; variables already set in the shell win), or pasted in the UI. A pasted key lives only in memory for that run. It is never written to `out/runs/` and never returned by the API (`SecretStr`). The UI only shows whether the environment variable is set.
- **Datasets:** bench (88, eval only), `viz16`, `val`, `test_compositional`, or custom characters, each with an optional limit and concurrency.
- **Storage:** `out/runs/<run_id>/summary.json` (settings, progress, mean reward, parse rate, errors) and `rollouts.jsonl` (raw response, reward info, latency, usage). A run still marked running after a server restart shows as `interrupted`.
- **Robustness:** errors never stop a run; each one becomes a zero-reward rollout. The parser skips `<think>…</think>` blocks, which local reasoning models emit.
- **Rate limits:** a transient HTTP 429 is retried after 10, 30 and 60 s (or `Retry-After`). A 429 that persists, or a daily cap (`per-day`), stops the run with status `rate_limited` and is **not** written as a zero.
- **Resume:** `run_agent.py --resume <run_id>` reruns unanswered items and rows with a transport `error` into the same run, keeping every scored answer. The summary stores `characters`, `extra_body` and a `resumed` count so a run can be rebuilt.
- **Provider options:** `--extra-body` JSON is merged into the request and saved in the summary, e.g. `{"reasoning": {"enabled": false}}` to switch off OpenRouter thinking. Separate reasoning text (`message.reasoning`) is kept in each rollout as `reasoning`.
- **Report:** `scripts/run_report.py <run_id>` prints parse rate, reward, stroke-count accuracy and placement next to a ceiling (converter answer), a wrong-character floor (another character's converter answer with the closest stroke count) and a random-strokes floor, and writes `report.json` and `sheet.png` (reference | model) into the run folder.

### 3.7 Phase 2: tools, episodes and traces (`tools.py`, `canvas.py`, `episode.py`)
- **Tools (`tools.py`).** Each tool is one `Tool` object: name, the description the model reads, the argument schema and the function that runs it. `TOOLS` lists them in order. The schemas sent to the model, the tool list in the system prompt (`phase2_messages`) and `dispatch` are all built from `TOOLS`, so a tool is added or changed in one place. The canvas holds the drawing; the tools only translate a call into a canvas operation and a text result.

  | Tool | Arguments | Result |
  |---|---|---|
  | `create_stroke` | `points`, `width` (as in phase 1) | `stroke N created; the canvas has K strokes` |
  | `delete_stroke` | `stroke_id` | `stroke N deleted; strokes on the canvas: 1-2, 4-10. The next stroke you create becomes stroke N.` |
  | `capture_canvas` | none | the canvas image with the 米字格 grid |
  | `submit` | none | ends the episode |

  A rejected call (bad JSON, malformed stroke, unknown stroke id, unknown tool) returns `error: …`, changes nothing, and the episode goes on.
- **Stroke numbers (`canvas.py`).** Strokes live in numbered slots, from 1. Deleting a stroke empties its slot without renumbering the others. The next `create_stroke` fills the lowest empty slot, so a redrawn stroke keeps its place in the stroke order and index-matched scoring stays meaningful.
  - A gap that is never refilled closes at scoring time: the reward sees the strokes left on the canvas, in slot order. Deleting a middle stroke without redrawing it therefore shifts every later stroke by one under index matching.
  - Known limit: a model that deletes a spurious middle stroke and then draws a *new* stroke will see it land in the old slot. Trailing strokes don't have this problem.
- **Episode (`episode.py`).** `Episode.call(name, arguments)` runs one tool call and records it; the driver also reports each model reply. Framework-free, so the agent runner and a trainer produce the same trace. Limits (`EpisodeConfig`): 64 model replies, 128 tool calls, 2 nudges.
  - It ends by `submit`, `max_turns`, `max_tool_calls`, `no_tool_call` (a reply without a tool call, after 2 reminders), `error` or `rate_limited`. However it ends, the canvas is scored as it stands.
  - The final canvas is serialized in the phase-1 format and goes through `compute_reward`, so both phases share one reward path and the viewer opens a phase-2 answer unchanged.
- **Trace events,** in order, each with an index `i`, the model reply number `turn` and seconds since the start `t_s`:
  - `model`: the reply's text, reasoning, stop reason, latency, token usage, number of tool calls, and any transport error;
  - `tool`: name, parsed `arguments` (or `arguments_raw` if the JSON was invalid), `ok`, the `result` text, `stroke_id`, and the `stroke_ids` on the canvas afterwards. A `create_stroke` also has `redraw` (it filled a deleted slot). A `delete_stroke` has `deleted_stroke_score` (how good the removed stroke was). A `capture_canvas` has `image`, the saved PNG the model was shown. A call that arrives after the episode ended is recorded with `skipped`;
  - `nudge` and `end`.
- **Step scores.** After every accepted `create_stroke` or `delete_stroke`, the event's `score` holds the structure and shape scores of the canvas as it stands (what it would get if submitted then) and each stroke's share by stroke id. They are logged only and never shown to the model. They jump around on partial canvases: with two or three strokes the alignment is fragile, and between a delete and its redraw every later stroke is off by one. Compare the draft with the final canvas, not step with step.
- **Episode summary** (`trace_summary` in `rollouts.jsonl`): end reason, turns, tool calls by name, rejected and skipped calls, captures, deletes, redraws, nudges, strokes left, `checked_final` (a capture after the last change), `draft_score` (just before the first delete), `peak_score` and `final_score`.
- **Files** (`--mode tools`): `rollouts.jsonl` holds the final answer, reward and `trace_summary`; `traces/<index>.json` holds the system prompt, the tool schemas, the limits and every event; `traces/<index>/step-NNN.png` holds each image the model was shown. Reasoning text lives only in the trace file, which keeps `rollouts.jsonl` small.
- **Transcript.** Kept in the OpenAI chat format for both backends: tool results are `tool` messages, and each canvas image follows the turn's tool results as a user message with an `image_url` part. The Anthropic backend converts this to one user turn of `tool_result` blocks followed by the image, and sends the API's own content blocks (thinking included) back unchanged. It is covered by a conversion test but has not been run against the live API.
- **Scripts.** `run_agent.py --mode tools [--max-turns N --max-tool-calls N --obs-size PX]`; `show_trace.py <run_id> <index|character>` prints an episode step by step and writes a filmstrip of what the model saw; `run_report.py` adds tool-use and self-correction metrics for tools-mode runs. `stub_model.py` plays a scripted agent that draws every stroke with one out of place, looks, deletes and redraws it, looks again and submits.
- **Replay.** `episode.replay(events, upto)` rebuilds the canvas as it stood after any event from the trace alone. The state before tool call *i* is `replay(events, i - 1)` and the state after it is `replay(events, i)`. A test checks that the rebuilt canvas has the recorded stroke numbers after every call and is pixel-identical to each image the model was shown.
- **API.** `POST /api/agent/runs` takes `mode`, `max_turns`, `max_tool_calls` and `obs_size`.
  - `GET /api/agent/runs/{run_id}/trace/{index}` returns the trace.
  - `GET …/trace/{index}/image/{name}` returns a saved image, exactly as the model was shown it.
  - `GET …/trace/{index}/state?step=&size=&mark=&tone=` renders the canvas after event `step` (−1: empty). `mark` draws one stroke in colour: blue for the stroke a call just drew (`tone=added`), red for the one it is about to delete (`tone=removed`).
- **Viewer.** A multi-turn answer has a **Tool calls** section under the score tiles. It lists every model reply and tool call in order. Each call shows its arguments, the result text the model got back, the strokes on the canvas and the score before → after, and two canvas images: before the call and after it. A `capture_canvas` row shows the saved image as the model saw it. The prompt and any thinking text are in collapsible blocks. The New run form has a "How the model answers" choice, so tools-mode runs can be started from the page.
- **A run started from the command line** now shows as `running` in the viewer while its process is alive (the summary records the `pid`), instead of `interrupted`.

## 4. Build order

| M | Deliverable | Exit |
|---|---|---|
| M0 | Data, svgpath, brushes, raster, Canvas, tools | 3.1 tests; 觀 is upright; renders are deterministic; incremental equals full; `bench_render.py` (about 2.3k stroke steps/s per core at 512 px) ✓ |
| M1 | Converter | `convert_report.py`: fixed beats naive on the full dataset ✓ |
| M2 | Scorer and reward v1 | 3.3 golden tests ✓ |
| Viewer | Server and UI | 觀 scrubs 25 strokes; stroke 1 runs from about (245, 110) to (308, 255) ✓ |
| M3 | Frozen splits | 3.4 tests; committed ✓ |

**Out of scope here:**
- NeMo Gym environment (dropped).
- Image spike (M5).
- SFT (M6) and GRPO (M7) on NeMo RL.
- Multi-turn training (M8). The environment side (tools, episodes, traces, evaluation harness) is built: section 3.7.
- VLM aesthetics judge (M9).

The v2 notes still apply to those milestones:
- LoRA rank 16–64 on the language layers;
- learning rate about 1e-4 for SFT and 5e-6 to 1e-5 for GRPO;
- group size 8, temperature about 1.0;
- drop prompts where every sample in the group scores the same.

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
  - `create_stroke` returns text only. Images come only from `capture_canvas`, which belongs to phase 2. Its default size is 224 px, about 49 tokens.
  - `update_stroke` and `delete_strokes` are deferred to phase 2.
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
1. Resample each median at **fixed spacing**, `spacing = 8`.
2. **Centerline:** recentre each sample on the midpoint of the outline chord along the local normal, which is a local medial axis. v2 suggested skeletonizing; this approach avoids skeleton spurs. The fallback is skimage `medial_axis`.
3. **End extension:** cast a ray along the end tangent out to the outline boundary. Trim endpoints that lie outside the outline.
4. Radius = the inscribed distance to the outline edge, multiplied by `radius_scale` (default 1.0). Samples outside the outline get 0.
5. Take `r_max` as the largest radius in the **whole character**. Then `p = r / r_max` and `width = 2 · r_max`. For 觀, `width ≈ 88`.
6. Downsample to **≤ 16 points**: Ramer–Douglas–Peucker over (x, y, r), raising epsilon until no more than 16 points remain. Endpoints are always kept. Then quantize: integer coordinates, `p` to 2 decimal places.

The `naive` preset reproduces v2's validation converter: 40 fixed points, no centerline, no extension.

**Report:** `scripts/convert_report.py` gives per-stroke and per-character IoU distributions over the full dataset, for naive and fixed, plus a `radius_scale` sweep (1.0, 1.05, 1.1). Fixed must beat naive on mean IoU. v2 reported 0.89 on 觀 for naive. This IoU measures label and brush quality only; it does not cap the agent.

**Known systematic errors:** strokes always come out slightly thin, and corners come out round. A later fix is an angled elliptical tip, for aesthetics only.

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
  - Scale + translation (no rotation) in closed form, on index-matched point pairs: each agent stroke and the target median, both resampled to 32 points.
  - **Robust version:** fit, keep the 60% of strokes with the lowest residual, refit, 3 iterations. Clamp `s` to [0.25, 4].
- **Levels (`levels.py`):**
  - **Placement:** the offset between the bounding-box centres and the size ratio, computed on the raw drawing.
  - **Structure:** one robust transform for the whole character, then per-stroke IoU. Score = Σ IoU / max(n_agent, n_target).
  - **Stroke shape:** a separate transform per stroke, then IoU, with the same normalisation.
- **Golden scenes (觀):**

  | Scene | Expected |
  |---|---|
  | Perfect (converter output) | Placement OK; structure ≈ shape ≈ the converter's IoU. |
  | A: small and in a corner | Only placement is flagged. |
  | B: the 見 strokes slid left (strokes found through `dictionary.txt` `matches`) | Only structure is flagged. The unmoved strokes keep their IoU and the moved ones score about 0. |
  | C: straight strokes with flat pressure | Shape is flagged. |

- **Reward v1 (`reward.py`):**
  - `reward = structure`, matching agent stroke *i* to target stroke *i* by index. A parse failure gives 0.
  - Every level is returned in `info` for logging.
  - Add these only once the failure they fix is observed: a placement term, a direction check, Hungarian matching.

### 3.4 Splits (`splits.py`, committed under `splits/`)
- **Exclusion:** all 82 bench characters are removed from every training set.
- **Compositional test:**
  - Candidates have a first-level IDS decomposition into two or more known components (no `？`).
  - Selection is greedy over a seeded shuffle. A candidate is accepted only if every component still appears in at least 2 training characters, and no training character has the same (operator, components).
- **Other sets:** a fixed validation set (in-distribution), `viz16` (16 validation characters for image grids) and `meta.json` (seed, sizes, criteria, sha256 of the source files).
- **Tests:** the sets are disjoint; no bench character appears in train or val; every test component is seen in train; no combination is shared; the split is reproducible.

### 3.5 Viewer (`server/`, `web/`)
- FastAPI over the core. The React/Vite UI has:
  - a task list (bench 88, with MMAH search);
  - three 米字格 panels: the 64 × 64 reference, the canonical strokes with order and direction, and the engine render of the converter output;
  - a shared stroke scrubber;
  - per-stroke IoU, Fréchet and direction.

## 4. Build order

| M | Deliverable | Exit |
|---|---|---|
| M0 | Data, svgpath, brushes, raster, Canvas, tools | 3.1 tests; 觀 is upright; renders are deterministic; incremental equals full; `bench_render.py` |
| M1 | Converter | `convert_report.py`: fixed beats naive on the full dataset |
| M2 | Scorer and reward v1 | 3.3 golden tests |
| Viewer | Server and UI | 觀 scrubs 25 strokes; stroke 1 runs from about (245, 110) to (308, 255) |
| M3 | Frozen splits | 3.4 tests; committed |

**Out of scope here:**
- NeMo Gym environment (dropped).
- Image spike (M5).
- SFT (M6) and GRPO (M7) on NeMo RL.
- Multi-turn (M8).
- VLM aesthetics judge (M9).

The v2 notes still apply to those milestones:
- LoRA rank 16–64 on the language layers;
- learning rate about 1e-4 for SFT and 5e-6 to 1e-5 for GRPO;
- group size 8, temperature about 1.0;
- drop prompts where every sample in the group scores the same.

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
  - **Cost:** about 50 ms for a 25-stroke character on one core. Geometry is built once per stroke and transformed under the fit; target moments are cached on the entry.

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

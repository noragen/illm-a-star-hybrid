# iLLM-A*-Inspired Hybrid Path Planner

A minimal, **practical** hybrid path planner that combines classic **A\*** with
an **LLM-based waypoint proposer**. Supports **OpenAI**, **Ollama**, and **Hugging Face**
backends out of the box. Includes a robust JSON-only prompt, fallback heuristics, and a
simple CLI plus a small benchmark script.

> Repo owner: **noragen**

---

## Features

- 🧠 **LLM Waypoint Proposer** (OpenAI / Ollama / Hugging Face)
- 🛟 **Automatic Fallback** to a local heuristic proposer if the LLM fails or is unavailable
- ⚡ **Optimized A\*** (binary heap, hash-set closed, lazy heuristics)
- 🧪 **CLI Demo** and **benchmark.py** for quick comparisons
- 🧰 Single-file core (`illm_astar_llm.py`) for easy reuse

---

## Quickstart

```bash
git clone https://github.com/noragen/illm-a-star-hybrid.git
cd illm-a-star-hybrid
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run (OpenAI)
```bash
# set your key first
# Windows (PowerShell):
#   setx OPENAI_API_KEY "<YOUR_KEY>"
# Linux/macOS:
#   export OPENAI_API_KEY="<YOUR_KEY>"

python illm_astar_llm.py --provider openai --model gpt-4o-mini
```

### Run (Ollama, local)
```bash
# Make sure Ollama is running, e.g.:
#   ollama serve
# Pull a model (example):
#   ollama pull mistral

python illm_astar_llm.py --provider ollama --model mistral
```

### Run (Hugging Face Inference API)
```bash
# Windows:
#   setx HUGGINGFACEHUB_API_TOKEN "<YOUR_TOKEN>"
# Linux/macOS:
#   export HUGGINGFACEHUB_API_TOKEN="<YOUR_TOKEN>"

python illm_astar_llm.py --provider hf --model state-spaces/mamba-2.8b
```

**Tip:** If the LLM returns anything but a clean JSON array like `[[x,y],[x,y]]`,
the parser tries to recover. If recovery fails, the system automatically falls
back to the heuristic proposer — your run keeps working.

---

## Example Output

```
Pfad-Länge: 93 | Laufzeit: 12.4 ms | LLM: hf
....................................................########
..S**********.......................................########
...***********......................................########
....************....................................########
.....*************..................................########
......**************................................########
.......***************..............................########
........****************............................########
.........*****************..........................########
..........******************........................########
...........*******************......................########
............********************....................########
.............*********************..................########
..............**********************................########
...............***********************..............########
................************************............########
.................*************************..........########
..................**************************........########
...................***************************......########
....................****************************....########
.....................*****************************..########
......................******************************.########
.......................******************************########
........................*****************************########
.........................****************************########
..........................***************************########
...........................**************************########
............................*************************########
.............................************************########
..............................***********************#######G
```

(ASCII map with obstacles `#`, path `*`, start `S`, goal `G`)

---

## Files

- `illm_astar_llm.py` – core hybrid planner (OpenAI/Ollama/HF clients + A* + fallback)
- `benchmark.py` – quick A* vs. Hybrid comparison (uses heuristic fallback by default)
- `examples/sample_map.json` – tiny sample map for experimentation
- `benchmarks/` – place for results (ignored by git via `.gitignore`)

---

## Benchmark (quick taste)

```bash
python benchmark.py --maps 5 --width 80 --height 40 --density 0.10
```

Outputs a table with average times. By default, the benchmark compares:
- **A\*** (pure) vs. **Hybrid (heuristic only)** — deterministic and offline.
To benchmark a real LLM, pass `--provider`/`--model` to `benchmark.py`.

---

## Notes
- The LLM output is forced to **pure JSON**; a robust parser extracts the first JSON array if the model adds text.
- Waypoints are validated, clamped to grid bounds, and filtered for passability.

---

## References
- iLLM-A* concept & discussion (incremental LLM-driven subgoal selection).
- Mamba 2.8b (state-spaces) for efficient sequence modeling.
- A* algorithmic optimizations (heap-based OPEN, hash-closed, lazy heuristic evaluation).

If you use this, a star ⭐ on GitHub would be awesome!
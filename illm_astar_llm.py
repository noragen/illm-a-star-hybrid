from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable, Any
import heapq
import math
import random
import json
import os
import time
import argparse


try:
    import requests  # für HF & Ollama HTTP-APIs
except Exception:
    requests = None

Coord = Tuple[int, int]

# ================================
# Grid & Geometrie
# ================================
@dataclass
class GridMap:
    width: int
    height: int
    obstacles: List[Tuple[int, int, int, int]]  # (x0,y0,x1,y1) inkl. Grenzen

    def in_bounds(self, p: Coord) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, p: Coord) -> bool:
        x, y = p
        for x0, y0, x1, y1 in self.obstacles:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return False
        return True

    def neighbors(self, p: Coord) -> Iterable[Coord]:
        x, y = p
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            n = (x+dx, y+dy)
            if self.in_bounds(n) and self.passable(n):
                yield n

    def segment_intersects_any_obstacle(self, a: Coord, b: Coord) -> bool:
        ax, ay = a; bx, by = b
        minx, maxx = min(ax, bx), max(ax, bx)
        miny, maxy = min(ay, by), max(ay, by)
        for x0, y0, x1, y1 in self.obstacles:
            if maxx < x0 or minx > x1 or maxy < y0 or miny > y1:
                continue
            if self._bresenham_hits_rect(a, b, (x0, y0, x1, y1)):
                return True
        return False

    @staticmethod
    def _bresenham_hits_rect(a: Coord, b: Coord, rect: Tuple[int,int,int,int]) -> bool:
        (x0, y0), (x1, y1) = a, b
        rx0, ry0, rx1, ry1 = rect
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            if rx0 <= x <= rx1 and ry0 <= y <= ry1:
                return True
            if (x, y) == (x1, y1):
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return False

# ================================
# A* (optimiert)
# ================================
class AStar:
    def __init__(self, grid: GridMap, heuristic: str = "manhattan"):
        self.grid = grid
        self.heuristic = heuristic

    def h(self, a: Coord, b: Coord) -> float:
        if self.heuristic == "euclid":
            return math.hypot(a[0]-b[0], a[1]-b[1])
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def search(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        if not self.grid.passable(start) or not self.grid.passable(goal):
            return None
        open_heap: List[Tuple[float, int, Coord]] = []
        g: Dict[Coord, float] = {start: 0.0}
        came: Dict[Coord, Coord] = {}
        closed: set[Coord] = set()
        counter = 0
        heapq.heappush(open_heap, (self.h(start, goal), counter, start))
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct(came, current)
            closed.add(current)
            for n in self.grid.neighbors(current):
                if n in closed:
                    continue
                tentative = g[current] + 1.0
                if tentative < g.get(n, float('inf')):
                    came[n] = current
                    g[n] = tentative
                    counter += 1
                    f = tentative + self.h(n, goal)
                    heapq.heappush(open_heap, (f, counter, n))
        return None

    @staticmethod
    def _reconstruct(came: Dict[Coord, Coord], current: Coord) -> List[Coord]:
        path = [current]
        while current in came:
            current = came[current]
            path.append(current)
        path.reverse()
        return path

# ================================
# Baseline Waypoint-Proposer (heuristisch)
# ================================
class HeuristicWaypointProposer:
    """
    Leichter, lokaler Proposer als Fallback/Debug: prüft Sichtlinie S->G; falls blockiert,
    nimmt Ecken des nächstliegenden Hindernisses und bewertet sie.
    """
    def __init__(self, grid: GridMap):
        self.grid = grid

    def propose(self, start: Coord, goal: Coord, max_wp: int = 2) -> List[Coord]:
        if not self.grid.segment_intersects_any_obstacle(start, goal):
            return []
        sx, sy = start; gx, gy = goal
        line_minx, line_maxx = min(sx, gx), max(sx, gx)
        line_miny, line_maxy = min(sy, gy), max(sy, gy)
        candidates: List[Tuple[float, Tuple[int,int,int,int]]] = []
        for rect in self.grid.obstacles:
            x0,y0,x1,y1 = rect
            if line_maxx < x0 or line_minx > x1 or line_maxy < y0 or line_miny > y1:
                continue
            cx, cy = (x0+x1)/2.0, (y0+y1)/2.0
            d = self._point_to_segment_dist((cx,cy), (sx,sy), (gx,gy))
            candidates.append((d, rect))
        if not candidates:
            return []
        candidates.sort(key=lambda t: t[0])
        block = candidates[0][1]
        x0,y0,x1,y1 = block
        corners = [(x0-1,y0-1),(x0-1,y1+1),(x1+1,y0-1),(x1+1,y1+1)]
        def score(pt: Coord) -> float:
            if not self.grid.in_bounds(pt) or not self.grid.passable(pt):
                return float('inf')
            vis_start = not self.grid.segment_intersects_any_obstacle(start, pt)
            vis_goal = not self.grid.segment_intersects_any_obstacle(pt, goal)
            penalty = 0.0
            if not vis_start: penalty += 10.0
            if not vis_goal: penalty += 3.0
            return penalty + abs(pt[0]-start[0])+abs(pt[1]-start[1]) + 0.5*(abs(pt[0]-goal[0])+abs(pt[1]-goal[1]))
        corners.sort(key=score)
        waypoints: List[Coord] = []
        for c in corners:
            sc = score(c)
            if math.isinf(sc):
                continue
            waypoints.append(c)
            if len(waypoints) >= max_wp:
                break
        return waypoints

    @staticmethod
    def _point_to_segment_dist(p: Tuple[float,float], a: Coord, b: Coord) -> float:
        (px,py), (ax,ay), (bx,by) = p, a, b
        abx, aby = bx-ax, by-ay
        apx, apy = px-ax, py-ay
        ab2 = abx*abx + aby*aby
        if ab2 == 0:
            return math.hypot(apx, apy)
        t = max(0.0, min(1.0, (apx*abx + apy*aby)/ab2))
        cx, cy = ax + t*abx, ay + t*aby
        return math.hypot(px-cx, py-cy)

# ================================
# LLM-Clients (OpenAI / Ollama / Hugging Face)
# ================================
class LLMClient:
    def generate_waypoints(self, context: Dict[str, Any]) -> List[Coord]:
        raise NotImplementedError

class OpenAIClient(LLMClient):
    """OpenAI-Client über das offizielle Python-SDK (`openai`).
    Erwartet OPENAI_API_KEY; optional OPENAI_BASE_URL/OPENAI_ORG/OPENAI_PROJECT.
    Der Prompt erzwingt JSON-Only-Output.
    """
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, organization: str = None, project: str = None):
        try:
            from openai import OpenAI  # offizielles SDK
        except Exception as e:
            raise RuntimeError("Python-Paket 'openai' fehlt. Installiere mit: pip install openai") from e
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY fehlt.")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        organization = organization or os.environ.get("OPENAI_ORG")
        project = project or os.environ.get("OPENAI_PROJECT")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        if project:
            client_kwargs["project"] = project
        self._client = OpenAI(**client_kwargs)

    def generate_waypoints(self, context: Dict[str, Any]) -> List[Coord]:
        prompt = self._build_prompt(context)
        messages = [
            {"role": "system", "content": "You are a routing planner. Always output pure JSON only."},
            {"role": "user", "content": prompt},
        ]
        last_err = None
        for attempt in range(5):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                )
                txt = resp.choices[0].message.content
                return parse_waypoints_from_text(txt)
            except Exception as e:
                last_err = e
                # einfacher Backoff bei RateLimit/Transient-Fehlern
                time.sleep(min(2 ** attempt, 15))
        raise last_err if last_err else RuntimeError("OpenAI request failed")

    @staticmethod
    def _build_prompt(ctx: Dict[str, Any]) -> str:
        return (
            "Return up to TWO grid waypoints to help A* path planning."
            "Respond with a JSON array of coordinate pairs ONLY, e.g. [[x,y],[x,y]]. No prose."
            f"Grid: {ctx['width']}x{ctx['height']}."
            f"Start: {ctx['start']}. Goal: {ctx['goal']}."
            f"Obstacles (rectangles x0,y0,x1,y1): {ctx['obstacles']}"
            "Rules: Prefer corners that open line-of-sight from start and reduce distance to goal."
        )

class OllamaClient(LLMClient):
    """Ollama unter http://localhost:11434. Benötigt requests. Modell z. B. 'mistral' oder 'phi3'."""
    def __init__(self, model: str = None, host: str = None):
        from langchain_ollama import ChatOllama
        self.model = ChatOllama(model = (model or os.environ.get("OLLAMA_MODEL", "mistral")), temperature=0.0)
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate_waypoints(self, context: Dict[str, Any]) -> List[Coord]:
        url = f"{self.host}/api/generate"
        prompt = self._build_prompt(context)
        response_message = self.model.invoke(prompt)
        txt = response_message.content
        return parse_waypoints_from_text(txt)

    @staticmethod
    def _build_prompt(ctx: Dict[str, Any]) -> str:
        return (
            "You are a routing planner. Output JSON only: [[x,y],[x,y]] (0,1 or 2 pairs)."
            f"Grid {ctx['width']}x{ctx['height']}; Start {ctx['start']}; Goal {ctx['goal']}; Obstacles {ctx['obstacles']}."
        )

class HFClient(LLMClient):
    """Hugging Face Inference-API (Text Generation). Z. B. state-spaces/mamba-2.8b.

    Setze HUGGINGFACEHUB_API_TOKEN und HF_MODEL (default: state-spaces/mamba-2.8b).
    """
    def __init__(self, model: str = None, api_token: str = None):
        if not requests:
            raise RuntimeError("'requests' nicht installiert")
        self.model = model or os.environ.get("HF_MODEL", "state-spaces/mamba-2.8b")
        self.token = api_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not self.token:
            raise RuntimeError("HUGGINGFACEHUB_API_TOKEN fehlt.")

    def generate_waypoints(self, context: Dict[str, Any]) -> List[Coord]:
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.token}"}
        prompt = self._build_prompt(context)
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 64, "temperature": 0}}
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # HF-API hat unterschiedliche Rückgabeformen; wir extrahieren Text robust
        if isinstance(data, list) and data and "generated_text" in data[0]:
            txt = data[0]["generated_text"][len(prompt):]
        elif isinstance(data, dict) and "generated_text" in data:
            txt = data["generated_text"]
        else:
            # Fallback: rohe Antwort in Text
            txt = json.dumps(data)
        return parse_waypoints_from_text(txt)

    @staticmethod
    def _build_prompt(ctx: Dict[str, Any]) -> str:
        return (
            "Task: Output up to TWO waypoints to guide A* path planning on a discrete grid."
            "Return JSON ONLY: [[x,y],[x,y]] or [] — no explanations."
            f"Grid: {ctx['width']}x{ctx['height']}; Start: {ctx['start']}; Goal: {ctx['goal']}; Obstacles: {ctx['obstacles']}."
            "Prefer corners enabling visibility and shorter Manhattan distance to goal."
        )

# ================================
# Hybrid-Proposer, der LLM nutzt (mit Fallback)
# ================================
class HybridWaypointProposer:
    def __init__(self, grid: GridMap, llm_client: Optional[LLMClient] = None, fallback: Optional[HeuristicWaypointProposer] = None):
        self.grid = grid
        self.llm = llm_client
        self.fallback = fallback or HeuristicWaypointProposer(grid)

    def propose(self, start: Coord, goal: Coord, max_wp: int = 2) -> List[Coord]:
        # 1) Wenn direkte Sichtlinie frei: keine Wegpunkte nötig
        if not self.grid.segment_intersects_any_obstacle(start, goal):
            return []
        # 2) Versuche LLM
        if self.llm is not None:
            ctx = {
                "width": self.grid.width,
                "height": self.grid.height,
                "start": list(start),
                "goal": list(goal),
                "obstacles": [list(t) for t in self.grid.obstacles],
            }
            try:
                wps = self.llm.generate_waypoints(ctx)
                wps = [tuple(map(int, wp)) for wp in wps if isinstance(wp, (list, tuple)) and len(wp) == 2]
                # Validierung & Clipping
                valid: List[Coord] = []
                for (x,y) in wps:
                    pt = (max(0, min(self.grid.width-1, x)), max(0, min(self.grid.height-1, y)))
                    if self.grid.passable(pt):
                        valid.append(pt)
                    if len(valid) >= max_wp:
                        break
                if valid:
                    return valid
            except Exception as e:
                print(f"[LLM-Proposer] Fehler, verwende Fallback: {e}")
        # 3) Fallback
        return self.fallback.propose(start, goal, max_wp=max_wp)

# ================================
# Pipeline: (Start) -> (<=2 Wegpunkte) -> (Ziel)
# ================================

def plan_path(grid: GridMap, start: Coord, goal: Coord, llm: Optional[LLMClient] = None) -> Optional[List[Coord]]:
    proposer = HybridWaypointProposer(grid, llm_client=llm)
    waypoints = proposer.propose(start, goal, max_wp=2)
    sequence = [start] + waypoints + [goal]
    astar = AStar(grid)
    full_path: List[Coord] = []
    for a, b in zip(sequence, sequence[1:]):
        sub = astar.search(a, b)
        if not sub:
            return None
        if full_path and sub and full_path[-1] == sub[0]:
            full_path.extend(sub[1:])
        else:
            full_path.extend(sub)
    return full_path

# ================================
# Utility: ASCII-Renderer & Parser
# ================================

def render_ascii(grid: GridMap, path: Optional[List[Coord]], start: Coord, goal: Coord):
    S, G = start, goal
    path_set = set(path) if path else set()
    rows = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            ch = '.'
            if not grid.passable((x,y)):
                ch = '#'
            if (x,y) in path_set:
                ch = '*'
            if (x,y) == S:
                ch = 'S'
            if (x,y) == G:
                ch = 'G'
            row.append(ch)
        rows.append(''.join(row))
    for r in rows:
        print(r)

def parse_waypoints_from_text(txt: str) -> List[Coord]:
    """Extrahiere eine JSON-Array-Struktur aus beliebigem LLM-Text. Robust gegen Prosa."""
    # Versuche direkt zu parsen
    def _try(s: str) -> Optional[List[Coord]]:
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                out: List[Coord] = []
                for item in arr:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        x, y = int(item[0]), int(item[1])
                        out.append((x, y))
                return out
        except Exception:
            return None
        return None
    # 1) direkter Versuch
    out = _try(txt.strip())
    if out is not None:
        return out
    # 2) Heuristik: suche erstes '[' und letztes ']' und parse diesen Slice
    i, j = txt.find('['), txt.rfind(']')
    if i != -1 and j != -1 and j > i:
        out = _try(txt[i:j+1])
        if out is not None:
            return out
    # 3) Fallback: keine Waypoints
    return []

# ================================
# Demo & CLI
# ================================

def demo(llm_provider: Optional[str] = None, model: Optional[str] = None):
    random.seed(0)
    W, H = 60, 30
    obstacles = [
        (18, 3, 40, 6),   # horizontale Barriere oben
        (25, 10, 26, 24), # vertikale Barriere
        (5, 15, 14, 16),  # kleiner Block links
        (45, 18, 55, 22), # Block rechts
    ]
    g = GridMap(W, H, obstacles)
    start, goal = (2, 2), (57, 27)

    llm: Optional[LLMClient] = None
    if llm_provider:
        llm = build_llm_client(llm_provider, model)

    t0 = time.time()
    path = plan_path(g, start, goal, llm=llm)
    dt = (time.time() - t0) * 1000
    print(f"Pfad-Länge: {len(path) if path else None} | Laufzeit: {dt:.1f} ms | LLM: {llm_provider or 'none'}")
    render_ascii(g, path, start, goal)


def build_llm_client(provider: str, model: Optional[str]) -> LLMClient:
    p = provider.lower().strip()
    if p in ("openai", "oa"):
        return OpenAIClient(model=model)
    if p in ("ollama", "ol"):
        return OllamaClient(model=model)
    if p in ("hf", "huggingface"):
        return HFClient(model=model)
    raise ValueError(f"Unbekannter LLM-Provider: {provider}")


def main():
    ap = argparse.ArgumentParser(description="iLLM-A*-inspiriertes Pfadplanen mit optionalem LLM-Waypointing")
    ap.add_argument("--provider", choices=["openai","ollama","hf"], help="LLM-Provider auswählen")
    ap.add_argument("--model", help="Modellname (optional). Beispiele: gpt-4o-mini, mistral, state-spaces/mamba-2.8b")
    args = ap.parse_args()
    demo(llm_provider=args.provider, model=args.model)

if __name__ == "__main__":
    main()

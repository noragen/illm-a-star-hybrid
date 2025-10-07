
import time, random, argparse, statistics
from illm_astar_llm import GridMap, AStar, plan_path, HeuristicWaypointProposer, HybridWaypointProposer, build_llm_client

def random_map(width, height, density=0.10, blocks=20, seed=None):
    rnd = random.Random(seed)
    obstacles = []
    for _ in range(blocks):
        w = max(1, int(rnd.expovariate(1.5) * (width * density / 4)))
        h = max(1, int(rnd.expovariate(1.5) * (height * density / 4)))
        x0 = rnd.randint(0, max(0, width - w))
        y0 = rnd.randint(0, max(0, height - h))
        x1 = min(width-1, x0 + w)
        y1 = min(height-1, y0 + h)
        obstacles.append((x0,y0,x1,y1))
    return GridMap(width, height, obstacles)

def run_once(grid, start, goal, mode="pure", llm=None):
    t0 = time.time()
    if mode == "pure":
        astar = AStar(grid)
        path = astar.search(start, goal)
        dt = (time.time() - t0) * 1000
        nodes = astar.nodes_expanded
        return dt, nodes, path
    elif mode == "hybrid":
        # Use proposer (llm or heuristic fallback)
        t0 = time.time()
        path = plan_path(grid, start, goal, llm=llm)
        dt = (time.time() - t0) * 1000
        # nodes not directly available for composed path; skip nodes count here
        return dt, None, path
    else:
        raise ValueError("unknown mode")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=5)
    ap.add_argument("--width", type=int, default=80)
    ap.add_argument("--height", type=int, default=40)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--provider", choices=["openai","ollama","hf"], help="optional: benchmark with a real LLM")
    ap.add_argument("--model", help="LLM model name (optional)")
    args = ap.parse_args()

    llm = build_llm_client(args.provider, args.model) if args.provider else None

    pure_times, pure_nodes = [], []
    hybrid_times = []

    for i in range(args.maps):
        grid = random_map(args.width, args.height, density=args.density, blocks=args.blocks, seed=i)
        start = (1,1)
        goal = (args.width-2, args.height-2)

        dt_p, nodes_p, path_p = run_once(grid, start, goal, mode="pure")
        dt_h, nodes_h, path_h = run_once(grid, start, goal, mode="hybrid", llm=llm)

        if path_p is None or path_h is None:
            # regenerate if unsolvable
            continue

        pure_times.append(dt_p)
        pure_nodes.append(nodes_p if nodes_p is not None else 0)
        hybrid_times.append(dt_h)

        print(f"[{i+1}/{args.maps}] pure: {dt_p:.1f} ms, nodes={nodes_p} | hybrid: {dt_h:.1f} ms")

    if not pure_times or not hybrid_times:
        print("Not enough successful runs.")
        return

    print("\n=== Summary ===")
    print(f"pure A*:     mean {statistics.mean(pure_times):.1f} ms  (nodes ~{int(statistics.mean(pure_nodes))})")
    print(f"hybrid mode: mean {statistics.mean(hybrid_times):.1f} ms")

if __name__ == "__main__":
    main()

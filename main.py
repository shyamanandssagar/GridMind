"""Run a GridMind simulation.

Usage:
    python main.py                 # use the LLM agent (needs ANTHROPIC_API_KEY)
    python main.py --offline       # use the deterministic heuristic agent
    python main.py --ticks 12      # run for a given number of ticks

The loop is the same in both modes: observe -> let the agent act -> inject the
occasional fault -> advance the world -> print a one-line summary.
"""

from __future__ import annotations

import argparse

from gridmind import GridEnvironment, build_agent


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title.center(64))
    print("=" * 64)


def print_tick(env: GridEnvironment, record) -> None:
    snap = env.snapshot()
    state = "STABLE" if snap["stable"] else "UNSTABLE"
    served = sum(1 for l in snap["loads"] if l["served"])
    print(
        f"Tick {snap['tick']:>2} | {snap['frequency_hz']:.2f} Hz [{state}] | "
        f"gen {snap['total_generation_mw']:>5.1f} / "
        f"dem {snap['total_demand_mw']:>5.1f} MW | "
        f"loads {served}/{len(snap['loads'])}"
    )
    if record.reasoning:
        print(f"        reasoning: {record.reasoning[:90]}")
    if record.actions:
        print(f"        actions  : {', '.join(record.actions)}")
    for line in env.state.log:
        print(f"        {line}")
    env.state.log.clear()


def run(ticks: int, offline: bool, seed: int) -> None:
    env = GridEnvironment(seed=seed)
    agent = build_agent(offline=offline)

    mode = "Heuristic" if offline or type(agent).__name__ == "HeuristicAgent" else "LLM"
    banner(f"GridMind  -  {mode} agent  -  {ticks} ticks")

    stable_ticks = 0
    for _ in range(ticks):
        record = agent.act(env)          # agent reads state and issues commands
        env.maybe_inject_fault()         # the world throws something at it
        env.step()                       # physics advances
        print_tick(env, record)
        if env.is_stable():
            stable_ticks += 1

    banner("Run summary")
    print(f"Stable for {stable_ticks}/{ticks} ticks "
          f"({100 * stable_ticks / ticks:.0f}% of the time).")
    final = env.snapshot()
    served = sum(1 for l in final["loads"] if l["served"])
    print(f"Final frequency: {final['frequency_hz']:.2f} Hz | "
          f"loads energised: {served}/{len(final['loads'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GridMind agent.")
    parser.add_argument("--ticks", type=int, default=10, help="number of ticks to simulate")
    parser.add_argument("--offline", action="store_true", help="use the heuristic agent")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for the grid")
    args = parser.parse_args()
    run(ticks=args.ticks, offline=args.offline, seed=args.seed)


if __name__ == "__main__":
    main()

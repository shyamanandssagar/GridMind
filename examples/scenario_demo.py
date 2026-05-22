"""A scripted scenario: force a generator trip and watch the agent recover.

Run with:
    python examples/scenario_demo.py

This is handy for a live demo because the fault is deterministic rather than
random, so the agent always has something interesting to respond to.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridmind import GridEnvironment, build_agent


def main() -> None:
    env = GridEnvironment(seed=1)
    agent = build_agent()       # uses LLM if a key is set, else heuristic

    print("Initial state:")
    snap = env.snapshot()
    print(f"  frequency {snap['frequency_hz']} Hz, "
          f"imbalance {snap['imbalance_mw']} MW\n")

    # Tick 0-1: let the agent settle the grid.
    for _ in range(2):
        agent.act(env)
        env.step()

    # Force the cheap hydro plant offline to create a sudden deficit.
    print(">>> Forcing Koshi Hydro (G2) offline\n")
    g2 = env.generator("G2")
    g2.online = False
    g2.output_mw = 0.0

    # Tick 2-5: the agent should ramp other sources / use the battery.
    for _ in range(4):
        record = agent.act(env)
        env.step()
        snap = env.snapshot()
        served = sum(1 for l in snap["loads"] if l["served"])
        print(f"Tick {snap['tick']:>2} | {snap['frequency_hz']:.2f} Hz | "
              f"loads {served}/{len(snap['loads'])} | "
              f"{', '.join(record.actions) or 'no action'}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()

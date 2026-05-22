"""The grid-operator agent.

``LLMAgent`` is the real thing: on every tick it sends the grid state to the
model, lets it call tools in a loop until it decides the grid is fine, and
records what it did. ``HeuristicAgent`` is a deterministic fallback that uses
simple control rules, so the project still runs and can be demonstrated when no
API key is available. Both expose the same ``act(env)`` method.
"""

from __future__ import annotations

import json

from .config import settings
from .environment import GridEnvironment, NOMINAL_FREQUENCY
from .memory import Memory, TickRecord
from .tools import DISPATCH, TOOL_SCHEMA

SYSTEM_PROMPT = """\
You are GridMind, an autonomous control agent for a small electrical grid.
Your standing objectives, in priority order:

1. Stability  - keep frequency inside 49.5-50.5 Hz (nominal 50 Hz).
2. Reliability - keep every load energised; if you must shed load, shed the
   lowest-priority load first (priority 3 before 2 before 1). Never shed a
   priority-1 load unless the grid would otherwise collapse.
3. Economy    - once stable and reliable, minimise cost by favouring cheap
   sources (solar, then hydro, then thermal) and using the battery sensibly.

Frequency rises with surplus generation and falls with a deficit, so you
balance generation against demand. Work in steps: inspect the grid, decide,
act through tools, and call finish_tick when the state is acceptable. Keep your
spoken reasoning to one or two sentences before each action.
"""


def _format_observation(env: GridEnvironment, memory: Memory) -> str:
    """Build the user-turn text describing the current situation."""
    snap = env.snapshot()
    return (
        f"Current grid state:\n{json.dumps(snap, indent=2)}\n\n"
        f"Recent history (frequency trend: {memory.frequency_trend()}):\n"
        f"{memory.recent_summary()}\n\n"
        "Decide what to do this tick."
    )


class LLMAgent:
    """An agent whose reasoning engine is an Anthropic model."""

    def __init__(self) -> None:
        settings.validate()
        # Imported lazily so the heuristic agent works without the SDK installed.
        from anthropic import Anthropic
        self.client = Anthropic(api_key=settings.api_key)
        self.memory = Memory()

    def act(self, env: GridEnvironment) -> TickRecord:
        messages = [{"role": "user", "content": _format_observation(env, self.memory)}]
        actions_taken: list[str] = []
        reasoning_bits: list[str] = []

        for _ in range(settings.max_tool_iterations):
            response = self.client.messages.create(
                model=settings.model,
                max_tokens=1024,
                temperature=settings.temperature,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMA,
                messages=messages,
            )

            # Capture any spoken reasoning the model produced this turn.
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    reasoning_bits.append(block.text.strip())

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if not tool_calls:
                break

            messages.append({"role": "assistant", "content": response.content})

            results = []
            done = False
            for call in tool_calls:
                output = DISPATCH[call.name](env, **call.input)
                actions_taken.append(self._describe(call.name, call.input))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(output),
                })
                if call.name == "finish_tick":
                    done = True

            messages.append({"role": "user", "content": results})
            if done:
                break

        return self._commit(env, actions_taken, " ".join(reasoning_bits))

    @staticmethod
    def _describe(name: str, args: dict) -> str:
        if name == "get_grid_status":
            return "checked status"
        if name == "finish_tick":
            return f"finished ({args.get('reason', '')})"
        return f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})"

    def _commit(self, env: GridEnvironment, actions: list[str], reasoning: str) -> TickRecord:
        snap = env.snapshot()
        record = TickRecord(
            tick=snap["tick"],
            frequency_hz=snap["frequency_hz"],
            imbalance_mw=snap["imbalance_mw"],
            actions=[a for a in actions if a != "checked status"],
            reasoning=reasoning,
        )
        self.memory.record(record)
        return record


class HeuristicAgent:
    """Deterministic baseline controller used when no API key is present."""

    def __init__(self) -> None:
        self.memory = Memory()

    def act(self, env: GridEnvironment) -> TickRecord:
        actions: list[str] = []
        imbalance = env.imbalance()

        # 1. Restore any shed load if we now have comfortable surplus.
        if imbalance > 10:
            for load in sorted(env.state.loads, key=lambda l: l.priority):
                if not load.served and imbalance - load.demand_mw > 2:
                    load.served = True
                    imbalance -= load.demand_mw
                    actions.append(f"restored {load.id}")

        # 2. Correct a deficit: lean on the battery, then ramp cheap generators.
        if imbalance < -1:
            deficit = -imbalance
            bat = env.state.battery
            if bat.charge_mwh > 1:
                discharge = min(deficit, bat.max_rate_mw)
                bat.flow_mw = round(discharge, 1)
                deficit -= discharge
                actions.append(f"battery discharge {round(discharge, 1)} MW")
            for gen in sorted(env.state.generators, key=lambda g: g.cost_per_mwh):
                if deficit <= 0:
                    break
                ramp = min(deficit, gen.headroom_mw)
                if ramp > 0:
                    gen.output_mw = round(gen.output_mw + ramp, 1)
                    deficit -= ramp
                    actions.append(f"ramped {gen.id} +{round(ramp, 1)} MW")
            # Still short? Shed the lowest-priority load.
            if deficit > 1:
                for load in sorted(env.state.loads, key=lambda l: -l.priority):
                    if load.served and load.priority > 1:
                        load.served = False
                        actions.append(f"shed {load.id}")
                        break

        # 3. Trim surplus by backing off the most expensive plant.
        if imbalance > 5:
            for gen in sorted(env.state.generators, key=lambda g: -g.cost_per_mwh):
                if gen.output_mw > 0 and imbalance > 5:
                    cut = min(imbalance - 2, gen.output_mw)
                    gen.output_mw = round(gen.output_mw - cut, 1)
                    imbalance -= cut
                    actions.append(f"backed off {gen.id} -{round(cut, 1)} MW")

        snap = env.snapshot()
        record = TickRecord(
            tick=snap["tick"],
            frequency_hz=snap["frequency_hz"],
            imbalance_mw=snap["imbalance_mw"],
            actions=actions,
            reasoning="rule-based control",
        )
        self.memory.record(record)
        return record


def build_agent(offline: bool = False):
    """Return an LLM agent, falling back to the heuristic one on request or
    when no API key is configured."""
    if offline or not settings.api_key:
        return HeuristicAgent()
    return LLMAgent()

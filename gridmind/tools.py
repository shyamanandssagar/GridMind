"""Tools the agent is allowed to invoke.

Each tool is a thin wrapper around the environment. The ``TOOL_SCHEMA`` list
below is what we hand to the model so it knows the available actions and their
arguments; ``dispatch`` maps a tool name back to the matching Python function.
Keeping the schema and the implementations side by side makes it hard for them
to drift apart.
"""

from __future__ import annotations

from .environment import GridEnvironment

# Schema advertised to the language model (Anthropic tool-use format).
TOOL_SCHEMA = [
    {
        "name": "get_grid_status",
        "description": "Return the full current state of the grid: frequency, "
                       "generators, loads and battery. Call this first.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_generator_output",
        "description": "Set a generator's real-power output in MW. Cannot exceed "
                       "capacity and only works on online generators.",
        "input_schema": {
            "type": "object",
            "properties": {
                "generator_id": {"type": "string"},
                "output_mw": {"type": "number"},
            },
            "required": ["generator_id", "output_mw"],
        },
    },
    {
        "name": "set_battery_flow",
        "description": "Charge or discharge the battery. Positive MW discharges "
                       "into the grid, negative MW charges from it.",
        "input_schema": {
            "type": "object",
            "properties": {"flow_mw": {"type": "number"}},
            "required": ["flow_mw"],
        },
    },
    {
        "name": "shed_load",
        "description": "Disconnect a load to reduce demand. Use only when supply "
                       "cannot meet demand. Prefer low-priority loads.",
        "input_schema": {
            "type": "object",
            "properties": {"load_id": {"type": "string"}},
            "required": ["load_id"],
        },
    },
    {
        "name": "restore_load",
        "description": "Reconnect a previously shed load once supply allows.",
        "input_schema": {
            "type": "object",
            "properties": {"load_id": {"type": "string"}},
            "required": ["load_id"],
        },
    },
    {
        "name": "finish_tick",
        "description": "Signal that no further action is needed this tick and the "
                       "grid is in an acceptable state. Include a one-line reason.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


def get_grid_status(env: GridEnvironment) -> dict:
    return env.snapshot()


def set_generator_output(env: GridEnvironment, generator_id: str, output_mw: float) -> dict:
    gen = env.generator(generator_id)
    if gen is None:
        return {"ok": False, "error": f"No generator {generator_id}"}
    if not gen.online:
        return {"ok": False, "error": f"{gen.name} is offline and cannot be dispatched"}
    clamped = max(0.0, min(output_mw, gen.capacity_mw))
    gen.output_mw = round(clamped, 1)
    return {"ok": True, "generator_id": generator_id, "output_mw": gen.output_mw}


def set_battery_flow(env: GridEnvironment, flow_mw: float) -> dict:
    bat = env.state.battery
    clamped = max(-bat.max_rate_mw, min(flow_mw, bat.max_rate_mw))
    bat.flow_mw = round(clamped, 1)
    return {"ok": True, "flow_mw": bat.flow_mw, "charge_mwh": round(bat.charge_mwh, 1)}


def shed_load(env: GridEnvironment, load_id: str) -> dict:
    load = env.load(load_id)
    if load is None:
        return {"ok": False, "error": f"No load {load_id}"}
    load.served = False
    return {"ok": True, "load_id": load_id, "shed_mw": load.demand_mw}


def restore_load(env: GridEnvironment, load_id: str) -> dict:
    load = env.load(load_id)
    if load is None:
        return {"ok": False, "error": f"No load {load_id}"}
    load.served = True
    return {"ok": True, "load_id": load_id, "restored_mw": load.demand_mw}


def finish_tick(env: GridEnvironment, reason: str) -> dict:
    return {"ok": True, "done": True, "reason": reason}


# Name -> callable used by the agent loop to execute a tool call.
DISPATCH = {
    "get_grid_status": get_grid_status,
    "set_generator_output": set_generator_output,
    "set_battery_flow": set_battery_flow,
    "shed_load": shed_load,
    "restore_load": restore_load,
    "finish_tick": finish_tick,
}

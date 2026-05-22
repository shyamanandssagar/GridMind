# GridMind — An Autonomous Agent for Smart-Grid Energy Management

GridMind is a small **agentic AI** system that operates a simulated electrical
grid on its own. Every tick it looks at the grid, reasons about what is wrong,
and takes corrective action through a set of control tools — ramping
generators, dispatching a battery, or shedding load — to keep the system stable,
reliable, and cheap to run.

It is built around the classic agent loop:

```
        observe  ──►  reason  ──►  act  ──►  observe ...
      (grid state)   (LLM)     (tools)    (new state)
```

The reasoning engine is a large language model that decides *which* tool to call
and *with what arguments*. A deterministic rule-based controller is included as a
fallback so the project runs and can be demonstrated even without an API key.

---

## Why this is "agentic"

A normal script follows fixed instructions. An **agent** is given a goal and the
freedom to choose its own actions to reach it. GridMind has all four ingredients
of an agent:

| Ingredient | In GridMind |
|------------|-------------|
| **Goal** | Keep frequency in 49.5–50.5 Hz, keep loads served, minimise cost |
| **Perception** | Reads grid state each tick (`get_grid_status`) |
| **Tools / actions** | Ramp generators, charge/discharge battery, shed/restore load |
| **Memory** | Remembers recent ticks to reason about trends |

The model is never told *how* to fix the grid — only what "good" looks like. It
works out the steps itself.

---

## Domain background

The simulated grid models a single bus serving Bihar-flavoured assets:

- **Generators** — a thermal plant, a hydro plant and a solar farm, each with a
  capacity, cost per MWh and an online/offline flag.
- **Loads** — a hospital feeder (critical), a residential grid, and a steel
  plant (sheddable), each with a priority.
- **Battery** — charges from surplus and discharges to cover a deficit.
- **Frequency** — rises with a generation surplus and falls with a deficit,
  exactly like a real interconnected system. Outside the safe band the grid is
  flagged unstable.

Faults (a generator tripping offline) are injected at random, so the agent has to
react to disturbances rather than just hold a steady state.

---

## Project layout

```
gridmind/
├── main.py                  # run the simulation from the command line
├── gridmind/
│   ├── environment.py       # the simulated grid (state + physics)
│   ├── tools.py             # the actions the agent may take + their schema
│   ├── agent.py             # LLM agent loop + heuristic fallback agent
│   ├── memory.py            # short rolling memory of recent ticks
│   └── config.py            # settings read from environment variables
├── examples/
│   └── scenario_demo.py     # scripted fault-recovery demo
├── requirements.txt
└── .env.example
```

---

## Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. (optional) add your Anthropic API key for the LLM agent
cp .env.example .env
#   then edit .env and paste your key

# 3a. run with the LLM agent
python main.py --ticks 10

# 3b. or run the deterministic agent, no key needed
python main.py --offline --ticks 10
```

Example output:

```
Tick  4 | 49.54 Hz [STABLE] | gen 174.2 / dem 188.3 MW | loads 3/3
        reasoning: Demand is climbing past supply, so I'll cover the gap
                   from the battery before ramping the costly thermal plant.
        actions  : set_battery_flow(flow_mw=12), set_generator_output(...)
```

Run the fault-recovery demo:

```bash
python examples/scenario_demo.py
```

---

## How a single tick works

1. `main.py` asks the agent to `act(env)`.
2. The agent sends the grid state plus a recap of recent ticks to the model.
3. The model calls tools — e.g. `get_grid_status`, then `set_battery_flow` — and
   the agent executes each one against the environment, feeding the result back.
4. When the model is satisfied it calls `finish_tick`.
5. The environment may inject a fault, then advances one step (demand drifts,
   frequency is recomputed, the battery charge updates).
6. A one-line summary is printed and the loop repeats.

---

## Configuration

All settings have sensible defaults and can be overridden via environment
variables (or a `.env` file):

| Variable | Default | Meaning |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | — | Required for the LLM agent |
| `GRIDMIND_MODEL` | `claude-sonnet-4-6` | Model used for reasoning |
| `GRIDMIND_MAX_ITER` | `8` | Max tool calls allowed per tick |
| `GRIDMIND_TEMPERATURE` | `0.2` | Sampling temperature |

---

## Possible extensions

- Add a renewable-forecast tool so the agent plans ahead instead of reacting.
- Track cumulative generation cost and reward the agent for cheaper dispatch.
- Model multiple buses with line limits and a power-flow solver.
- Log every decision to CSV and plot frequency vs. time.

---

## License

MIT
# GridMind

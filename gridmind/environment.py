"""Discrete-time simulation of a small power grid.

The environment models generators, controllable loads and a battery store
connected to a single bus. Each tick advances demand along a daily profile,
recomputes the supply/demand balance and updates the system frequency. The
agent never touches this class directly; it acts through the tools defined in
``tools.py``, which mutate the shared state here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Nominal grid frequency for an Indian-style 50 Hz system.
NOMINAL_FREQUENCY = 50.0

# Frequency moves this many Hz per MW of imbalance, scaled by system inertia.
FREQUENCY_GAIN = 0.04

# Outside this band the grid is considered unstable and protection may trip.
SAFE_FREQUENCY_BAND = (49.5, 50.5)


@dataclass
class Generator:
    id: str
    name: str
    kind: str            # "thermal", "hydro", "solar"
    capacity_mw: float
    cost_per_mwh: float
    output_mw: float = 0.0
    online: bool = True

    @property
    def headroom_mw(self) -> float:
        """Spare capacity that can still be dispatched."""
        if not self.online:
            return 0.0
        return max(0.0, self.capacity_mw - self.output_mw)


@dataclass
class Load:
    id: str
    name: str
    demand_mw: float
    priority: int        # 1 = critical (hospital), 3 = sheddable (industrial)
    served: bool = True
    base_demand_mw: float = 0.0

    def __post_init__(self) -> None:
        # Remember the nominal demand so daily drift stays bounded around it.
        if self.base_demand_mw == 0.0:
            self.base_demand_mw = self.demand_mw


@dataclass
class Battery:
    capacity_mwh: float
    charge_mwh: float
    max_rate_mw: float
    # Positive flow = discharging into the grid, negative = charging from it.
    flow_mw: float = 0.0


@dataclass
class GridState:
    generators: list[Generator]
    loads: list[Load]
    battery: Battery
    frequency_hz: float = NOMINAL_FREQUENCY
    tick: int = 0
    log: list[str] = field(default_factory=list)


class GridEnvironment:
    """A self-contained grid the agent has to keep stable and economical."""

    def __init__(self, seed: int | None = 7) -> None:
        self.rng = random.Random(seed)
        self.state = self._build_initial_state()

    def _build_initial_state(self) -> GridState:
        generators = [
            Generator("G1", "Barauni Thermal", "thermal", capacity_mw=120, cost_per_mwh=3200, output_mw=80),
            Generator("G2", "Koshi Hydro", "hydro", capacity_mw=90, cost_per_mwh=900, output_mw=60),
            Generator("G3", "Rooftop Solar Farm", "solar", capacity_mw=60, cost_per_mwh=0, output_mw=40),
        ]
        loads = [
            Load("L1", "City Hospital Feeder", demand_mw=35, priority=1),
            Load("L2", "Residential Grid", demand_mw=70, priority=2),
            Load("L3", "Steel Plant", demand_mw=55, priority=3),
        ]
        battery = Battery(capacity_mwh=80, charge_mwh=50, max_rate_mw=30)
        return GridState(generators=generators, loads=loads, battery=battery)

    # -- read helpers ---------------------------------------------------------

    def generator(self, gen_id: str) -> Generator | None:
        return next((g for g in self.state.generators if g.id == gen_id), None)

    def load(self, load_id: str) -> Load | None:
        return next((l for l in self.state.loads if l.id == load_id), None)

    def total_generation(self) -> float:
        gen = sum(g.output_mw for g in self.state.generators if g.online)
        return gen + self.state.battery.flow_mw

    def total_demand(self) -> float:
        return sum(l.demand_mw for l in self.state.loads if l.served)

    def imbalance(self) -> float:
        """Generation minus demand. Positive means surplus."""
        return self.total_generation() - self.total_demand()

    def is_stable(self) -> bool:
        low, high = SAFE_FREQUENCY_BAND
        return low <= self.state.frequency_hz <= high

    # -- dynamics -------------------------------------------------------------

    def _recompute_frequency(self) -> None:
        """Frequency tracks the supply/demand balance with mild damping."""
        target = NOMINAL_FREQUENCY + FREQUENCY_GAIN * self.imbalance()
        # Move part-way towards the target so changes look like real inertia.
        self.state.frequency_hz += 0.6 * (target - self.state.frequency_hz)
        self.state.frequency_hz = round(self.state.frequency_hz, 3)

    def _drift_demand(self) -> None:
        """Nudge each load along a smooth daily curve around its baseline."""
        phase = self.state.tick / 24 * 2 * math.pi
        daily_factor = 1 + 0.12 * math.sin(phase)
        for load in self.state.loads:
            noise = self.rng.uniform(-0.03, 0.03)
            load.demand_mw = round(load.base_demand_mw * (daily_factor + noise), 1)

    def maybe_inject_fault(self, probability: float = 0.12) -> None:
        """Occasionally knock a generator offline to test the agent."""
        if self.rng.random() < probability:
            online = [g for g in self.state.generators if g.online]
            if len(online) > 1:                       # never black out everything
                victim = self.rng.choice(online)
                victim.online = False
                victim.output_mw = 0.0
                self.state.log.append(
                    f"[FAULT] {victim.name} ({victim.id}) tripped offline."
                )

    def step(self) -> None:
        """Advance the simulation by one tick."""
        self._drift_demand()
        self._recompute_frequency()
        self._update_battery_charge()
        self.state.tick += 1

    def _update_battery_charge(self) -> None:
        bat = self.state.battery
        # flow_mw acts over a one-hour tick, so MW maps directly to MWh.
        bat.charge_mwh = max(0.0, min(bat.capacity_mwh, bat.charge_mwh - bat.flow_mw))
        if bat.charge_mwh in (0.0, bat.capacity_mwh):
            bat.flow_mw = 0.0     # nothing left to give or absorb

    # -- snapshot for the agent ----------------------------------------------

    def snapshot(self) -> dict:
        """A compact, JSON-friendly view of everything the agent can see."""
        return {
            "tick": self.state.tick,
            "frequency_hz": self.state.frequency_hz,
            "stable": self.is_stable(),
            "total_generation_mw": round(self.total_generation(), 1),
            "total_demand_mw": round(self.total_demand(), 1),
            "imbalance_mw": round(self.imbalance(), 1),
            "generators": [
                {
                    "id": g.id, "name": g.name, "kind": g.kind,
                    "output_mw": g.output_mw, "capacity_mw": g.capacity_mw,
                    "cost_per_mwh": g.cost_per_mwh, "online": g.online,
                }
                for g in self.state.generators
            ],
            "loads": [
                {
                    "id": l.id, "name": l.name, "demand_mw": l.demand_mw,
                    "priority": l.priority, "served": l.served,
                }
                for l in self.state.loads
            ],
            "battery": {
                "charge_mwh": round(self.state.battery.charge_mwh, 1),
                "capacity_mwh": self.state.battery.capacity_mwh,
                "flow_mw": self.state.battery.flow_mw,
            },
        }

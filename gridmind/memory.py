"""Lightweight working memory for the agent.

The model is stateless between calls, so we keep a short rolling record of what
happened on each tick and feed a summary back in. This is what lets the agent
reason about trends ("frequency has been falling for three ticks") instead of
treating every tick as brand new.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TickRecord:
    tick: int
    frequency_hz: float
    imbalance_mw: float
    actions: list[str]
    reasoning: str


@dataclass
class Memory:
    history: list[TickRecord] = field(default_factory=list)
    window: int = 5      # how many recent ticks to surface to the model

    def record(self, record: TickRecord) -> None:
        self.history.append(record)

    def recent_summary(self) -> str:
        """A plain-text recap of the last few ticks for the prompt."""
        if not self.history:
            return "No prior ticks. This is the first decision."
        lines = []
        for r in self.history[-self.window:]:
            acts = "; ".join(r.actions) if r.actions else "no action"
            lines.append(
                f"Tick {r.tick}: freq={r.frequency_hz} Hz, "
                f"imbalance={r.imbalance_mw} MW -> {acts}"
            )
        return "\n".join(lines)

    def frequency_trend(self) -> str:
        """Quick qualitative read on where frequency is heading."""
        if len(self.history) < 2:
            return "unknown"
        delta = self.history[-1].frequency_hz - self.history[-2].frequency_hz
        if abs(delta) < 0.05:
            return "steady"
        return "rising" if delta > 0 else "falling"

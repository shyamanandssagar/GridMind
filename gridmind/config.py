"""Central configuration.

Values are read from environment variables where it makes sense so the same
code runs unchanged on a laptop or in CI. Drop a ``.env`` file next to the
project (see ``.env.example``) and it gets loaded automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:        # dotenv is optional; env vars still work without it
    pass


@dataclass
class Settings:
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("GRIDMIND_MODEL", "claude-sonnet-4-6")
    max_tool_iterations: int = int(os.getenv("GRIDMIND_MAX_ITER", "8"))
    temperature: float = float(os.getenv("GRIDMIND_TEMPERATURE", "0.2"))

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "add your key, or export it in the shell."
            )


settings = Settings()

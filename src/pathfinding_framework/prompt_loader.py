from __future__ import annotations

from importlib.resources import files


def load_prompt_text(filename: str, fallback: str) -> str:
    """Load prompt text from package prompts directory, fallback if missing."""
    try:
        prompt_path = files("pathfinding_framework").joinpath("prompts").joinpath(filename)
        content = prompt_path.read_text(encoding="utf-8")
    except Exception:
        return fallback
    content = content.strip()
    return content if content else fallback

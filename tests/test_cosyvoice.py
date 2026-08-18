"""Dependency-free tests for CosyVoice prompt preparation."""

from __future__ import annotations

from core.planner import NarrationPlanner
from core.profile import ProfileManager
from core.cosyvoice_backend.prompt_builder import build_prompt


def test_cosyvoice_prompt_uses_narration_plan() -> None:
    profile = ProfileManager().load("gaming_news")
    plan = NarrationPlanner(profile).plan("Imagine HooperTTS. Officially confirmed!")

    prompt = build_prompt(plan, profile)

    assert "Imagine" in prompt.optimized_text
    assert "gaming news narration" in prompt.instruction
    assert "officially" in prompt.instruction
    assert prompt.speed == 1.0


def test_cosyvoice_prompt_maps_short_form_delivery() -> None:
    profile = ProfileManager().load("youtube_shorts")
    plan = NarrationPlanner(profile).plan("Imagine this.")

    prompt = build_prompt(plan, profile)

    assert "delivery speed of about fast" in prompt.instruction
    assert prompt.speed == 1.10


if __name__ == "__main__":
    test_cosyvoice_prompt_uses_narration_plan()
    test_cosyvoice_prompt_maps_short_form_delivery()

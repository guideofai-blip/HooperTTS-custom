"""Map HooperTTS narration plans to CosyVoice2 instructions."""

from __future__ import annotations

from dataclasses import dataclass

from core.planner import SentencePlan
from core.profile import NarrationProfile


@dataclass(frozen=True)
class CosyVoicePrompt:
    """Text and expressiveness controls for CosyVoice2 inference."""

    optimized_text: str
    instruction: str
    speed: float


def build_prompt(
    narration_plan: list[SentencePlan], profile: NarrationProfile
) -> CosyVoicePrompt:
    """Build a natural-language instruction from the narration plan.

    CosyVoice2 accepts an instruction alongside the reference waveform.  The
    values here are intentionally descriptive rather than Qwen prompt fields,
    leaving room for model-specific emotion and delivery controls later.
    """
    optimized_text = "\n\n".join(
        "\n".join(plan.chunks).strip() for plan in narration_plan
    ).strip()
    average_energy = 5.0
    if narration_plan:
        average_energy = sum(plan.estimated_energy for plan in narration_plan) / len(
            narration_plan
        )
    emphasized_words = sorted(
        {word for plan in narration_plan for word in plan.emphasized_words}
    )
    emphasis = (
        f" Emphasize: {', '.join(emphasized_words)}."
        if emphasized_words
        else ""
    )
    instruction = (
        f"Deliver this as {profile.name.replace('_', ' ')} narration. "
        f"Use a {profile.hook_style} opening, {profile.reveal_style} reveals, "
        f"and {profile.question_style} questions. "
        f"Target energy is {average_energy:.1f} out of 10. "
        f"Use a delivery speed of about {speed_description(profile.name)}. "
        "Preserve dramatic pauses and speak clearly with expressive, natural pacing."
        f"{emphasis}"
    )
    return CosyVoicePrompt(
        optimized_text=optimized_text,
        instruction=instruction,
        speed=_speed_for_profile(profile.name),
    )


def _speed_for_profile(profile_name: str) -> float:
    """Return a conservative initial speed mapping for experimental inference."""
    return {
        "gta_shorts": 1.15,
        "youtube_shorts": 1.10,
        "podcast": 0.95,
        "documentary": 0.95,
    }.get(profile_name, 1.0)


def speed_description(profile_name: str) -> str:
    """Express pace in natural language for the current public CosyVoice API."""
    speed = _speed_for_profile(profile_name)
    if speed >= 1.1:
        return "fast"
    if speed <= 0.95:
        return "measured"
    return "natural"

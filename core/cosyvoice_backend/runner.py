"""Experimental CosyVoice2 backend using zero-shot reference voice cloning."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.optimizer import ScriptOptimizer
from core.planner import NarrationPlanner
from core.profile import ProfileManager

from .environment import diagnose, format_diagnostics
from .prompt_builder import CosyVoicePrompt, build_prompt


@dataclass(frozen=True)
class GenerationResult:
    """Result of an attempted experimental CosyVoice generation."""

    success: bool
    output_path: str | None
    diagnostics: str
    prompt: CosyVoicePrompt | None = None


def generate(
    script_path: str | Path,
    reference_audio: str | Path | None,
    profile: str,
    output_path: str | Path,
) -> GenerationResult:
    """Generate a WAV with CosyVoice2 instruction-controlled voice cloning."""
    script = Path(script_path)
    output = Path(output_path)
    if not script.exists():
        return GenerationResult(False, None, f"Script not found: {script}")

    narration_profile = ProfileManager().load(profile)
    optimized_text = ScriptOptimizer().optimize(
        script.read_text(encoding="utf-8"), profile=narration_profile.name
    )
    narration_plan = NarrationPlanner(narration_profile).plan(optimized_text)
    prompt = build_prompt(narration_plan, narration_profile)
    diagnostics = diagnose()
    if not diagnostics.ready:
        return GenerationResult(False, None, format_diagnostics(diagnostics), prompt)

    try:
        if reference_audio is None:
            raise ValueError("Reference audio is required for CosyVoice zero-shot cloning.")
        model = load_model(Path(diagnostics.model_location))
        speech, sample_rate = run_inference(model, prompt, Path(reference_audio))
        save_wav(output, speech, sample_rate)
    except Exception:
        return GenerationResult(
            False,
            None,
            "CosyVoice generation failed:\n" + traceback.format_exc(),
            prompt,
        )
    return GenerationResult(True, str(output), f"Wrote {output}", prompt)


def load_model(model_dir: Path) -> Any:
    """Load an official current CosyVoice model from an already-local directory."""
    from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore[import-not-found]

    return AutoModel(model_dir=str(model_dir))


def run_inference(
    model: Any, prompt: CosyVoicePrompt, reference_audio: Path
) -> tuple[Any, int]:
    """Run CosyVoice2 ``inference_instruct2`` with a reference WAV path."""
    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
    import torch  # type: ignore[import-not-found]

    instruction = prompt.instruction
    if not instruction.endswith("<|endofprompt|>"):
        instruction = f"{instruction}<|endofprompt|>"
    outputs = list(
        model.inference_instruct2(
            prompt.optimized_text,
            instruction,
            str(reference_audio),
            stream=False,
        )
    )
    if not outputs:
        raise RuntimeError("CosyVoice produced no audio output.")
    speech = torch.cat([output["tts_speech"] for output in outputs], dim=-1)
    return speech.squeeze(0).detach().cpu().numpy(), int(model.sample_rate)


def save_wav(output_path: Path, speech: Any, sample_rate: int) -> None:
    """Write the generated CosyVoice waveform to the requested output path."""
    import soundfile as sf  # type: ignore[import-not-found]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, speech, sample_rate)

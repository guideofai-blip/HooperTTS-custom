"""Native Qwen3-TTS generation runner."""

from __future__ import annotations

import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.optimizer import ScriptOptimizer
from core.planner import NarrationPlanner
from core.profile import ProfileManager
from .environment import diagnose, format_diagnostics
from .prompt_builder import QwenPrompt, build_prompt


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


@dataclass(frozen=True)
class GenerationResult:
    """Result of an attempted Qwen generation."""

    success: bool
    output_path: str | None
    diagnostics: str
    prompt: QwenPrompt | None = None


def generate(
    script_path: str | Path,
    reference_audio: str | Path | None,
    profile: str,
    output_path: str | Path,
) -> GenerationResult:
    """Optimize a script, build a Qwen prompt, generate audio, and save WAV."""

    script = Path(script_path)
    output = Path(output_path)

    if not script.exists():
        return GenerationResult(
            False,
            None,
            f"Script not found: {script}",
        )

    narration_profile = ProfileManager().load(profile)

    original_text = script.read_text(encoding="utf-8")

    optimized_text = ScriptOptimizer().optimize(
        original_text,
        profile=narration_profile.name,
    )

    narration_plan = NarrationPlanner(narration_profile).plan(
        optimized_text
    )

    prompt = build_prompt(
        narration_plan,
        narration_profile,
    )

    diagnostics = diagnose()

    if not diagnostics.ready:
        return GenerationResult(
            success=False,
            output_path=None,
            diagnostics=format_diagnostics(diagnostics),
            prompt=prompt,
        )

    try:
        model = load_model(diagnostics.model_location)

        wavs, sample_rate = run_inference(
            model=model,
            prompt=prompt,
            reference_audio=(
                Path(reference_audio)
                if reference_audio
                else None
            ),
        )

        raw_output = output.with_name(
            output.stem + "_raw.wav"
        )

        save_wav(
            raw_output,
            wavs[0],
            sample_rate,
        )

        # GTA Shorts: faster delivery + tighter dynamics.
        # Qwen Base does not expose a direct speed/energy control,
        # so this is applied after generation.
        if profile == "gta_shorts":
            process_audio(
                input_wav=raw_output,
                output_wav=output,
                speed=1.15,
                energy=1.20,
            )

            try:
                raw_output.unlink()
            except FileNotFoundError:
                pass
        else:
            raw_output.replace(output)

    except Exception:
        return GenerationResult(
            success=False,
            output_path=None,
            diagnostics=(
                "Qwen generation failed:\n"
                f"{traceback.format_exc()}"
            ),
            prompt=prompt,
        )

    return GenerationResult(
        success=True,
        output_path=str(output),
        diagnostics=f"Wrote {output}",
        prompt=prompt,
    )


def process_audio(
    input_wav: Path,
    output_wav: Path,
    speed: float = 1.0,
    energy: float = 1.0,
) -> None:
    """
    Post-process generated audio.

    speed:
        1.00 = normal
        1.10 = 10% faster
        1.15 = 15% faster
        1.20 = 20% faster

    energy:
        1.00 = unchanged
        1.10 = slightly more energetic
        1.20 = noticeably more energetic
        1.30 = strong processing
    """

    speed = max(0.5, min(float(speed), 2.0))
    energy = max(0.8, min(float(energy), 1.5))

    gain_db = (energy - 1.0) * 8.0

    filter_chain = (
        # Remove a very short leading vocal artifact that can sometimes
        # appear with Qwen ICL generation ("aaa", "dee", etc.).
        # Keep this deliberately small so real speech is not clipped.
        "atrim=start=0.18,"
        "asetpts=PTS-STARTPTS,"
        f"atempo={speed},"
        "acompressor="
        "threshold=-18dB:"
        "ratio=3:"
        "attack=5:"
        "release=80:"
        "makeup=2,"
        f"volume={gain_db}dB"
    )

    output_wav.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-af",
        filter_chain,
        "-ar",
        "24000",
        "-ac",
        "1",
        str(output_wav),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg audio processing failed:\n"
            f"{result.stderr}"
        )


def load_model(
    model_location: str | None,
) -> Any:
    """Load a Qwen3-TTS model with the official qwen_tts wrapper."""

    checkpoint = resolve_model_checkpoint(
        model_location
    )

    import torch  # type: ignore[import-not-found]
    from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

    register_qwen_tts_model()

    device_map = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    dtype = (
        torch.bfloat16
        if device_map != "cpu"
        else torch.float32
    )

    load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "dtype": dtype,
    }

    if device_map != "cpu":
        load_kwargs[
            "attn_implementation"
        ] = "flash_attention_2"

    try:
        return Qwen3TTSModel.from_pretrained(
            checkpoint,
            **load_kwargs,
        )

    except Exception as exc:
        if (
            load_kwargs.get("attn_implementation")
            != "flash_attention_2"
        ):
            raise

        load_kwargs.pop(
            "attn_implementation",
            None,
        )

        try:
            return Qwen3TTSModel.from_pretrained(
                checkpoint,
                **load_kwargs,
            )
        except Exception:
            raise exc


def register_qwen_tts_model() -> None:
    """Register Qwen3-TTS classes with Transformers when available."""

    try:
        from qwen_tts.core.models import (  # type: ignore[import-not-found]
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
            Qwen3TTSProcessor,
        )
        from transformers import (  # type: ignore[import-not-found]
            AutoConfig,
            AutoModel,
            AutoProcessor,
        )
    except ImportError:
        return

    register_calls = (
        lambda: AutoConfig.register(
            "qwen3_tts",
            Qwen3TTSConfig,
        ),
        lambda: AutoModel.register(
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
        ),
        lambda: AutoProcessor.register(
            Qwen3TTSConfig,
            Qwen3TTSProcessor,
        ),
    )

    for register_call in register_calls:
        try:
            register_call()
        except ValueError as exc:
            if "already" not in str(exc).lower():
                raise


def resolve_model_checkpoint(
    model_location: str | None,
) -> str:
    """Return an official model id or concrete local snapshot path."""

    if not model_location:
        return DEFAULT_QWEN_MODEL_ID

    candidate = Path(model_location)

    if not candidate.exists():
        return model_location

    if (candidate / "config.json").exists():
        return str(candidate)

    snapshots_dir = candidate / "snapshots"

    if snapshots_dir.exists():
        snapshots = [
            path
            for path in snapshots_dir.iterdir()
            if (
                path.is_dir()
                and (path / "config.json").exists()
            )
        ]

        if snapshots:
            latest_snapshot = max(
                snapshots,
                key=lambda path: path.stat().st_mtime,
            )
            return str(latest_snapshot)

    return str(candidate)


def transcribe_reference_audio(
    reference_audio: Path,
) -> str:
    """
    Automatically transcribe the reference clip.

    This lets us use Qwen Base voice cloning with
    x_vector_only_mode=False without requiring the user
    to manually type the reference transcript.
    """

    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required to automatically "
            "transcribe the reference audio."
        ) from exc

    # Small model is sufficient for a short reference clip
    # and keeps the transcription step practical in Colab.
    whisper_model = WhisperModel(
        "small",
        device="cuda",
        compute_type="float16",
    )

    segments, _ = whisper_model.transcribe(
        str(reference_audio),
        beam_size=5,
        vad_filter=True,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()

    if not text:
        raise RuntimeError(
            "Could not transcribe the reference audio. "
            "Please use a clear 5–15 second reference clip."
        )

    return text


def run_inference(
    model: Any,
    prompt: QwenPrompt,
    reference_audio: Path | None,
) -> tuple[Any, int]:
    """
    Run Qwen3-TTS Base voice cloning with ICL conditioning.

    The reference transcript is generated automatically with
    faster-whisper, then passed to Qwen with
    x_vector_only_mode=False so the reference speech provides
    more conditioning than speaker identity alone.
    """

    if reference_audio is None:
        raise ValueError(
            "Reference audio is required for voice cloning."
        )

    ref_audio_single = load_reference_audio(reference_audio)

    # Automatically transcribe the exact reference recording.
    # This avoids requiring the user to manually enter the
    # reference transcript.
    ref_text_single = transcribe_reference_audio(
        reference_audio
    )

    syn_text_single = prompt.optimized_text
    syn_lang_single = "Auto"

    common_gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.95,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.95,
    )

    # ICL mode: use the reference speech + transcript in
    # addition to the speaker embedding.
    xvec_only = False

    return model.generate_voice_clone(
        text=syn_text_single,
        language=syn_lang_single,
        ref_audio=ref_audio_single,
        ref_text=ref_text_single,
        x_vector_only_mode=xvec_only,
        **common_gen_kwargs,
    )

def load_reference_audio(
    reference_audio: Path,
) -> str:
    """Return a reference audio path for Qwen voice cloning."""

    if not reference_audio.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {reference_audio}"
        )

    return str(reference_audio)


def save_wav(
    output_path: Path,
    wav: Any,
    sample_rate: int,
) -> None:
    """Write generated audio to a WAV file."""

    import soundfile as sf  # type: ignore[import-not-found]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        output_path,
        wav,
        sample_rate,
    )

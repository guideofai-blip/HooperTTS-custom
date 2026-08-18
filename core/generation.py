"""Backend selection for optional HooperTTS speech generators.

Backends are imported lazily so optional model packages are never required for
the default Qwen workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal


BackendName = Literal["qwen", "cosyvoice"]
BACKEND_CHOICES: tuple[BackendName, ...] = ("qwen", "cosyvoice")


def get_backend_generator(backend: str) -> Callable[..., Any]:
    """Return the selected backend's generator without eagerly importing it."""
    if backend == "qwen":
        from qwen.runner import generate as qwen_generate

        return qwen_generate
    if backend == "cosyvoice":
        from core.cosyvoice_backend.runner import generate as cosyvoice_generate

        return cosyvoice_generate
    choices = ", ".join(BACKEND_CHOICES)
    raise ValueError(f"Unsupported TTS backend {backend!r}. Choose one of: {choices}.")


def generate(
    *,
    backend: BackendName = "qwen",
    script_path: str | Path,
    reference_audio: str | Path | None,
    profile: str,
    output_path: str | Path,
) -> Any:
    """Generate speech with the selected backend.

    ``qwen`` remains the default and is delegated to without altering the
    existing Qwen runner or its request shape.
    """
    generator = get_backend_generator(backend)
    return generator(
        script_path=script_path,
        reference_audio=reference_audio,
        profile=profile,
        output_path=output_path,
    )

"""Environment diagnostics for the optional CosyVoice2 backend."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_DIR = Path("pretrained_models") / "CosyVoice2-0.5B"


@dataclass(frozen=True)
class EnvironmentDiagnostics:
    """Readable CosyVoice generation environment diagnostics."""

    cuda_available: bool
    torch_available: bool
    cosyvoice_available: bool
    soundfile_available: bool
    model_location: str | None
    model_available: bool
    messages: list[str]

    @property
    def ready(self) -> bool:
        """Return whether the local environment can attempt generation."""
        return (
            self.cuda_available
            and self.torch_available
            and self.cosyvoice_available
            and self.soundfile_available
            and self.model_available
        )


def diagnose(model_location: str | Path | None = None) -> EnvironmentDiagnostics:
    """Return diagnostics without downloading a model or importing CosyVoice."""
    messages: list[str] = []
    torch_available = importlib.util.find_spec("torch") is not None
    cosyvoice_available = importlib.util.find_spec("cosyvoice") is not None
    soundfile_available = importlib.util.find_spec("soundfile") is not None
    cuda_available = False

    if torch_available:
        try:
            import torch  # type: ignore[import-not-found]

            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            messages.append(f"Torch is installed but CUDA check failed: {exc}")

    candidate = Path(model_location) if model_location else DEFAULT_MODEL_DIR
    model_available = candidate.is_dir()

    if not torch_available:
        messages.append("Missing torch. Install a CUDA-enabled PyTorch build first.")
    if not cuda_available:
        messages.append("CUDA is not available. CosyVoice2 experimental inference needs a GPU.")
    if not cosyvoice_available:
        messages.append("Missing CosyVoice source/package.")
    if not soundfile_available:
        messages.append("Missing soundfile. It is required to write WAV output.")
    if not model_available:
        messages.append(
            f"CosyVoice2 model directory was not found: {candidate}. "
            "No model was downloaded."
        )
    if not messages:
        messages.append("CosyVoice2 experimental environment looks ready.")

    return EnvironmentDiagnostics(
        cuda_available=cuda_available,
        torch_available=torch_available,
        cosyvoice_available=cosyvoice_available,
        soundfile_available=soundfile_available,
        model_location=str(candidate) if model_available else None,
        model_available=model_available,
        messages=messages,
    )


def format_diagnostics(diagnostics: EnvironmentDiagnostics) -> str:
    """Return diagnostics as human-readable text."""
    lines = [
        "CosyVoice2 Experimental Environment",
        "====================================",
        f"Torch installed: {diagnostics.torch_available}",
        f"CUDA available: {diagnostics.cuda_available}",
        f"CosyVoice installed: {diagnostics.cosyvoice_available}",
        f"soundfile installed: {diagnostics.soundfile_available}",
        f"Model available: {diagnostics.model_available}",
        f"Model location: {diagnostics.model_location or 'not found'}",
        "",
        "Diagnostics:",
    ]
    lines.extend(f"- {message}" for message in diagnostics.messages)
    return "\n".join(lines)

"""Tests for backend selection without optional model dependencies."""

from __future__ import annotations

from pathlib import Path

import core.generation as generation


def test_generate_uses_qwen_by_default() -> None:
    calls: list[dict[str, object]] = []

    def fake_generator(**kwargs: object) -> str:
        calls.append(kwargs)
        return "generated"

    original = generation.get_backend_generator
    try:
        generation.get_backend_generator = lambda backend: fake_generator  # type: ignore[assignment]
        result = generation.generate(
            script_path=Path("script.txt"),
            reference_audio=Path("voice.wav"),
            profile="default",
            output_path=Path("output.wav"),
        )
    finally:
        generation.get_backend_generator = original

    assert result == "generated"
    assert calls == [
        {
            "script_path": Path("script.txt"),
            "reference_audio": Path("voice.wav"),
            "profile": "default",
            "output_path": Path("output.wav"),
        }
    ]


def test_generate_selects_cosyvoice() -> None:
    selected: list[str] = []

    def fake_selector(backend: str):
        selected.append(backend)
        return lambda **_kwargs: "cosyvoice"

    original = generation.get_backend_generator
    try:
        generation.get_backend_generator = fake_selector  # type: ignore[assignment]
        result = generation.generate(
            backend="cosyvoice",
            script_path=Path("script.txt"),
            reference_audio=Path("voice.wav"),
            profile="default",
            output_path=Path("output.wav"),
        )
    finally:
        generation.get_backend_generator = original

    assert result == "cosyvoice"
    assert selected == ["cosyvoice"]


def test_unknown_backend_is_rejected() -> None:
    try:
        generation.get_backend_generator("unknown")
    except ValueError as exc:
        assert "Unsupported TTS backend" in str(exc)
    else:
        raise AssertionError("Expected an unsupported backend to fail")


if __name__ == "__main__":
    test_generate_uses_qwen_by_default()
    test_generate_selects_cosyvoice()
    test_unknown_backend_is_rejected()

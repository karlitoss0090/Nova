import asyncio
import io
import logging
import os
from pathlib import Path
from typing import Literal

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-speech via Kokoro TTS (local, offline, high-quality neural voices).

    Resolution order for synthesize():
      1. Kokoro local  — kokoro-onnx installed AND model files present
      2. Remote server — NOVA_SERVER_URL/speak fallback (e.g. home server running Piper)

    Download model files:
      python scripts/download_voices.py

    English voices:  af_heart (default), af_bella, am_adam, am_michael, ...
    French voices:   ff_siwis (default)
    """

    def __init__(self):
        self._kokoro = None
        self._voice_map = {
            "fr": settings.KOKORO_VOICE_FR,
            "en": settings.KOKORO_VOICE_EN,
        }
        self._lang_map = {
            "fr": "fr-fr",
            "en": "en-us",
        }

    # ── Capability checks ─────────────────────────────────────────────────

    def _kokoro_installed(self) -> bool:
        try:
            import kokoro_onnx  # noqa: F401
            return True
        except ImportError:
            return False

    def _models_present(self) -> bool:
        return (
            Path(settings.KOKORO_MODEL).exists()
            and Path(settings.KOKORO_VOICES).exists()
        )

    def _kokoro_available(self) -> bool:
        return self._kokoro_installed() and self._models_present()

    def is_available(self) -> bool:
        """True if TTS can be served — Kokoro locally or via remote fallback."""
        return self._kokoro_available() or bool(settings.NOVA_SERVER_URL)

    def mode(self) -> Literal["local", "remote", "unavailable"]:
        if self._kokoro_available():
            return "local"
        if settings.NOVA_SERVER_URL:
            return "remote"
        return "unavailable"

    def voices_available(self) -> dict:
        present = self._models_present()
        return {lang: present for lang in self._voice_map}

    # ── Lazy model loader ─────────────────────────────────────────────────

    def _get_kokoro(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(settings.KOKORO_MODEL, settings.KOKORO_VOICES)
        return self._kokoro

    # ── Synthesis ─────────────────────────────────────────────────────────

    async def synthesize(self, text: str, language: str = "en") -> bytes:
        """
        Synthesize text to WAV bytes.
        Tries Kokoro locally first, falls back to remote if Kokoro fails or is unavailable.
        """
        if self._kokoro_available():
            try:
                return await self._synthesize_kokoro(text, language)
            except Exception as exc:
                if not settings.NOVA_SERVER_URL:
                    raise
                logger.warning("Kokoro synthesis failed, falling back to remote: %s", exc)

        if settings.NOVA_SERVER_URL:
            return await self._synthesize_remote(text, language)

        raise RuntimeError(
            "TTS unavailable: kokoro-onnx not installed or model files missing, "
            "and NOVA_SERVER_URL is not set. "
            "Run: python scripts/download_voices.py"
        )

    def _synthesize_kokoro_sync(self, text: str, language: str) -> bytes:
        import soundfile as sf

        kokoro = self._get_kokoro()
        voice = self._voice_map.get(language, self._voice_map["en"])
        lang_code = self._lang_map.get(language, "en-us")

        samples, sample_rate = kokoro.create(
            text,
            voice=voice,
            speed=settings.KOKORO_SPEED,
            lang=lang_code,
        )

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    async def _synthesize_kokoro(self, text: str, language: str) -> bytes:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._synthesize_kokoro_sync(text, language),
        )

    async def _synthesize_remote(self, text: str, language: str) -> bytes:
        url = f"{settings.NOVA_SERVER_URL.rstrip('/')}/speak"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={"content": text})
            resp.raise_for_status()
            return resp.content


tts_engine = TTSEngine()

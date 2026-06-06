import asyncio
import os
import tempfile
from typing import Dict

from core.config import settings


class STTEngine:
    """
    Speech-to-text via faster-whisper (CTranslate2 backend, ~4x faster than openai-whisper).

    Model sizes: tiny / base / small / medium / large-v2 / large-v3
    Set WHISPER_MODEL in .env. 'base' is a good default.

    The model loads lazily on first call and stays in memory.
    """

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                settings.WHISPER_MODEL,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE,
            )
        return self._model

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _transcribe_sync(self, audio_path: str) -> Dict:
        model = self._get_model()
        # segments is a generator — must be consumed before returning
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "segments": [],
        }

    async def transcribe(self, audio_bytes: bytes, audio_format: str = "wav") -> Dict:
        """
        Transcribe audio bytes to text.
        Returns: { text, language, segments }
        """
        if not self.is_available():
            raise RuntimeError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

        ext = audio_format.lstrip(".")
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._transcribe_sync(tmp_path),
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


stt_engine = STTEngine()

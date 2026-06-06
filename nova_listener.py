#!/usr/bin/env python3
"""
nova_listener.py — Phase 3: hands-free wake-word voice interface for NOVA.

Architecture:
  Wake Word → Record → Transcribe → Chat API → Speak API → Playback → (repeat)

Run:
  python nova_listener.py

Dependencies (in addition to requirements.txt):
  pip install openwakeword sounddevice

Environment (.env):
  NOVA_API_URL=http://localhost:8000
  WAKE_WORD_SENSITIVITY=0.5
  WAKE_WORD_MODEL=                   # path to custom ONNX (empty = built-in placeholder)
  LISTENER_SILENCE_THRESHOLD=0.03    # RMS fraction of full-scale for VAD
  LISTENER_SILENCE_DURATION=1.5      # seconds of silence before stopping recording
  LISTENER_MAX_DURATION=30           # max recording seconds
  NOVA_API_TIMEOUT=30                # HTTP timeout in seconds

Phase 4 (not implemented — hooks are present):
  Hey NOVA → Conversation mode → Multiple exchanges → Exit after inactivity
"""
from __future__ import annotations

import io
import logging
import os
import queue
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nova.listener")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ListenerConfig:
    nova_api_url: str = "http://localhost:8000"
    wake_word_sensitivity: float = 0.5
    # Path to a custom ONNX model file (e.g. /path/to/hey_nova.onnx).
    # Leave empty to use the built-in placeholder model.
    wake_word_model: Optional[str] = None
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    silence_threshold_rms: float = 0.005  # 0.5 % of int16 full-scale ≈ ~164 units
    silence_duration_sec: float = 1.5
    max_recording_sec: float = 30.0
    api_timeout_sec: float = 30.0
    beep_freq_hz: float = 440.0
    beep_duration_ms: int = 200
    cooldown_sec: float = 1.0             # pause after playback before re-enabling mic


def load_config() -> ListenerConfig:
    """Load settings from .env (if present) and environment variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv optional — fall through to os.getenv

    def _float(key: str, default: float) -> float:
        raw = os.getenv(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid value for %s='%s', using default %.3f", key, raw, default)
            return default

    return ListenerConfig(
        nova_api_url=os.getenv("NOVA_API_URL", "http://localhost:8000").rstrip("/"),
        wake_word_sensitivity=_float("WAKE_WORD_SENSITIVITY", 0.5),
        wake_word_model=os.getenv("WAKE_WORD_MODEL") or None,
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        silence_threshold_rms=_float("LISTENER_SILENCE_THRESHOLD", 0.03),
        silence_duration_sec=_float("LISTENER_SILENCE_DURATION", 1.5),
        max_recording_sec=_float("LISTENER_MAX_DURATION", 30.0),
        api_timeout_sec=_float("NOVA_API_TIMEOUT", 30.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# WakeWordDetector
# ─────────────────────────────────────────────────────────────────────────────

class WakeWordDetector:
    """
    Detects the NOVA wake word using openWakeWord.

    Built-in placeholder
    ────────────────────
    The built-in 'hey_jarvis' model is used until a custom 'Hey NOVA' model is
    trained and dropped in via WAKE_WORD_MODEL.  It will respond to phrases
    semantically similar to "hey jarvis" and can be used for development.

    Switching to a custom model
    ───────────────────────────
    1. Train an ONNX model with openWakeWord's training pipeline:
       https://github.com/dscripka/openWakeWord#custom-models
    2. Set WAKE_WORD_MODEL=/absolute/path/to/hey_nova.onnx in .env
    3. Restart nova_listener.py — no code changes needed.

    The model key in prediction results is the filename stem (e.g. "hey_nova"
    for hey_nova.onnx), resolved automatically at load time.
    """

    SAMPLE_RATE: int = 16_000
    # 80 ms chunks — optimum for openWakeWord's internal feature extraction.
    CHUNK_SAMPLES: int = 1280
    # Built-in placeholder; replace with a trained "hey_nova" model via env var.
    _BUILTIN_PLACEHOLDER: str = "hey_jarvis"

    def __init__(self, sensitivity: float, model_path: Optional[str] = None) -> None:
        self.sensitivity = sensitivity
        self.model_path = model_path
        self._model = None
        self._model_key: str = self._BUILTIN_PLACEHOLDER

    def load(self) -> None:
        try:
            from openwakeword.model import Model  # type: ignore[import]
            import openwakeword.utils as oww_utils  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "openwakeword not installed.  Run: pip install openwakeword"
            ) from exc

        # tflite_runtime is not available on Apple Silicon / Python 3.13.
        # Always force the ONNX backend — onnxruntime works on all platforms.

        if self.model_path:
            p = Path(self.model_path)
            if not p.exists():
                logger.warning(
                    "WAKE_WORD_MODEL path not found (%s) — falling back to built-in placeholder",
                    self.model_path,
                )
            else:
                self._model = Model(
                    wakeword_models=[str(p)],
                    inference_framework="onnx",
                )
                self._model_key = p.stem
                logger.info(
                    "Wake word: custom model '%s' loaded (sensitivity=%.2f)",
                    self._model_key, self.sensitivity,
                )
                self._model.reset()
                return

        # Built-in placeholder — download ONNX files if not already cached.
        print(f"  Downloading built-in wake-word model '{self._BUILTIN_PLACEHOLDER}' (first run only)...",
              end="", flush=True)
        try:
            oww_utils.download_models([self._BUILTIN_PLACEHOLDER])
            print("  ✓")
        except Exception as exc:
            print(f"  (warning: {exc})")
            logger.warning("download_models failed (model may already be cached): %s", exc)

        self._model = Model(
            wakeword_models=[self._BUILTIN_PLACEHOLDER],
            inference_framework="onnx",
        )
        self._model_key = self._BUILTIN_PLACEHOLDER
        logger.info(
            "Wake word: built-in placeholder '%s' via ONNX (sensitivity=%.2f). "
            "Set WAKE_WORD_MODEL=/path/to/hey_nova.onnx to use a custom model.",
            self._model_key, self.sensitivity,
        )
        self._model.reset()

    def process_chunk(self, audio: np.ndarray) -> bool:
        """
        Feed one ~80 ms audio chunk (int16, 16 kHz) to the model.
        Returns True when the wake-word score exceeds the sensitivity threshold.
        """
        if self._model is None:
            return False
        try:
            prediction: dict = self._model.predict(audio)
            score: float = float(prediction.get(self._model_key, 0.0))
            return score >= self.sensitivity
        except Exception as exc:
            logger.debug("Wake word prediction error: %s", exc)
            return False

    def reset(self) -> None:
        """Reset internal state after a detection to prevent repeated triggers."""
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# AudioRecorder
# ─────────────────────────────────────────────────────────────────────────────

class AudioRecorder:
    """
    Records microphone audio with energy-based voice-activity detection.

    Stops automatically when:
      • silence_duration_sec of consecutive low-energy audio is detected, OR
      • max_duration_sec is reached.

    Returns a 1-D int16 numpy array at 16 kHz, or None on hardware failure.
    """

    SAMPLE_RATE: int = 16_000
    CHANNELS: int = 1
    DTYPE: str = "int16"
    CHUNK_SAMPLES: int = 1_600  # 100 ms per VAD evaluation window

    def __init__(
        self,
        silence_threshold_rms: float = 0.005,
        silence_duration_sec: float = 1.5,
        max_duration_sec: float = 30.0,
        min_duration_sec: float = 2.0,
    ) -> None:
        self.silence_threshold = silence_threshold_rms
        self.silence_duration = silence_duration_sec
        self.max_duration = max_duration_sec
        self.min_duration = min_duration_sec

    def record(self) -> Optional[np.ndarray]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice not installed.  Run: pip install sounddevice"
            ) from exc

        # Print the device that will actually be used so the user can verify.
        try:
            dev_info = sd.query_devices(kind="input")
            dev_name = dev_info.get("name", "unknown") if isinstance(dev_info, dict) else str(dev_info)
            print(f"  [recorder] device : {dev_name}")
        except Exception:
            print("  [recorder] device : (could not query)")
        print(f"  [recorder] threshold: {self.silence_threshold:.4f} rms  |  "
              f"silence: {self.silence_duration}s  |  min: {self.min_duration}s")

        audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=600)

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.debug("Recorder stream status: %s", status)
            try:
                audio_q.put_nowait(indata.copy())
            except queue.Full:
                pass  # drop frame rather than block the audio thread

        frames: list[np.ndarray] = []
        silence_chunks = 0
        chunk_dur = self.CHUNK_SAMPLES / self.SAMPLE_RATE          # seconds per chunk
        silence_limit = max(1, round(self.silence_duration / chunk_dur))
        min_chunks   = round(self.min_duration   / chunk_dur)
        max_chunks   = round(self.max_duration   / chunk_dur)

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                blocksize=self.CHUNK_SAMPLES,
                callback=_callback,
            ):
                for chunk_idx in range(max_chunks):
                    try:
                        chunk = audio_q.get(timeout=2.0)
                    except queue.Empty:
                        logger.warning("Recorder: no audio received — microphone may be blocked")
                        break

                    frames.append(chunk)
                    rms = self._rms(chunk)
                    elapsed = (chunk_idx + 1) * chunk_dur
                    status_char = "█" if rms >= self.silence_threshold else "·"
                    print(f"  [recorder] {elapsed:5.1f}s  rms={rms:.4f}  {status_char}",
                          flush=True)

                    # Silence detection is suppressed for the first min_duration seconds
                    # so a brief pause right after the beep doesn't cut the recording short.
                    if chunk_idx >= min_chunks:
                        if rms < self.silence_threshold:
                            silence_chunks += 1
                            if silence_chunks >= silence_limit:
                                print(f"  [recorder] silence detected — stopping")
                                break
                        else:
                            silence_chunks = 0

        except OSError as exc:
            logger.error("Microphone error: %s", exc)
            return None
        except Exception as exc:
            logger.error("Recording failed: %s", exc)
            return None

        if not frames:
            logger.warning("Recorder: no frames captured")
            return None

        audio = np.concatenate(frames, axis=0).flatten()
        duration = len(audio) / self.SAMPLE_RATE
        max_rms = max(self._rms(f) for f in frames)
        print(f"  [recorder] captured {duration:.2f}s  max_rms={max_rms:.4f}")

        # Save a debug copy so the raw capture can be inspected with any audio player.
        debug_path = Path("/tmp/nova_debug.wav")
        try:
            import soundfile as sf
            sf.write(str(debug_path), audio, self.SAMPLE_RATE, subtype="PCM_16")
            print(f"  [recorder] saved → {debug_path}")
        except Exception as exc:
            print(f"  [recorder] debug save failed: {exc}")
        return audio

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        """Normalised RMS energy in [0, 1] relative to int16 full-scale."""
        samples = chunk.astype(np.float64).flatten()
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2)) / 32_768.0)


# ─────────────────────────────────────────────────────────────────────────────
# SpeechToText
# ─────────────────────────────────────────────────────────────────────────────

class SpeechToText:
    """
    Transcribes audio using faster-whisper (CTranslate2 backend).

    Follows the same model-loading strategy as core/stt.py:
      • Model loaded once at startup via load().
      • Reused for every transcription — no per-request reload overhead.
      • Compatible with the same WHISPER_* env vars used by the FastAPI server.
    """

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper not installed.  Run: pip install faster-whisper"
            ) from exc
        logger.info(
            "STT: loading faster-whisper '%s' on %s/%s",
            self.model_name, self.device, self.compute_type,
        )
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("STT: ready")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> Optional[str]:
        """
        Transcribe a 1-D int16 audio array. Returns clean text or None.

        Uses a temporary WAV file — same pattern as core/stt.py — because
        faster-whisper's transcribe() accepts a file path, not raw arrays.
        """
        if self._model is None:
            logger.error("STT model not loaded — call load() first")
            return None

        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "soundfile not installed.  Run: pip install soundfile"
            ) from exc

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            sf.write(str(tmp_path), audio, sample_rate, subtype="PCM_16")

            # The segments return value is a generator — must be consumed before
            # returning (same invariant documented in core/stt.py).
            segments, info = self._model.transcribe(str(tmp_path), beam_size=5)
            text = "".join(seg.text for seg in segments).strip()
            logger.debug("Transcribed [%s]: %s", info.language, text)
            return text if text else None

        except Exception as exc:
            logger.error("Transcription error: %s", exc)
            return None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# NovaClient
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatResult:
    response: str
    language: str
    session_id: str


class NovaClient:
    """
    HTTP client for the NOVA FastAPI server.

    Endpoints:
      GET  /health  — liveness check
      POST /chat    — blocking LLM request → ChatResult
      POST /speak   — TTS synthesis → raw WAV bytes
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ping(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, message: str, session_id: Optional[str] = None) -> Optional[ChatResult]:
        """
        Send a message to NOVA and return the response.

        session_id is passed through unchanged so conversation history is preserved
        across multiple calls — the Phase 4 foundation for continuous conversation.
        """
        try:
            import httpx
            resp = httpx.post(
                f"{self.base_url}/chat",
                json={"message": message, "session_id": session_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return ChatResult(
                response=data["response"],
                language=data.get("language", "en"),
                session_id=data["session_id"],
            )
        except ImportError as exc:
            raise RuntimeError("httpx not installed.  Run: pip install httpx") from exc
        except __import__("httpx").TimeoutException:
            logger.error("Chat API timed out after %.0f s", self.timeout)
            return None
        except __import__("httpx").ConnectError:
            logger.error("Cannot connect to NOVA at %s — is the server running?", self.base_url)
            return None
        except Exception as exc:
            logger.error("Chat API error: %s", exc)
            return None

    def speak(self, text: str, language: str = "en") -> Optional[bytes]:
        """Request TTS synthesis. Returns raw WAV bytes or None on failure."""
        try:
            import httpx
            resp = httpx.post(
                f"{self.base_url}/speak",
                json={"text": text, "language": language},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            if not resp.content:
                logger.warning("TTS endpoint returned empty audio")
                return None
            return resp.content
        except ImportError as exc:
            raise RuntimeError("httpx not installed.  Run: pip install httpx") from exc
        except __import__("httpx").TimeoutException:
            logger.error("TTS API timed out after %.0f s", self.timeout)
            return None
        except __import__("httpx").ConnectError:
            logger.error("Cannot connect to NOVA TTS at %s", self.base_url)
            return None
        except Exception as exc:
            logger.error("TTS API error: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# AudioPlayer
# ─────────────────────────────────────────────────────────────────────────────

class AudioPlayer:
    """
    Plays audio through the system's default output device.

    play_beep  — synthesises a confirmation tone entirely in-process (numpy)
    play_wav   — decodes and plays WAV bytes received from the TTS API (soundfile)

    Both methods block until playback is complete, which is deliberate:
    the caller relies on completion to enforce the playback-protection window
    (wake-word and microphone are inactive while this class is playing).
    """

    _BEEP_SAMPLE_RATE: int = 44_100  # standard audio rate for better DAC compatibility

    def play_beep(
        self,
        freq_hz: float = 440.0,
        duration_ms: int = 200,
        volume: float = 0.45,
    ) -> None:
        """Generate and play a sine-wave confirmation tone (440 Hz, 200 ms)."""
        try:
            import sounddevice as sd
        except ImportError:
            logger.debug("sounddevice not available — skipping beep")
            return
        try:
            n = int(self._BEEP_SAMPLE_RATE * duration_ms / 1000)
            t = np.linspace(0.0, duration_ms / 1000.0, n, endpoint=False)
            # Apply a short fade-in/out to avoid clicking artefacts.
            envelope = np.ones(n, dtype=np.float32)
            fade = min(100, n // 4)
            envelope[:fade] = np.linspace(0.0, 1.0, fade)
            envelope[-fade:] = np.linspace(1.0, 0.0, fade)
            wave = (np.sin(2.0 * np.pi * freq_hz * t) * envelope * volume * 32_767).astype(np.int16)
            sd.play(wave, self._BEEP_SAMPLE_RATE)
            sd.wait()
        except Exception as exc:
            logger.debug("Beep failed (non-fatal): %s", exc)

    def play_wav(self, wav_bytes: bytes) -> None:
        """Decode and play WAV bytes. Blocks until playback is complete."""
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as exc:
            logger.error("Playback dependency missing: %s", exc)
            return
        try:
            data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="int16")
            sd.play(data, sample_rate)
            sd.wait()
        except Exception as exc:
            logger.error("Playback failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# NovaListener  (main orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class NovaListener:
    """
    Orchestrates the full wake-word → interaction pipeline.

    Phase 3 flow
    ────────────
      _run_loop:
        while running:
          _wait_for_wake_word()  ← continuous low-overhead detection
          _handle_interaction()  ← one complete request/response cycle

      _handle_interaction:
        beep → record → transcribe → chat API → speak API → play → cooldown

    Playback protection (requirement §7)
    ─────────────────────────────────────
    Protection is inherent in the sequential architecture: the microphone
    InputStream is always closed before AudioPlayer.play_wav() is called, and
    does not reopen until _wait_for_wake_word() is entered again after the
    cooldown. NOVA cannot hear its own voice.

    Phase 4 foundation
    ──────────────────
    • _session_id is preserved across interactions → conversation context
      persists between turns without extra wiring.
    • _handle_interaction() is designed to be extended to return a
      ConversationState so _run_loop can stay in conversation mode (shorter
      silence timeout, no wake word required) for multiple exchanges before
      reverting to wake-word listening after inactivity.
    • Stub:  _enter_conversation_mode()  (not implemented)
    """

    def __init__(self, config: ListenerConfig) -> None:
        self.cfg = config
        self.detector = WakeWordDetector(
            sensitivity=config.wake_word_sensitivity,
            model_path=config.wake_word_model,
        )
        self.recorder = AudioRecorder(
            silence_threshold_rms=config.silence_threshold_rms,
            silence_duration_sec=config.silence_duration_sec,
            max_duration_sec=config.max_recording_sec,
        )
        self.stt = SpeechToText(
            model_name=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )
        self.client = NovaClient(
            base_url=config.nova_api_url,
            timeout=config.api_timeout_sec,
        )
        self.player = AudioPlayer()
        self._running: bool = False
        # Preserved across interactions so NOVA remembers conversation context.
        # Phase 4: also drives continuous-conversation session management.
        self._session_id: Optional[str] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load models, verify connectivity, then enter the listen loop."""
        try:
            self._startup()
            self._running = True
            self._run_loop()
        except KeyboardInterrupt:
            print("\n\n👋  NOVA Listener stopped.")
        except RuntimeError as exc:
            print(f"\n❌  Startup failed: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            logger.critical("Unhandled exception: %s", exc, exc_info=True)
            sys.exit(1)
        finally:
            self._running = False

    def _startup(self) -> None:
        print()
        print("━" * 54)
        print("  NOVA Listener — Phase 3  (wake-word voice mode)")
        print("━" * 54)
        print(f"  API       : {self.cfg.nova_api_url}")
        print(f"  STT model : {self.cfg.whisper_model}  ({self.cfg.whisper_device}/{self.cfg.whisper_compute_type})")
        print(f"  Sensitivity: {self.cfg.wake_word_sensitivity}  •  Silence: {self.cfg.silence_duration_sec}s")
        print()

        # STT model load (blocking, ~1–5 s)
        print("  Loading STT model ...", end="", flush=True)
        self.stt.load()
        print("  ✓")

        # Wake-word model load
        print("  Loading wake-word model ...", end="", flush=True)
        self.detector.load()
        print("  ✓")

        # NOVA API check (non-fatal — server may be started later)
        print("  Checking NOVA API ...", end="", flush=True)
        if self.client.ping():
            print("  ✓ online")
        else:
            print(f"  ✗ offline (will retry per request)")

        print()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while self._running:
            print(f'\n👂  Listening for "Hey NOVA"...')
            detected = self._wait_for_wake_word()

            if not detected:
                # _wait_for_wake_word returns False on microphone error or shutdown.
                time.sleep(0.5)
                continue

            print("\n🔔  Wake word detected!")
            self.detector.reset()  # clear model state to avoid re-triggering

            try:
                self._handle_interaction()
            except Exception as exc:
                # Interaction errors must never crash the listener.
                logger.error("Interaction error (returning to listen mode): %s", exc, exc_info=True)

    # ── Wake-word detection ───────────────────────────────────────────────────

    def _wait_for_wake_word(self) -> bool:
        """
        Open a microphone InputStream and feed 80 ms chunks to the wake-word model.

        Returns True when the wake word is detected.
        Returns False if the microphone is unavailable or _running becomes False.

        The InputStream is closed as soon as this method returns, releasing the
        microphone before the recording phase begins.
        """
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice not installed.  Run: pip install sounddevice"
            ) from exc

        audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.debug("Wake-word stream: %s", status)
            try:
                audio_q.put_nowait(indata.copy())
            except queue.Full:
                pass  # a dropped frame causes at most ~80 ms of missed detection

        try:
            with sd.InputStream(
                samplerate=WakeWordDetector.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=WakeWordDetector.CHUNK_SAMPLES,
                callback=_callback,
            ):
                while self._running:
                    try:
                        chunk = audio_q.get(timeout=0.5)
                        if self.detector.process_chunk(chunk.flatten()):
                            return True
                    except queue.Empty:
                        continue

        except OSError as exc:
            logger.error("Microphone unavailable: %s", exc)
            time.sleep(2.0)
        except Exception as exc:
            logger.error("Wake-word listener error: %s", exc)
            time.sleep(1.0)

        return False

    # ── Interaction cycle ─────────────────────────────────────────────────────

    def _handle_interaction(self) -> None:
        """
        One complete request/response cycle.  Never raises.

        Steps:
          1. Confirmation beep    — signals NOVA is ready
          2. Record user speech   — VAD auto-stop, max 30 s
          3. Transcribe           — faster-whisper
          4. Chat API call        — POST /chat (blocking)
          5. Speak API call       — POST /speak → WAV bytes
          6. Playback             — microphone stays closed during this step
          7. Cooldown             — 1 s pause before re-enabling wake-word

        Phase 4 extension point:
          This method could return a ConversationState enum.  If CONTINUE,
          _run_loop skips wake-word detection and calls _handle_interaction again
          with a shorter silence timeout (set on self.recorder before the call).
          After INACTIVITY_TIMEOUT with no speech, it returns EXIT and the loop
          reverts to wake-word listening.
        """

        # ── 1. Confirmation beep ──────────────────────────────────────────────
        self.player.play_beep(
            freq_hz=self.cfg.beep_freq_hz,
            duration_ms=self.cfg.beep_duration_ms,
        )

        # ── 2. Record ─────────────────────────────────────────────────────────
        print("🎙️   Recording... (silence to stop)")
        audio = self.recorder.record()
        if audio is None:
            logger.warning("Recording produced no audio — returning to listen mode")
            return

        # ── 3. Transcribe ─────────────────────────────────────────────────────
        text = self.stt.transcribe(audio)
        if not text:
            print("  (nothing transcribed — returning to listen mode)")
            return
        print(f"📝  Karl: {text}")

        # ── 4. Chat API ───────────────────────────────────────────────────────
        print("🤔  NOVA is thinking...")
        result = self.client.chat(text, session_id=self._session_id)
        if result is None:
            print("  ❌  NOVA is unreachable.  Is the server running?")
            return

        # Persist session_id for conversation continuity (Phase 4).
        self._session_id = result.session_id
        logger.debug("Session: %s", self._session_id)

        # ── 5. Speak API ──────────────────────────────────────────────────────
        print("🔊  Playing response...")
        audio_bytes = self.client.speak(result.response, result.language)

        # ── 6. Playback ───────────────────────────────────────────────────────
        # The microphone InputStream is closed at this point (we exited
        # _wait_for_wake_word's `with` block before _handle_interaction was
        # called).  NOVA cannot hear its own voice during playback.
        if audio_bytes:
            self.player.play_wav(audio_bytes)
        else:
            # TTS unavailable — print response as text fallback.
            print(f"  NOVA: {result.response}")

        # ── 7. Cooldown ───────────────────────────────────────────────────────
        # Extra buffer after sd.wait() to let room reverb decay before the
        # microphone reopens.
        time.sleep(self.cfg.cooldown_sec)

    # ── Phase 4 stub ──────────────────────────────────────────────────────────

    def _enter_conversation_mode(self) -> None:
        """
        Phase 4 — not implemented.

        Intended behaviour:
          • Shorten silence timeout (e.g. 2 s instead of 30 s)
          • Skip wake-word detection between turns
          • Loop _handle_interaction until inactivity timeout
          • Restore normal settings and return to _run_loop on exit
        """
        raise NotImplementedError("Phase 4 conversation mode is not yet implemented")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    listener = NovaListener(config)
    listener.start()


if __name__ == "__main__":
    main()

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "NOVA"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    # Remote NOVA server (used as TTS fallback when Piper is not installed locally)
    NOVA_SERVER_URL: Optional[str] = None

    # ── PostgreSQL ────────────────────────────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "jarvis"
    DB_USER: str = "jarvis"
    DB_PASSWORD: str = "password"

    @property
    def db_dsn(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # ── Ollama (local LLM) ────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_TOP_P: float = 0.9
    OLLAMA_NUM_PREDICT: int = 1024

    # ── Claude API (future — complex tasks) ──────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-sonnet-4-6"

    # ── Kokoro TTS ────────────────────────────────────────────────────────
    KOKORO_MODEL: str = "./kokoro/kokoro-v1.0.int8.onnx"
    KOKORO_VOICES: str = "./kokoro/voices-v1.0.bin"
    KOKORO_VOICE_FR: str = "ff_siwis"
    KOKORO_VOICE_EN: str = "af_heart"
    KOKORO_SPEED: float = 1.0

    # ── faster-whisper STT ────────────────────────────────────────────────
    WHISPER_MODEL: str = "base"         # tiny / base / small / medium / large-v3
    WHISPER_DEVICE: str = "cpu"         # cpu / cuda
    WHISPER_COMPUTE_TYPE: str = "int8"  # int8 (CPU) / float16 (GPU) / float32

    # ── ChromaDB (future — vector memory) ────────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION: str = "nova_memory"


settings = Settings()

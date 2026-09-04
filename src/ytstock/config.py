"""Runtime configuration.

All settings come from environment variables (or a ``.env`` file). Nothing here
performs I/O; ``get_settings()`` is cached so the pipeline shares one instance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Credentials -------------------------------------------------------
    youtube_api_key: str = Field(default="", description="YouTube Data API v3 key.")
    anthropic_api_key: str = Field(
        default="",
        description="Optional; the SDK also resolves ANTHROPIC_API_KEY / `ant auth login`.",
    )

    # --- Storage -----------------------------------------------------------
    database_url: str = "sqlite:///data/ytstock.db"
    reports_dir: Path = Path("reports")
    sources_file: Path = Path("configs/sources.yaml")

    # --- Discovery ---------------------------------------------------------
    market_timezone: str = "America/New_York"
    top_n_videos: int = 10
    min_duration_seconds: int = 180  # drop Shorts and clips
    max_duration_seconds: int = 7200
    min_view_count: int = 1000
    region_code: str = "US"
    relevance_language: str = "en"

    # --- Transcripts -------------------------------------------------------
    transcript_languages: list[str] = ["en", "en-US", "en-GB"]
    yt_proxy_url: str = Field(
        default="",
        description="Generic HTTP(S) proxy for transcript fetches (cloud IPs are often blocked).",
    )
    webshare_proxy_username: str = ""
    webshare_proxy_password: str = ""
    whisper_fallback: bool = Field(
        default=False,
        description="Transcribe audio with faster-whisper when captions are unavailable "
        "(requires the `whisper` extra).",
    )
    whisper_model: str = Field(
        default="small",
        description="faster-whisper model; multilingual 'small' handles Chinese/English, "
        "'large-v3-turbo' is better but slower on CPU.",
    )
    whisper_beam_size: int = 1
    whisper_backend: Literal["auto", "faster", "mlx"] = Field(
        default="auto",
        description="'mlx' = Apple-GPU mlx-whisper (macOS arm64, `mlx` extra); 'faster' = "
        "CPU faster-whisper; 'auto' prefers mlx when importable.",
    )
    whisper_mlx_model: str = "mlx-community/whisper-large-v3-turbo"

    # --- LLM backend -------------------------------------------------------
    llm_backend: Literal["auto", "api", "claude-cli"] = Field(
        default="auto",
        description="'api' = Anthropic SDK with an API key; 'claude-cli' = shell out to "
        "`claude -p` on your Claude Code subscription; 'auto' = api if a key is set, else cli.",
    )
    claude_cli_path: str = "claude"
    claude_cli_model: str = Field(default="", description="Model/alias for `claude --model`.")
    claude_cli_max_turns: int = 25

    # --- YouTube backend ---------------------------------------------------
    youtube_backend: Literal["auto", "api", "ytdlp"] = Field(
        default="auto",
        description="'api' = YouTube Data API (needs key); 'ytdlp' = keyless scraping via "
        "yt-dlp; 'auto' = api if a key is set, else ytdlp.",
    )
    ytdlp_hydrate_workers: int = 6

    # --- Claude ------------------------------------------------------------
    claude_model: str = "claude-opus-5"
    report_language: str = Field(
        default="English",
        description="Language for all model-written output (reports, briefs). Transcripts "
        "may be in any language.",
    )
    claude_video_effort: EffortLevel = "medium"
    claude_synthesis_effort: EffortLevel = "high"
    claude_factcheck_effort: EffortLevel = "high"
    claude_brief_effort: EffortLevel = "xhigh"
    factcheck_max_searches: int = 12
    brief_max_searches: int = 10
    claude_max_tokens: int = 16000
    claude_timeout_seconds: float = 1800.0
    claude_fallbacks: bool = Field(
        default=True,
        description="Enable server-side refusal fallbacks (beta) so a safety decline "
        "re-runs on a fallback model instead of failing the video.",
    )
    max_transcript_chars: int = Field(
        default=400_000,
        description="Hard ceiling before a transcript is rejected rather than silently cut.",
    )
    analysis_concurrency: int = 3
    transcript_retention_days: int = Field(
        default=0,
        description="If > 0, `ytstock run`/`brief` drop transcript text (not analyses) for videos "
        "older than this many days at the end of each run. 0 = keep forever.",
    )
    report_json_retention_days: int = Field(
        default=0, description="If > 0, delete per-day report JSON files older than this."
    )

    # --- Observability -----------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    @field_validator("transcript_languages", mode="before")
    @classmethod
    def _split_languages(cls, v: object) -> object:
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

    @property
    def sqlite_path(self) -> Path | None:
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            return Path(self.database_url[len(prefix) :])
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

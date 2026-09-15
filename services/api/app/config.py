"""Service configuration from environment variables (or a `.env` file in the working directory).

Relative paths resolve against the working directory; run the API from the repository root.
Model artifacts are located through MODEL_DIR/manifest.json (see ADR-021).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_dir: Path = Path("./models")
    data_dir: Path = Path("./data")
    app_state_dir: Path = Field(
        default=Path("./data/app"), description="Runtime state: analysis database and stored samples"
    )
    cors_origins: str = Field(default="http://localhost:3000", description="Comma-separated allowed origins")
    max_upload_mb: float = Field(default=10.0, gt=0)
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def database_path(self) -> Path:
        return self.app_state_dir / "analyses.sqlite3"

    @property
    def sample_dir(self) -> Path:
        return self.app_state_dir / "samples"

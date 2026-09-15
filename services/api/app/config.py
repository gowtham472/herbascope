"""Service configuration from environment variables (or a `.env` file in the working directory).

Relative paths resolve against the working directory; run the API from the repository root.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.inference.screening_pipeline import ArtifactPaths


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_dir: Path = Path("./models")
    data_dir: Path = Path("./data")
    reference_index: Path = Path("./models/indexes/references.faiss")
    reference_metadata: Path = Path("./models/indexes/reference_metadata.json")
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

    @cached_property
    def artifacts(self) -> ArtifactPaths:
        return ArtifactPaths.from_layout(self.model_dir, self.reference_index, self.reference_metadata)

    @property
    def database_path(self) -> Path:
        return self.app_state_dir / "analyses.sqlite3"

    @property
    def sample_dir(self) -> Path:
        return self.app_state_dir / "samples"

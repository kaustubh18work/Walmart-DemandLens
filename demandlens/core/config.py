from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Walmart DemandLens"
    version: str = "2.0.0"
    log_level: str = "INFO"
    data_dir: Path = ROOT / "data"
    model_dir: Path = ROOT / "models"
    report_dir: Path = ROOT / "reports"
    model_timeout_seconds: int = 900

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_prefix="",
        extra="ignore",
    )


settings = Settings()

for directory in (
    settings.model_dir,
    settings.report_dir,
    settings.report_dir / "charts",
    settings.report_dir / "csv",
    settings.report_dir / "pdf",
):
    directory.mkdir(parents=True, exist_ok=True)

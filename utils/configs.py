from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class YamlBaseConfig(BaseSettings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources = (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

        yaml_path = cls.model_config.get("yaml_file", None)
        if yaml_path:
            sources += (
                YamlConfigSettingsSource(
                    settings_cls=settings_cls,
                    yaml_file=yaml_path,
                    yaml_file_encoding=cls.model_config.get(
                        "yaml_file_encoding", "utf-8"
                    ),
                ),
            )  # type: ignore
        return sources


class GlobalConfig(YamlBaseConfig):
    log_level: str = Field(description="The logging level for the application.")
    seed: int = Field(description="The seed for reproducibility.")
    model_config = SettingsConfigDict(
        yaml_file="configs/global.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )


class MlflowLoggerConfig(YamlBaseConfig):
    """Configuration for the Mlflow logger implementation."""

    tracking_uri: AnyHttpUrl | None = Field(
        description="The tracking URI for the Mlflow logger.",
        default=None,
    )
    remote_tracking_uri: AnyHttpUrl | None = Field(
        description="The remote tracking URI for the Mlflow logger.",
        default=None,
    )
    instance: str = Field(description="The instance name for the Mlflow logger.")
    remote_flag: bool = Field(description="Whether to use remote tracking.")
    trace: bool = Field(description="Whether to enable tracing.")
    templates_path: Path = Field(description="The path to the templates directory.")
    artifact_path: str = Field(description="The path to the artifacts directory.")
    run_name: str | None = Field(
        description="The name of the MLflow run.", default=None
    )

    model_config = SettingsConfigDict(
        yaml_file="configs/mlflow_logger.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )

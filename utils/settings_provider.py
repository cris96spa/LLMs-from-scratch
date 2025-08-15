from utils.settings import GlobalSettings, MlflowLoggerConfig
from utils.singleton import SingletonMeta


class SettingsProvider(meta=SingletonMeta):
    """Singleton class to provide global settings for the application."""

    def __init__(self):
        self._global_settings = GlobalSettings()
        self._mlflow_settings = MlflowLoggerConfig()

    @property
    def global_settings(self) -> GlobalSettings:
        """Get the global settings."""
        return self._global_settings

    @property
    def mlflow_settings(self) -> MlflowLoggerConfig:
        """Get the MLflow logger settings."""
        return self._mlflow_settings

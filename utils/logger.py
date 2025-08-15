import sys

from loguru import logger

from utils.settings_provider import SettingsProvider

global_settings = SettingsProvider().global_settings

# Configure the logger
logger.remove()
logger.add(
    sys.stdout,
    level=global_settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {extra[app]} | {message}",
)

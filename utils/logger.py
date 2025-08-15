import sys

from loguru import logger

from utils.configs_provider import BaseSettingsProvider

global_settings = BaseSettingsProvider().global_config

# Configure the logger
logger.remove()
logger.add(
    sys.stdout,
    level=global_settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {extra[app]} | {message}",
)

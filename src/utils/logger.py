from loguru import logger
from pathlib import Path
from src.utils.config_loader import config

LOG_PATH = Path(config.log["log_path"])
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# 初始化日志
logger.add(
    LOG_PATH,
    level=config.log["log_level"],
    rotation="1 day",
    retention="7 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# 导出全局logger
__all__ = ["logger"]
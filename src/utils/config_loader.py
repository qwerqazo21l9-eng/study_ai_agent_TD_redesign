from pathlib import Path
import yaml
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

class Config(BaseSettings):
    model: dict
    rag: dict
    log: dict
    data: dict
    tools: dict = {}  # 工具配置
    dedup: dict = {}  # 去重配置
    memory: dict = {}  # 记忆配置
    agent: dict = {}  # Agent 配置
    api: dict = {}  # API 服务器配置
    data_processing: dict = {}  # 数据自动化处理配置

    @classmethod
    def load_config(cls) -> "Config":
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

# 全局配置实例
config = Config.load_config()
"""
Embedding 模型封装
"""

from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.config_loader import config
from src.utils.logger import logger


def get_embedding_model():
    """
    获取 Embedding 模型实例（单例）
    """
    model_name = config.model.get("embedding_model_name", "BAAI/bge-base-zh-v1.5")
    
    logger.info(f"初始化 Embedding 模型: {model_name}")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )


# 全局单例
_embedding_model = None


def get_embedding_modelSingleton():
    """获取单例 Embedding 模型"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model

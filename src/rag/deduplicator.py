"""
文档去重模块
- 文件级去重：SHA256 哈希 + 注册表
- Chunk级去重：SimHash + 海明距离
"""

import hashlib
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from src.utils.config_loader import config
from src.utils.logger import logger


@dataclass
class ChunkInfo:
    """Chunk元信息"""
    chunk_id: str
    simhash: int
    file_hash: str
    source: str


class DocumentRegistry:
    """
    文件级去重注册表
    记录已入库文件的SHA256哈希和对应的chunk_ids
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        if registry_path:
            self.registry_path = Path(registry_path)
        else:
            self.registry_path = Path(config.dedup["registry_path"])
        
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, list[str]] = {}  # file_hash -> chunk_ids
        self._load()
    
    def _load(self):
        """从磁盘加载注册表"""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self._registry = json.load(f)
            logger.info(f"Loaded document registry: {len(self._registry)} files")
        else:
            logger.info("No existing registry, starting fresh")
    
    def _save(self):
        """保存注册表到磁盘"""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """计算文件的SHA256哈希"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def is_duplicate(self, file_path: Path) -> bool:
        """检查文件是否已入库"""
        file_hash = self.compute_file_hash(file_path)
        is_dup = file_hash in self._registry
        if is_dup:
            logger.info(f"Duplicate file detected: {file_path.name} (hash: {file_hash[:16]}...)")
        return is_dup
    
    def register(self, file_path: Path, chunk_ids: list[str]):
        """
        注册新文件
        chunk_ids: 该文件产生的所有chunk_id列表
        """
        file_hash = self.compute_file_hash(file_path)
        self._registry[file_hash] = chunk_ids
        self._save()
        logger.info(f"Registered file: {file_path.name}, {len(chunk_ids)} chunks")
    
    def remove(self, file_hash: str):
        """从注册表中移除文件（用于更新场景）"""
        if file_hash in self._registry:
            del self._registry[file_hash]
            self._save()
            logger.info(f"Removed file from registry: {file_hash[:16]}...")
    
    def get_chunk_ids(self, file_hash: str) -> list[str]:
        """获取已注册文件的chunk_ids"""
        return self._registry.get(file_hash, [])


def _tokenize(text: str) -> list[str]:
    """简单分词"""
    # 简化为按空格/标点分割
    import re
    tokens = re.findall(r'\w+', text.lower())
    return tokens


def simhash(text: str, hashbits: int = 64) -> int:
    """
    计算文本的SimHash值
    
    Args:
        text: 输入文本
        hashbits: 哈希位数，默认64位
    
    Returns:
        int: SimHash值
    """
    # 分词
    tokens = _tokenize(text)
    if not tokens:
        return 0
    
    # 获取n-gram哈希
    from src.utils.config_loader import config
    
    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=config.model["embedding_model_name"],
            model_kwargs={"device": "cpu"}
        )
        
        # 用embedding向量做SimHash（更准确）
        vectors = embedding_model.embed_documents([text])
        if vectors:
            # 将embedding向量转为二值向量
            vec = vectors[0]
            mean_val = sum(vec) / len(vec)
            hash_int = 0
            for i, v in enumerate(vec[:hashbits]):
                if v > mean_val:
                    hash_int |= (1 << i)
            return hash_int
    except Exception as e:
        logger.warning(f"Embedding-based simhash failed, using basic hash: {e}")
    
    # 降级：使用MD5哈希模拟
    hash_int = 0
    for i, token in enumerate(tokens):
        h = int(hashlib.md5(token.encode()).hexdigest()[:16], 16)
        hash_int ^= (h if i % 2 == 0 else ~h)
    
    return abs(hash_int) % (2 ** hashbits)


def hamming_distance(h1: int, h2: int, bits: int = 64) -> int:
    """
    计算两个SimHash的海明距离
    """
    x = h1 ^ h2
    total = 0
    while x:
        total += 1
        x &= x - 1
    return total


class ChunkDeduplicator:
    """
    Chunk级去重器
    使用SimHash过滤近似重复的文本块
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Args:
            similarity_threshold: 相似度阈值（0-1）
                                  阈值越高越严格，越接近1表示越不允许重复
                                  0.95 = 允许最多3位海明距离差异
        """
        self.similarity_threshold = similarity_threshold
        # 根据阈值计算允许的最大海明距离
        # 相似度 = 1 - (海明距离 / 总位数)
        self.max_hamming_distance = int((1 - similarity_threshold) * 64)
        self._seen_chunks: list[tuple[int, str]] = []  # (simhash, chunk_id)
        logger.info(f"ChunkDeduplicator initialized, max_hamming_distance={self.max_hamming_distance}")
    
    def deduplicate(self, chunks: list, get_id_fn=None) -> list:
        """
        去重
        
        Args:
            chunks: Document对象列表（LangChain格式）
            get_id_fn: 可选，从chunk获取chunk_id的函数
        
        Returns:
            去重后的chunks列表
        """
        if not chunks:
            return []
        
        original_count = len(chunks)
        deduped_chunks = []
        
        for chunk in chunks:
            # 获取chunk_id
            chunk_id = chunk.metadata.get("chunk_id", str(hash(chunk.page_content)))
            if get_id_fn:
                chunk_id = get_id_fn(chunk)
            
            # 计算SimHash
            chunk_simhash = simhash(chunk.page_content)
            
            # 检查是否与已见过的chunk重复
            is_duplicate = False
            for seen_simhash, seen_id in self._seen_chunks:
                distance = hamming_distance(chunk_simhash, seen_simhash)
                if distance <= self.max_hamming_distance:
                    logger.debug(f"Chunk {chunk_id} is duplicate of {seen_id} (distance={distance})")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduped_chunks.append(chunk)
                self._seen_chunks.append((chunk_simhash, chunk_id))
        
        removed_count = original_count - len(deduped_chunks)
        if removed_count > 0:
            logger.info(f"Chunk deduplication: removed {removed_count}/{original_count} duplicates")
        
        return deduped_chunks
    
    def reset(self):
        """重置已见chunk记录（用于新会话/新文档库）"""
        self._seen_chunks = []
        logger.info("Chunk deduplicator reset")


def create_deduplicators() -> tuple[DocumentRegistry, ChunkDeduplicator]:
    """
    工厂函数：创建去重器实例
    """
    registry = DocumentRegistry()
    
    dedup_config = config.dedup if hasattr(config, 'dedup') else {}
    threshold = config.rag.get("chunk_similarity_threshold", 0.95) if hasattr(config, 'rag') else 0.95
    
    chunk_dedup = ChunkDeduplicator(similarity_threshold=threshold)
    
    return registry, chunk_dedup

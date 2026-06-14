from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredImageLoader
)
from src.utils.config_loader import config
from src.utils.logger import logger


class DocumentProcessor:
    def __init__(self):
        self.chunk_size = config.rag["chunk_size"]
        self.chunk_overlap = config.rag["chunk_overlap"]
        self.use_semantic_chunk = config.rag.get("use_semantic_chunk", True)
        
        # 基础分块器（用于超长块的二次切分）
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        
        logger.info(f"DocumentProcessor initialized (semantic_chunk={self.use_semantic_chunk})")

    def load_single_document(self, file_path: str | Path):
        """加载单个文件，支持PDF/TXT/DOCX/图片"""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # 根据后缀选择加载器
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif file_path.suffix.lower() in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(file_path))
        elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            loader = UnstructuredImageLoader(str(file_path), ocr_languages=["ch_sim", "en"])
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return None

        docs = loader.load()
        # 数据清洗（去空行、去乱码）
        for doc in docs:
            doc.page_content = self._clean_text(doc.page_content)
            # 设置source元数据
            doc.metadata["source"] = file_path.name
        return docs

    def _clean_text(self, text: str) -> str:
        """文本清洗：去空行、去冗余"""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def split_documents(self, docs):
        """文档分块 - 支持语义分块"""
        if not docs:
            return []
        
        all_chunks = []
        
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            
            # 根据文件类型选择分块策略
            if self.use_semantic_chunk and doc.metadata.get("source", "").endswith(".md"):
                # Markdown文件使用语义分块
                chunks = self._semantic_split(doc, source)
            else:
                # 其他文件使用传统分块
                chunks = self._basic_split(doc, source)
            
            all_chunks.extend(chunks)
        
        logger.info(f"Split {len(docs)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def _semantic_split(self, doc, source: str):
        """
        语义分块：按Markdown标题切分
        
        策略：
        1. 先按H1标题分割
        2. 再按H2/H3标题分割
        3. 超长块用fallback_splitter二次切分
        """
        # Markdown标题分割器
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False
        )
        
        # 按标题分割
        splits = markdown_splitter.split_text(doc.page_content)
        
        chunks = []
        for i, split in enumerate(splits):
            content = split.page_content.strip()
            
            # 跳过空块
            if not content or len(content) < 20:
                continue
            
            # 超长块二次切分
            if len(content) > self.chunk_size * 2:
                sub_splits = self.fallback_splitter.split_text(content)
                for j, sub in enumerate(sub_splits):
                    if sub.strip():
                        chunk = split.model_copy()
                        chunk.page_content = sub.strip()
                        chunk.metadata["chunk_id"] = f"{source}_{i}_{j}"
                        chunk.metadata["source"] = source
                        chunk.metadata["section"] = split.metadata.get("Header1", "") or split.metadata.get("Header2", "")
                        chunks.append(chunk)
            else:
                split.metadata["chunk_id"] = f"{source}_{i}"
                split.metadata["source"] = source
                chunks.append(split)
        
        return chunks
    
    def _basic_split(self, doc, source: str):
        """
        基础分块：使用RecursiveCharacterTextSplitter
        """
        splits = self.fallback_splitter.split_documents([doc])
        
        chunks = []
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = f"{source}_{i}"
            split.metadata["source"] = source
            chunks.append(split)
        
        return chunks

    def process_file(self, file_path: str | Path):
        """一站式处理：加载→清洗→分块"""
        docs = self.load_single_document(file_path)
        if not docs:
            return None
        split_docs = self.split_documents(docs)
        logger.info(f"Processed {file_path.name}: {len(split_docs)} chunks")
        return split_docs

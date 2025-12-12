from pathlib import Path
from typing import List, Dict, Any
import json
import tiktoken

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    from langchain_core.documents import Document as LangchainDocument
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    from langchain.schema import Document as LangchainDocument

from config import Config


class ChunkProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def chunk_document(self, file_path: Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract frontmatter if present
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()
        
        chunks = []
        
        if self.config.chunking.strategy == "recursive":
            chunks = self._recursive_chunk(content, metadata)
        elif self.config.chunking.strategy == "markdown":
            chunks = self._markdown_chunk(content, metadata)
        elif self.config.chunking.strategy == "hybrid":
            chunks = self._hybrid_chunk(content, metadata)
        else:
            chunks = self._recursive_chunk(content, metadata)
        
        return chunks
    
    def _recursive_chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            separators=self.config.chunking.separators,
            length_function=self.count_tokens,
        )
        
        docs = splitter.create_documents([content])
        chunks = []
        
        for i, doc in enumerate(docs):
            chunk_metadata = {
                **metadata,
                "chunk_id": i,
                "chunk_index": i,
                "total_chunks": len(docs),
            }
            
            chunks.append({
                "text": doc.page_content,
                "metadata": chunk_metadata,
                "token_count": self.count_tokens(doc.page_content),
            })
        
        return chunks
    
    def _markdown_chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=self.config.chunking.include_headers,
        )
        
        md_header_splits = markdown_splitter.split_text(content)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            length_function=self.count_tokens,
        )
        
        all_chunks = []
        for md_split in md_header_splits:
            if self.count_tokens(md_split.page_content) > self.config.chunking.chunk_size:
                sub_chunks = splitter.split_documents([md_split])
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(md_split)
        
        chunks = []
        for i, doc in enumerate(all_chunks):
            chunk_metadata = {
                **metadata,
                **doc.metadata,
                "chunk_id": i,
                "chunk_index": i,
                "total_chunks": len(all_chunks),
            }
            
            chunks.append({
                "text": doc.page_content,
                "metadata": chunk_metadata,
                "token_count": self.count_tokens(doc.page_content),
            })
        
        return chunks
    
    def _hybrid_chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        
        try:
            md_splits = markdown_splitter.split_text(content)
        except:
            md_splits = [LangchainDocument(page_content=content, metadata={})]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            separators=self.config.chunking.separators,
            length_function=self.count_tokens,
        )
        
        all_chunks = []
        for md_split in md_splits:
            if self.count_tokens(md_split.page_content) > self.config.chunking.chunk_size:
                sub_chunks = splitter.split_documents([md_split])
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(md_split)
        
        chunks = []
        for i, doc in enumerate(all_chunks):
            chunk_metadata = {
                **metadata,
                **doc.metadata,
                "chunk_id": i,
                "chunk_index": i,
                "total_chunks": len(all_chunks),
            }
            
            text = doc.page_content
            if self.config.chunking.include_headers and doc.metadata:
                header_parts = []
                for key, value in doc.metadata.items():
                    if key.startswith("Header") and value:
                        header_parts.append(value)
                if header_parts:
                    text = " ".join(header_parts) + "\n\n" + text
            
            chunks.append({
                "text": text,
                "metadata": chunk_metadata,
                "token_count": self.count_tokens(text),
            })
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    def load_chunks(self, input_path: Path) -> List[Dict[str, Any]]:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


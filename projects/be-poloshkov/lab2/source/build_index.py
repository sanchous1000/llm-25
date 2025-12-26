import os
import re
import json
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config import load_config, Config


@dataclass
class Chunk:
    id: str
    text: str
    book: str
    file: str
    section: str
    token_count: int
    char_count: int
    embedding: Optional[list] = None
    
    def to_dict(self):
        return asdict(self)


class TokenBasedChunker:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def split_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text.strip())
            
            if end >= len(tokens):
                break
            start = end - overlap_tokens
        
        return chunks


class MarkdownChunker(TokenBasedChunker):
    def __init__(self, config: Config):
        super().__init__(config)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def extract_sections(self, content: str) -> list[dict]:
        sections = []
        matches = list(self.header_pattern.finditer(content))
        
        if not matches:
            return [{"header": "", "level": 0, "content": content}]
        
        # text before first header
        if matches[0].start() > 0:
            preamble = content[:matches[0].start()].strip()
            if preamble:
                sections.append({"header": "", "level": 0, "content": preamble})
        
        for i, match in enumerate(matches):
            level = len(match.group(1))
            header = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()
            
            sections.append({
                "header": header,
                "level": level,
                "content": section_content
            })
        
        return sections
    
    def chunk_document(self, content: str, book: str, file: str) -> list[Chunk]:
        sections = self.extract_sections(content)
        chunks = []
        
        for section in sections:
            header = section["header"]
            text = section["content"]
            
            if not text:
                continue
            
            if self.config.include_headers_in_chunk and header:
                text = f"# {header}\n\n{text}"
            
            token_count = self.count_tokens(text)
            
            if token_count <= self.config.chunk_size:
                chunk_id = self._make_chunk_id(book, file, header, text)
                chunks.append(Chunk(
                    id=chunk_id,
                    text=text,
                    book=book,
                    file=file,
                    section=header,
                    token_count=token_count,
                    char_count=len(text)
                ))
            else:
                sub_chunks = self.split_by_tokens(
                    text, 
                    self.config.chunk_size, 
                    self.config.chunk_overlap
                )
                for i, sub_text in enumerate(sub_chunks):
                    chunk_id = self._make_chunk_id(book, file, header, sub_text, i)
                    chunks.append(Chunk(
                        id=chunk_id,
                        text=sub_text,
                        book=book,
                        file=file,
                        section=f"{header} (part {i+1})" if header else f"part {i+1}",
                        token_count=self.count_tokens(sub_text),
                        char_count=len(sub_text)
                    ))
        
        return chunks
    
    def _make_chunk_id(self, book: str, file: str, section: str, text: str, part: int = 0) -> str:
        content = f"{book}:{file}:{section}:{part}:{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class RecursiveChunker(TokenBasedChunker):
    def __init__(self, config: Config):
        super().__init__(config)
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def chunk_document(self, content: str, book: str, file: str) -> list[Chunk]:
        raw_chunks = self._recursive_split(content, self.separators)
        chunks = []
        
        for i, text in enumerate(raw_chunks):
            chunk_id = self._make_chunk_id(book, file, text, i)
            chunks.append(Chunk(
                id=chunk_id,
                text=text,
                book=book,
                file=file,
                section=f"chunk_{i+1}",
                token_count=self.count_tokens(text),
                char_count=len(text)
            ))
        
        return chunks
    
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self.split_by_tokens(text, self.config.chunk_size, self.config.chunk_overlap)
        
        sep = separators[0]
        rest_seps = separators[1:]
        
        if sep:
            parts = text.split(sep)
        else:
            parts = list(text)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            test_chunk = current_chunk + sep + part if current_chunk else part
            
            if self.count_tokens(test_chunk) <= self.config.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    if self.count_tokens(current_chunk) <= self.config.chunk_size:
                        chunks.append(current_chunk)
                    else:
                        chunks.extend(self._recursive_split(current_chunk, rest_seps))
                current_chunk = part
        
        if current_chunk:
            if self.count_tokens(current_chunk) <= self.config.chunk_size:
                chunks.append(current_chunk)
            else:
                chunks.extend(self._recursive_split(current_chunk, rest_seps))
        
        return [c.strip() for c in chunks if c.strip()]
    
    def _make_chunk_id(self, book: str, file: str, text: str, idx: int) -> str:
        content = f"{book}:{file}:{idx}:{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


def get_chunker(config: Config):
    if config.splitter_type == "markdown":
        return MarkdownChunker(config)
    elif config.splitter_type == "recursive":
        return RecursiveChunker(config)
    else:
        return MarkdownChunker(config)


class EmbeddingBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
    
    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> list[Chunk]:
        texts = [c.text for c in chunks]
        
        print(f"Building embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
        
        return chunks
    
    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()


def build_index(config: Config = None, rebuild: bool = False):
    if config is None:
        config = load_config()
    
    raw_dir = Path(config.raw_docs_dir)
    processed_dir = Path(config.processed_dir)
    index_dir = Path(config.index_dir)
    
    config_hash = hashlib.md5(
        f"{config.chunk_size}:{config.chunk_overlap}:{config.splitter_type}:{config.embedding_model}".encode()
    ).hexdigest()[:8]
    
    chunks_file = processed_dir / f"chunks_{config_hash}.json"
    embeddings_file = index_dir / f"embeddings_{config_hash}.npy"
    meta_file = index_dir / f"meta_{config_hash}.json"
    
    if not rebuild and chunks_file.exists() and embeddings_file.exists():
        print(f"Loading existing index from {chunks_file}")
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        embeddings = np.load(embeddings_file)
        
        chunks = [Chunk(**{k: v for k, v in c.items() if k != 'embedding'}) for c in chunks_data]
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
        
        return chunks
    
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Manifest not found. Run fetch_docs.py first.")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    chunker = get_chunker(config)
    all_chunks = []
    
    print(f"Chunking documents (size={config.chunk_size} tokens, overlap={config.chunk_overlap})...")
    
    for file_info in manifest["files"]:
        file_path = Path(file_info["path"])
        if not file_path.exists():
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        chunks = chunker.chunk_document(content, file_info["book"], file_info["file"])
        all_chunks.extend(chunks)
        print(f"  {file_info['book']}/{file_info['file']}: {len(chunks)} chunks")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    embedder = EmbeddingBuilder(config)
    all_chunks = embedder.embed_chunks(all_chunks)
    
    chunks_data = [c.to_dict() for c in all_chunks]
    for c in chunks_data:
        c.pop('embedding', None)
    
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    embeddings = np.array([c.embedding for c in all_chunks])
    np.save(embeddings_file, embeddings)
    
    meta = {
        "config_hash": config_hash,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "splitter_type": config.splitter_type,
        "embedding_model": config.embedding_model,
        "embedding_dim": embedder.embedding_dim,
        "total_chunks": len(all_chunks),
        "total_tokens": sum(c.token_count for c in all_chunks)
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nIndex built: {len(all_chunks)} chunks, {meta['total_tokens']} tokens")
    print(f"Saved to: {chunks_file}, {embeddings_file}")
    
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build document index")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap in tokens")
    parser.add_argument("--splitter", choices=["markdown", "recursive"], default="markdown")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    
    args = parser.parse_args()
    
    config = load_config(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        splitter_type=args.splitter,
        embedding_model=args.embedding_model
    )
    
    build_index(config, rebuild=args.rebuild)


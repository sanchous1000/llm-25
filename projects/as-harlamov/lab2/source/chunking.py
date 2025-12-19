import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    from langchain_core.documents import Document as LangchainDocument
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    from langchain.schema import Document as LangchainDocument

from config import Config


class ChunkProcessor:
    def __init__(self, config: Config, embedding_generator):
        self.config = config
        self.embedding_generator = embedding_generator
        self.max_model_length = embedding_generator.get_max_sequence_length()

        if self.max_model_length and self.config.chunking.chunk_size > self.max_model_length:
            warnings.warn(
                f"chunk_size ({self.config.chunking.chunk_size}) exceeds model max sequence length "
                f"({self.max_model_length}). Large chunks will be split during processing.",
                UserWarning,
            )

    def count_tokens(self, text: str) -> int:
        return self.embedding_generator.count_tokens(text)

    def chunk_document(self, file_path: Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()

        if self.config.chunking.strategy == "recursive":
            chunks = self._recursive_chunk(content, metadata)
        elif self.config.chunking.strategy == "markdown":
            chunks = self._markdown_chunk(content, metadata)
        elif self.config.chunking.strategy == "hybrid":
            chunks = self._hybrid_chunk(content, metadata)
        else:
            chunks = self._recursive_chunk(content, metadata)

        chunks = self._ensure_chunks_within_limit(chunks, metadata)

        return chunks

    def _extract_page_number(self, text: str, metadata: Dict[str, Any]) -> Optional[int]:
        page_num = metadata.get("page")
        if page_num:
            return page_num

        for key, value in metadata.items():
            if key.startswith("Header") and value:
                match = re.search(r'Page\s+(\d+)', value, re.IGNORECASE)
                if match:
                    return int(match.group(1))

        match = re.search(r'##\s*Page\s+(\d+)', text)
        if match:
            return int(match.group(1))

        lines = text.split('\n')
        for line in lines[:5]:
            match = re.search(r'##\s*Page\s+(\d+)', line)
            if match:
                return int(match.group(1))

        return None

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

            page_num = self._extract_page_number(doc.page_content, {})
            if page_num:
                chunk_metadata["page"] = page_num

            chunks.append(
                {
                    "text": doc.page_content,
                    "metadata": chunk_metadata,
                    "token_count": self.count_tokens(doc.page_content),
                },
            )

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

            page_num = self._extract_page_number(doc.page_content, doc.metadata)
            if page_num:
                chunk_metadata["page"] = page_num

            chunks.append(
                {
                    "text": doc.page_content,
                    "metadata": chunk_metadata,
                    "token_count": self.count_tokens(doc.page_content),
                },
            )

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

            page_num = self._extract_page_number(doc.page_content, doc.metadata)
            if page_num:
                chunk_metadata["page"] = page_num

            text = doc.page_content
            if self.config.chunking.include_headers and doc.metadata:
                header_parts = []
                for key, value in doc.metadata.items():
                    if key.startswith("Header") and value:
                        header_parts.append(value)
                if header_parts:
                    text = " ".join(header_parts) + "\n\n" + text

            chunks.append(
                {
                    "text": text,
                    "metadata": chunk_metadata,
                    "token_count": self.count_tokens(text),
                },
            )

        return chunks

    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    def load_chunks(self, input_path: Path) -> List[Dict[str, Any]]:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_chunks_within_limit(
        self,
        chunks: List[Dict[str, Any]],
        __metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self.max_model_length:
            return chunks

        result_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_model_length,
            chunk_overlap=min(self.config.chunking.chunk_overlap, self.max_model_length // 4),
            separators=self.config.chunking.separators,
            length_function=self.count_tokens,
        )

        for chunk in chunks:
            token_count = chunk.get("token_count", self.count_tokens(chunk["text"]))

            if token_count <= self.max_model_length:
                result_chunks.append(chunk)
            else:
                sub_chunks = splitter.split_text(chunk["text"])
                base_metadata = chunk.get("metadata", {}).copy()

                for sub_idx, sub_text in enumerate(sub_chunks):
                    sub_metadata = {
                        **base_metadata,
                        "original_chunk_id": base_metadata.get("chunk_id"),
                        "sub_chunk_index": sub_idx,
                        "is_split": True,
                    }

                    result_chunks.append(
                        {
                            "text": sub_text,
                            "metadata": sub_metadata,
                            "token_count": self.count_tokens(sub_text),
                        },
                    )

        for i, chunk in enumerate(result_chunks):
            chunk["metadata"]["chunk_id"] = i
            chunk["metadata"]["chunk_index"] = i
            chunk["metadata"]["total_chunks"] = len(result_chunks)

        return result_chunks

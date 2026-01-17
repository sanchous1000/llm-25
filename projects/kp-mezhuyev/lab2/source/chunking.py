"""
Модуль для разбиения Markdown документов на чанки.

Поддерживает различные стратегии разбиения:
- recursive: рекурсивное разбиение с учетом структуры
- markdown: разбиение по заголовкам H1-H3
- hybrid: комбинация markdown + окно с overlap
"""
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import tiktoken


@dataclass
class Chunk:
    """Представляет один чанк документа."""
    
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Метаданные из исходного документа
    doc_id: str = ""
    source_path: str = ""
    title: str = ""
    
    # Информация о позиции в документе
    chunk_index: int = 0
    header: str = ""
    header_level: int = 0
    
    # Статистика
    token_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует чанк в словарь для сериализации.
        
        Конвертирует datetime объекты в ISO строки для JSON сериализации.
        """
        # Преобразуем метаданные, конвертируя datetime в строки
        serializable_metadata: dict[str, Any] = {}
        for key, value in self.metadata.items():
            if isinstance(value, datetime):
                serializable_metadata[key] = value.isoformat()
            elif hasattr(value, 'isoformat'):  # Другие объекты с isoformat (date, time)
                serializable_metadata[key] = value.isoformat()
            else:
                serializable_metadata[key] = value
        
        return {
            "id": self.id,
            "text": self.text,
            "metadata": serializable_metadata,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "header": self.header,
            "header_level": self.header_level,
            "token_count": self.token_count,
        }


class TokenCounter:
    """Счетчик токенов для различных моделей."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Инициализирует счетчик токенов.
        
        Args:
            encoding_name: Название кодировки (cl100k_base для GPT, 
                          r50k_base для старых моделей)
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except KeyError:
            # Fallback на cl100k_base если указанная кодировка не найдена
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count(self, text: str) -> int:
        """Подсчитывает количество токенов в тексте."""
        return len(self.encoding.encode(text))


class BaseSplitter:
    """Базовый класс для сплиттеров."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        include_headers_in_text: bool = True,
        min_chunk_size: int = 100,
    ):
        """Инициализирует базовый сплиттер.
        
        Args:
            chunk_size: Максимальный размер чанка в токенах.
            chunk_overlap: Размер перекрытия между чанками в токенах.
            include_headers_in_text: Включать заголовки в текст чанка.
            min_chunk_size: Минимальный размер чанка в токенах.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_headers_in_text = include_headers_in_text
        self.min_chunk_size = min_chunk_size
        self.token_counter = TokenCounter()
    
    def split(self, text: str, metadata: dict) -> list[Chunk]:
        """Разбивает текст на чанки.
        
        Args:
            text: Текст для разбиения.
            metadata: Метаданные документа.
        
        Returns:
            Список чанков.
        """
        raise NotImplementedError


class RecursiveSplitter(BaseSplitter):
    """Рекурсивный сплиттер с учетом структуры документа."""
    
    def __init__(self, separators: list[str] | None = None, **kwargs):
        """Инициализирует рекурсивный сплиттер.
        
        Args:
            separators: Список разделителей для рекурсивного разбиения.
                       По умолчанию: ["\n\n", "\n", " ", ""]
        """
        super().__init__(**kwargs)
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split(self, text: str, metadata: dict) -> list[Chunk]:
        """Рекурсивно разбивает текст на чанки."""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        # Разбиваем по разделителям
        parts = self._split_recursive(text, self.separators)
        
        for part in parts:
            part_tokens = self.token_counter.count(part)
            
            if current_tokens + part_tokens <= self.chunk_size:
                current_chunk += part
                current_tokens += part_tokens
            else:
                # Сохраняем текущий чанк
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk.strip(),
                        metadata,
                        chunk_index,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Начинаем новый чанк с overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = self._get_overlap_text(
                        current_chunk,
                        self.chunk_overlap,
                    )
                    current_chunk = overlap_text + part
                    current_tokens = self.token_counter.count(current_chunk)
                else:
                    current_chunk = part
                    current_tokens = part_tokens
        
        # Добавляем последний чанк
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                metadata,
                chunk_index,
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Рекурсивно разбивает текст по разделителям."""
        if not separators:
            return [text]
        
        separator = separators[0]
        parts = text.split(separator)
        
        if len(parts) == 1:
            # Разделитель не найден, пробуем следующий
            return self._split_recursive(text, separators[1:])
        
        result = []
        for part in parts:
            if separator:
                result.append(part + separator)
            else:
                result.append(part)
        
        return result
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Получает текст для overlap из конца предыдущего чанка."""
        tokens = self.token_counter.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        overlap_tokens_list = tokens[-overlap_tokens:]
        return self.token_counter.encoding.decode(overlap_tokens_list)
    
    def _create_chunk(
        self,
        text: str,
        metadata: dict,
        chunk_index: int,
    ) -> Chunk:
        """Создает объект Chunk."""
        chunk_id = hashlib.sha1(
            f"{metadata.get('doc_id', '')}_{chunk_index}".encode()
        ).hexdigest()[:16]
        
        return Chunk(
            id=chunk_id,
            text=text,
            metadata=metadata.copy(),
            doc_id=metadata.get("doc_id", ""),
            source_path=metadata.get("source_path", ""),
            title=metadata.get("title", ""),
            chunk_index=chunk_index,
            token_count=self.token_counter.count(text),
        )


class MarkdownSplitter(BaseSplitter):
    """Сплиттер для Markdown с учетом заголовков."""
    
    def __init__(
        self,
        header_levels: list[str] = None,
        **kwargs,
    ):
        """Инициализирует Markdown сплиттер.
        
        Args:
            header_levels: Список уровней заголовков для разбиения.
                          Например: ['h1', 'h2', 'h3']
        """
        super().__init__(**kwargs)
        self.header_levels = header_levels or ['h1', 'h2', 'h3']
        # Преобразуем 'h1' -> 1, 'h2' -> 2 и т.д.
        self.header_nums = [int(h[1:]) for h in self.header_levels]
    
    def split(self, text: str, metadata: dict) -> list[Chunk]:
        """Разбивает Markdown текст по заголовкам."""
        # Парсим структуру документа
        sections = self._parse_markdown_structure(text)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_text = section["text"]
            section_tokens = self.token_counter.count(section_text)
            
            if section_tokens <= self.chunk_size:
                # Секция помещается в один чанк
                chunk = self._create_chunk(
                    section_text,
                    metadata,
                    chunk_index,
                    section["header"],
                    section["header_level"],
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Секция слишком большая, разбиваем рекурсивно
                sub_chunks = self._split_large_section(
                    section_text,
                    metadata,
                    chunk_index,
                    section["header"],
                    section["header_level"],
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
        
        return chunks
    
    def _parse_markdown_structure(self, text: str) -> list[dict]:
        """Парсит структуру Markdown документа."""
        lines = text.split("\n")
        sections = []
        current_section = {
            "header": "",
            "header_level": 0,
            "text": "",
        }
        
        for line in lines:
            # Проверяем, является ли строка заголовком
            header_match = re.match(r"^(#+)\s+(.+)$", line)
            
            if header_match:
                level = len(header_match.group(1))
                
                # Если это заголовок нужного уровня, начинаем новую секцию
                if level in self.header_nums:
                    # Сохраняем предыдущую секцию
                    if current_section["text"].strip():
                        sections.append(current_section)
                    
                    # Начинаем новую секцию
                    header_text = header_match.group(2).strip()
                    # Удаляем якорные ссылки для заголовка в метаданных
                    header_clean = re.sub(r'\s*\{[^}]*#\w[^}]*\}\s*$', '', header_text).strip()
                    
                    current_section = {
                        "header": header_clean,
                        "header_level": level,
                        "text": line + "\n" if self.include_headers_in_text else "",
                    }
                else:
                    # Заголовок другого уровня, добавляем к текущей секции
                    if self.include_headers_in_text:
                        current_section["text"] += line + "\n"
            else:
                # Обычная строка
                current_section["text"] += line + "\n"
        
        # Добавляем последнюю секцию
        if current_section["text"].strip():
            sections.append(current_section)
        
        return sections
    
    def _split_large_section(
        self,
        text: str,
        metadata: dict,
        start_index: int,
        header: str,
        header_level: int,
    ) -> list[Chunk]:
        """Разбивает большую секцию на подчанки."""
        # Используем рекурсивный сплиттер для больших секций
        recursive_splitter = RecursiveSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            include_headers_in_text=self.include_headers_in_text,
            min_chunk_size=self.min_chunk_size,
        )
        
        # Добавляем заголовок к метаданным
        enhanced_metadata = metadata.copy()
        enhanced_metadata["section_header"] = header
        enhanced_metadata["section_header_level"] = header_level
        
        chunks = recursive_splitter.split(text, enhanced_metadata)
        
        # Обновляем индексы и добавляем информацию о заголовке
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = start_index + i
            chunk.header = header
            chunk.header_level = header_level
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        metadata: dict,
        chunk_index: int,
        header: str = "",
        header_level: int = 0,
    ) -> Chunk:
        """Создает объект Chunk с информацией о заголовке."""
        chunk_id = hashlib.sha1(
            f"{metadata.get('doc_id', '')}_{chunk_index}".encode()
        ).hexdigest()[:16]
        
        return Chunk(
            id=chunk_id,
            text=text,
            metadata=metadata.copy(),
            doc_id=metadata.get("doc_id", ""),
            source_path=metadata.get("source_path", ""),
            title=metadata.get("title", ""),
            chunk_index=chunk_index,
            header=header,
            header_level=header_level,
            token_count=self.token_counter.count(text),
        )


class HybridSplitter(BaseSplitter):
    """Гибридный сплиттер: markdown + окно с overlap."""
    
    def __init__(
        self,
        header_levels: list[str] = None,
        window_size: int = 3,
        **kwargs,
    ):
        """Инициализирует гибридный сплиттер.
        
        Args:
            header_levels: Уровни заголовков для первичного разбиения.
            window_size: Количество секций в окне для overlap.
        """
        super().__init__(**kwargs)
        self.markdown_splitter = MarkdownSplitter(
            header_levels=header_levels,
            chunk_size=self.chunk_size * 2,  # Больше для первичного разбиения
            chunk_overlap=0,
            include_headers_in_text=self.include_headers_in_text,
            min_chunk_size=self.min_chunk_size,
        )
        self.window_size = window_size
    
    def split(self, text: str, metadata: dict) -> list[Chunk]:
        """Разбивает текст гибридным методом."""
        # Сначала разбиваем по заголовкам
        sections = self.markdown_splitter._parse_markdown_structure(text)
        
        chunks = []
        chunk_index = 0
        
        # Создаем окна с overlap
        for i in range(len(sections)):
            # Определяем границы окна
            start = max(0, i - self.window_size // 2)
            end = min(len(sections), i + self.window_size // 2 + 1)
            
            # Объединяем секции в окне
            window_text = ""
            window_header = sections[i]["header"]
            window_level = sections[i]["header_level"]
            
            for j in range(start, end):
                if self.include_headers_in_text:
                    window_text += sections[j]["text"]
                else:
                    # Убираем заголовки из текста, оставляем только в метаданных
                    section_lines = sections[j]["text"].split("\n")
                    text_lines = [
                        line for line in section_lines
                        if not re.match(r"^#+\s+", line)
                    ]
                    window_text += "\n".join(text_lines) + "\n"
            
            # Проверяем размер окна
            window_tokens = self.token_counter.count(window_text)
            
            if window_tokens <= self.chunk_size:
                chunk = self._create_chunk(
                    window_text.strip(),
                    metadata,
                    chunk_index,
                    window_header,
                    window_level,
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Окно слишком большое, разбиваем рекурсивно
                recursive_splitter = RecursiveSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    include_headers_in_text=self.include_headers_in_text,
                    min_chunk_size=self.min_chunk_size,
                )
                sub_chunks = recursive_splitter.split(window_text, metadata)
                for chunk in sub_chunks:
                    chunk.chunk_index = chunk_index
                    chunk.header = window_header
                    chunk.header_level = window_level
                    chunk_index += 1
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        metadata: dict,
        chunk_index: int,
        header: str = "",
        header_level: int = 0,
    ) -> Chunk:
        """Создает объект Chunk."""
        chunk_id = hashlib.sha1(
            f"{metadata.get('doc_id', '')}_{chunk_index}".encode()
        ).hexdigest()[:16]
        
        return Chunk(
            id=chunk_id,
            text=text,
            metadata=metadata.copy(),
            doc_id=metadata.get("doc_id", ""),
            source_path=metadata.get("source_path", ""),
            title=metadata.get("title", ""),
            chunk_index=chunk_index,
            header=header,
            header_level=header_level,
            token_count=self.token_counter.count(text),
        )


def get_splitter(
    splitter_type: Literal["recursive", "markdown", "hybrid"],
    config: dict,
) -> BaseSplitter:
    """Создает сплиттер указанного типа.
    
    Args:
        splitter_type: Тип сплиттера.
        config: Словарь с конфигурацией.
    
    Returns:
        Экземпляр сплиттера.
    """
    chunking_config = config.get("chunking", {})
    
    common_params = {
        "chunk_size": chunking_config.get("chunk_size", 512),
        "chunk_overlap": chunking_config.get("chunk_overlap", 50),
        "include_headers_in_text": chunking_config.get(
            "include_headers_in_text",
            True,
        ),
        "min_chunk_size": chunking_config.get("min_chunk_size", 100),
    }
    
    if splitter_type == "recursive":
        return RecursiveSplitter(**common_params)
    elif splitter_type == "markdown":
        return MarkdownSplitter(
            header_levels=chunking_config.get("header_levels", ["h1", "h2", "h3"]),
            **common_params,
        )
    elif splitter_type == "hybrid":
        return HybridSplitter(
            header_levels=chunking_config.get("header_levels", ["h1", "h2", "h3"]),
            **common_params,
        )
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")

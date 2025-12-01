import re
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer


class MarkdownSplitter:
    
    def __init__(self, chunk_size, chunk_overlap, header_levels, include_headers_in_text, min_chunk_size, tokenizer_model=None):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.header_levels = header_levels
        self.include_headers_in_text = include_headers_in_text
        
        # Загружаем токенизатор для модели эмбеддингов
        if tokenizer_model is None:
            tokenizer_model = 'BAAI/bge-base-en-v1.5'
        
        # Маппинг имен моделей на полные пути (как в embeddings.py)
        model_mapping = {
            'e5-large-v2': 'intfloat/e5-large-v2',
            'e5-base-v2': 'intfloat/e5-base-v2',
            'bge-base-en-v1.5': 'BAAI/bge-base-en-v1.5',
            'bge-large-en-v1.5': 'BAAI/bge-large-en-v1.5',
            'bge-m3': 'BAAI/bge-m3',
        }
        
        model_path = model_mapping.get(tokenizer_model, tokenizer_model)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Загружен токенизатор для модели: {model_path}")
        except Exception as e:
            print(f"Ошибка загрузки токенизатора {model_path}: {e}")
            print(f"  Используется токенизатор по умолчанию: BAAI/bge-base-en-v1.5")
            self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        self.header_patterns = {
            'h1': re.compile(r'^# (.+)$', re.MULTILINE),
            'h2': re.compile(r'^## (.+)$', re.MULTILINE),
            'h3': re.compile(r'^### (.+)$', re.MULTILINE),
            'h4': re.compile(r'^#### (.+)$', re.MULTILINE)
        }
    
    def count_tokens(self, text):
        """Подсчитывает количество токенов с помощью токенизатора модели."""
        # Используем encode, который возвращает список token IDs
        # add_special_tokens=False для точного подсчета только текстовых токенов
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def _create_chunk(self, text, metadata, chunk_index, headers = None):
        """Создает чанк с метаданными."""
        chunk = {
            'text': text.strip(),
            'metadata': {
                **metadata,
                'chunk_index': chunk_index,
                'token_count': self.count_tokens(text),
            }
        }
        
        if headers:
            chunk['metadata']['headers'] = headers
        
        return chunk
    
    def _extract_sections(self, content):
        # Удаляем frontmatter если есть
        if content.startswith('---'):
            end_idx = content.find('---', 3)
            if end_idx != -1:
                content = content[end_idx + 3:].strip()
        
        sections = []
        lines = content.split('\n')
        current_section = {
            'headers': [],
            'content': [],
            'level': 0
        }
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                hashes, title = header_match.groups()
                level = len(hashes)
                header_key = f'h{level}'
                # Если заголовок нужного уровня, начинаем новую секцию
                if header_key in self.header_levels:
                    if current_section['content']:
                        sections.append(current_section)

                    new_headers = []
                    for h in current_section['headers']:
                        if int(h['level'][1:]) < level:
                            new_headers.append(h)
                    
                    new_headers.append({
                        'level': header_key,
                        'title': title.strip(),
                        'text': line
                    })
                    
                    current_section = {
                        'headers': new_headers,
                        'content': [line] if self.include_headers_in_text else [],
                        'level': level
                    }
                else:
                    current_section['content'].append(line)
            else:
                current_section['content'].append(line)
        
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _split_large_section(self, section_text, headers):
        paragraphs = re.split(r'\n\n+', section_text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            # Если параграф слишком большой, разбиваем по предложениям
            if para_tokens > self.chunk_size:
                # Если есть накопленный чанк, сохраняем его
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                    
                sentences = re.split(r'([.!?]+\s+)', para)
                sentence_chunk = []
                sentence_tokens = 0
                
                for i in range(0, len(sentences), 2):
                    sent = sentences[i]
                    if i + 1 < len(sentences):
                        sent += sentences[i + 1]
                    
                    sent_tokens = self.count_tokens(sent)
                    
                    if sentence_tokens + sent_tokens > self.chunk_size:
                        if sentence_chunk:
                            chunks.append(' '.join(sentence_chunk))
                        sentence_chunk = [sent]
                        sentence_tokens = sent_tokens
                    else:
                        sentence_chunk.append(sent)
                        sentence_tokens += sent_tokens
                
                if sentence_chunk:
                    current_chunk = [' '.join(sentence_chunk)]
                    current_tokens = sentence_tokens
            
            # Обычный случай
            elif current_tokens + para_tokens > self.chunk_size:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                
                # Начинаем новый с перекрытием
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    overlap_tokens = self.count_tokens(overlap_text)
                    if overlap_tokens <= self.chunk_overlap:
                        current_chunk = [overlap_text, para]
                        current_tokens = overlap_tokens + para_tokens
                    else:
                        current_chunk = [para]
                        current_tokens = para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def split_document(self, content, metadata):
        """
        Разбивает документ на чанки по заголовкам.
        
        Args:
            content: Содержимое документа
            metadata: Метаданные документа
            
        Returns:
            Список чанков
        """
        sections = self._extract_sections(content)
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_text = '\n'.join(section['content']).strip()
            
            if not section_text or self.count_tokens(section_text) < self.min_chunk_size:
                continue
            
            # Извлекаем названия заголовков
            header_titles = [h['title'] for h in section['headers']]
            
            # Если секция помещается в один чанк
            if self.count_tokens(section_text) <= self.chunk_size:
                chunk = self._create_chunk(
                    section_text,
                    metadata,
                    chunk_index,
                    header_titles
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Разбиваем большую секцию
                sub_chunks = self._split_large_section(section_text, section['headers'])
                
                for sub_chunk_text in sub_chunks:
                    if self.count_tokens(sub_chunk_text) >= self.min_chunk_size:
                        chunk = self._create_chunk(
                            sub_chunk_text,
                            metadata,
                            chunk_index,
                            header_titles
                        )
                        chunks.append(chunk)
                        chunk_index += 1
        
        return chunks
import os
import json
from pathlib import Path
from typing import List, Dict
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

import config


class ChunkSplitter:
    def __init__(self, chunk_size=512, overlap=50, splitter_type='recursive'):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.splitter_type = splitter_type
    
    def split_recursive(self, text, separators=['\n\n', '\n', '. ', ' ']):
        chunks = []
        
        def split_text(txt, seps):
            if not seps or len(txt) <= self.chunk_size:
                return [txt] if txt.strip() else []
            
            sep = seps[0]
            parts = txt.split(sep)
            result = []
            current = ''
            
            for part in parts:
                if len(current) + len(part) + len(sep) <= self.chunk_size:
                    current += part + sep
                else:
                    if current:
                        result.append(current)
                    if len(part) > self.chunk_size:
                        result.extend(split_text(part, seps[1:]))
                        current = ''
                    else:
                        current = part + sep
            
            if current:
                result.append(current)
            
            return result
        
        raw_chunks = split_text(text, separators)
        
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and self.overlap > 0:
                prev_chunk = raw_chunks[i - 1]
                overlap_text = prev_chunk[-self.overlap:] if len(prev_chunk) > self.overlap else prev_chunk
                chunk = overlap_text + chunk
            
            chunks.append(chunk.strip())
        
        return chunks
    
    def split_by_headers(self, text):
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_header = ''
        
        for line in lines:
            if re.match(r'^#{1,3}\s+', line):
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append((current_header, chunk_text))
                
                current_header = line
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
                if len('\n'.join(current_chunk)) > self.chunk_size:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append((current_header, chunk_text))
                    current_chunk = []
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append((current_header, chunk_text))
        
        return [(h, c) for h, c in chunks if c]
    
    def split(self, text):
        if self.splitter_type == 'recursive':
            return self.split_recursive(text)
        elif self.splitter_type == 'headers':
            return [f"{h}\n\n{c}" if h else c for h, c in self.split_by_headers(text)]
        else:
            raise ValueError(f"Unknown splitter type: {self.splitter_type}")


class EmbeddingEngine:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', use_sparse=False):
        self.model_name = model_name
        self.use_sparse = use_sparse
        
        if not use_sparse:
            self.model = SentenceTransformer(model_name)
        else:
            self.bm25 = None
    
    def encode_dense(self, texts):
        return self.model.encode(texts, show_progress_bar=True)
    
    def encode_sparse(self, texts):
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        return tokenized
    
    def encode(self, texts):
        if not self.use_sparse:
            return self.encode_dense(texts)
        else:
            return self.encode_sparse(texts)


def build_index(markdown_dir, output_dir, chunk_size=512, overlap=50, 
                splitter_type='recursive', embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                use_sparse=False):
    
    markdown_dir = Path(markdown_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splitter = ChunkSplitter(chunk_size=chunk_size, overlap=overlap, splitter_type=splitter_type)
    embedder = EmbeddingEngine(model_name=embedding_model, use_sparse=use_sparse)
    
    all_chunks = []
    
    metadata_file = markdown_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            docs_metadata = json.load(f)
    else:
        docs_metadata = []
    
    for md_file in markdown_dir.glob('*.md'):
        print(f"Chunking {md_file.name}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = splitter.split(content)
        
        doc_meta = next((m for m in docs_metadata if Path(m['output']).name == md_file.name), {})
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'source': md_file.name,
                'chunk_id': i,
                'original_file': doc_meta.get('source', md_file.name),
                'file_type': doc_meta.get('type', 'markdown')
            })
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Build embeddings")
    
    texts = [c['text'] for c in all_chunks]
    embeddings = embedder.encode(texts)
    
    if not use_sparse:
        for i, chunk in enumerate(all_chunks):
            chunk['embedding'] = embeddings[i].tolist()
    
    chunks_file = output_dir / 'chunks.json'
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    if not use_sparse:
        embeddings_file = output_dir / 'embeddings.npy'
        np.save(embeddings_file, embeddings)
    
    config_data = {
        'chunk_size': chunk_size,
        'overlap': overlap,
        'splitter_type': splitter_type,
        'embedding_model': embedding_model,
        'use_sparse': False,
        'total_chunks': len(all_chunks)
    }
    
    config_file = output_dir / 'index_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Saved to {output_dir}")
    return all_chunks


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='output/markdown')
    parser.add_argument('--output', default='output/index')
    args = parser.parse_args()
    
    build_index(
        args.input, 
        args.output,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.OVERLAP,
        splitter_type=config.SPLITTER_TYPE,
        embedding_model=config.EMBEDDING_MODEL,
        use_sparse=False
    )

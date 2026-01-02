from typing import Callable, Protocol, Any
import langchain_text_splitters
import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from adaptix import Retort, dumper
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

import json
import click
import yaml
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings('ignore', message='Token indices sequence length is longer than the specified maximum')
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


from langchain_core.documents.base import Document


from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def token_length(text: str):
    return len(tokenizer.encode(text, add_special_tokens=True))


MODEL_MAX_TOKENS = tokenizer.model_max_length

HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ("####", "H4"),
    ("#####", "H5"),
]




def extract_chunks(input_path: str, output_path: str, model_name: str, overlap: int, chunk_size: int):
    markdown_splitter = langchain_text_splitters.MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )

    text_splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(
        length_function=token_length, 
        chunk_overlap=overlap, 
        chunk_size=chunk_size
    )

    
    

    loader = DirectoryLoader(
        input_path,
        glob="**/*.md",
        loader_cls=TextLoader,
    )
    documents = loader.load()


    return [
        {
            "content": chunk,
            "metadata": {
                "document": document.metadata | {"id": document_idx},
                "chunk": {
                    "id": chunk_idx,
                },
                "md_header": md_header.metadata | {"id": md_header_idx},
            },
        }
        for document_idx, document in enumerate[Any](documents)
        for md_header_idx, md_header in enumerate[Document](
            (markdown_splitter.split_text(document.page_content))
        )
        for chunk_idx, chunk in enumerate[str](
            text_splitter.split_text(md_header.page_content)
        )
    ]



def vectorize_chunks(chunks: list[str], model: SentenceTransformer):
    embeddings = []
    for idx, chunk in tqdm.tqdm(enumerate[str](chunks)):
        content = chunk["content"]
        metadata = chunk["metadata"]
        embeddings.append(
            {
                "id": idx,
                "vector": model.encode(content),
                "payload": {
                    "metadata": metadata,
                    "text": content,
                },
            }
        )
    return embeddings





@click.command()
@click.option("--config", default="scripts/config.yaml", help="Path to YAML config")
def main(config: str):
    with open(config) as f:
        cfg = yaml.safe_load(f)["build"]
    
    input_path = cfg["input_path"]
    output_path = cfg["output_path"]
    model_name = cfg["model"]
    overlap = cfg["overlap"]
    chunk_size = cfg.get("chunk_size", 250)
    
    embedding_model = SentenceTransformer(model_name)
    chunks = extract_chunks(input_path, output_path, model_name, overlap, chunk_size)
    
    embeddings = vectorize_chunks(chunks, model=embedding_model)
    
    with open(output_path, "w") as f:
        json.dump(
            {"embeddings": embeddings, "config": cfg}, 
            f, 
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
    
    print(f"Created {len(embeddings)} embeddings -> {output_path}")


if __name__ == "__main__":
    main()

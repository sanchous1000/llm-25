"""
Dataset loader for Langfuse.
Uploads evaluation dataset to Langfuse for experiment runs.
"""
import json
from typing import List, Dict
from langfuse import Langfuse

import config


def load_dataset_to_langfuse(
    dataset_path: str,
    dataset_name: str = None,
    description: str = None
) -> str:
    """
    Load a Q&A dataset from JSON file into Langfuse.
    
    Expected JSON format (from lab2):
    [
        {
            "query": "question text",
            "relevant_chunks": [chunk_ids]
        },
        ...
    ]
    
    Returns the dataset ID.
    """
    dataset_name = dataset_name or config.DATASET_NAME
    
    # Initialize Langfuse
    langfuse = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST
    )
    
    # Load data from file
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loading {len(data)} items to Langfuse dataset '{dataset_name}'...")
    
    # Create or get dataset
    dataset = langfuse.create_dataset(
        name=dataset_name,
        description=description or f"RAG evaluation dataset with {len(data)} queries"
    )
    
    # Upload each item
    for i, item in enumerate(data):
        # Prepare input
        input_data = {
            "query": item["query"]
        }
        
        # Prepare expected output (relevant chunks for retrieval evaluation)
        expected_output = {
            "relevant_chunks": item.get("relevant_chunks", [])
        }
        
        # Create dataset item
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=input_data,
            expected_output=expected_output,
            metadata={"item_index": i}
        )
        
        print(f"  Added item {i+1}/{len(data)}: {item['query'][:50]}...")
    
    langfuse.flush()
    print(f"\nDataset '{dataset_name}' created successfully!")
    print(f"View in Langfuse UI: http://localhost:3001/project/datasets")
    
    return dataset_name


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Load dataset to Langfuse')
    parser.add_argument('--load', type=str, required=True,
                       help='Path to dataset JSON file to load')
    parser.add_argument('--dataset-name', type=str, default=config.DATASET_NAME,
                       help='Name for the dataset in Langfuse')
    args = parser.parse_args()
    
    load_dataset_to_langfuse(
        dataset_path=args.load,
        dataset_name=args.dataset_name
    )


if __name__ == '__main__':
    main()

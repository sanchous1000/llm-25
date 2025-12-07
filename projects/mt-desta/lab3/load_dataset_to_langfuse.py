"""
Load a Q&A dataset into Langfuse as a Dataset for evaluation.
"""
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()


def load_evaluation_questions(file_path: Path) -> List[Dict[str, Any]]:
    """Load evaluation questions from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("questions", [])


def create_dataset_in_langfuse(
    langfuse: Langfuse,
    dataset_name: str,
    questions: List[Dict[str, Any]]
) -> str:
    """
    Create a dataset in Langfuse and upload items.
    Returns the dataset name (ID).
    """
    # Create dataset (or get existing)
    try:
        dataset = langfuse.create_dataset(name=dataset_name)
        print(f"Created dataset '{dataset_name}'")
    except Exception as e:
        # Dataset might already exist, try to use it
        print(f"Note: {e}")
        print(f"Using existing dataset '{dataset_name}' or will create new one")
        dataset = None
    
    # Add items to dataset
    for i, item in enumerate(questions, 1):
        # Format input and expected output
        input_data = {
            "question": item["question"],
            "ground_truth_chunks": item.get("ground_truth_chunks", []),
            "expected_topics": item.get("expected_topics", [])
        }
        
        expected_output = {
            "ground_truth_chunks": item.get("ground_truth_chunks", []),
            "expected_topics": item.get("expected_topics", [])
        }
        
        try:
            # Create dataset item
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=input_data,
                expected_output=expected_output
            )
            print(f"  [{i}/{len(questions)}] Added item: {item['question'][:60]}...")
        except Exception as e:
            print(f"  [{i}/{len(questions)}] Error adding item: {e}")
            continue
    
    print(f"\nSuccessfully loaded {len(questions)} items into dataset '{dataset_name}'")
    return dataset_name


def main():
    parser = argparse.ArgumentParser(description="Load dataset into Langfuse")
    parser.add_argument(
        "--questions",
        type=str,
        default=str(Path(__file__).parent.parent / "lab2" / "evaluation_questions.json"),
        help="Path to evaluation questions JSON file"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rag-evaluation-dataset",
        help="Name of the dataset in Langfuse"
    )
    args = parser.parse_args()
    
    # Initialize Langfuse
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3001")
    )
    
    # Load questions
    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"Error: Questions file not found at {questions_path}")
        return
    
    questions = load_evaluation_questions(questions_path)
    print(f"Loaded {len(questions)} questions from {questions_path}")
    
    # Create dataset in Langfuse
    dataset_id = create_dataset_in_langfuse(
        langfuse=langfuse,
        dataset_name=args.dataset_name,
        questions=questions
    )
    
    print(f"\nDataset ID: {dataset_id}")
    print(f"View dataset at: {os.getenv('LANGFUSE_HOST', 'http://localhost:3001')}/datasets/{dataset_id}")


if __name__ == "__main__":
    main()


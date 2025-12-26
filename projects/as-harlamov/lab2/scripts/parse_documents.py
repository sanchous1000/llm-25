#!/usr/bin/env python3
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from parsers import ParserFactory
from config import Config
from tqdm import tqdm


def main():
    project_root = Path(__file__).parent.parent
    config = Config()
    input_dir = project_root / config.documents.get("input_dir", "data/raw")
    output_dir = project_root / config.documents.get("output_dir", "data/processed")

    output_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = [".pdf", ".docx", ".pptx", ".md", ".html", ".htm"]
    files = []
    for ext in supported_extensions:
        files.extend(input_dir.glob(f"**/*{ext}"))

    files = list(set(files))

    if not files:
        print(f"No supported documents found in {input_dir}")
        return

    print(f"Found {len(files)} documents to parse")

    parsed_count = 0
    for file_path in tqdm(files, desc="Parsing documents"):
        try:
            output_path = ParserFactory.parse_file(file_path, output_dir)
            if output_path:
                parsed_count += 1
                print(f"Parsed: {file_path.name} -> {output_path.name}")
        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")

    print(f"\nSuccessfully parsed {parsed_count}/{len(files)} documents")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

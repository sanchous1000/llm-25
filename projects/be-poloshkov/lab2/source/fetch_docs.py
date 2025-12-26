import os
import json
import requests
from pathlib import Path
from datetime import datetime
from config import load_config


GITHUB_API = "https://api.github.com"
REPO_OWNER = "getify"
REPO_NAME = "You-Dont-Know-JS"
BRANCH = "2nd-ed"

BOOKS = [
    "get-started",
    "scope-closures",
    "objects-classes",
    "types-grammar",
]


def get_repo_contents(path: str = "") -> list:
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    params = {"ref": BRANCH}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def download_file(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def fetch_book_files(book_name: str, output_dir: Path) -> list:
    book_dir = output_dir / book_name
    book_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    contents = get_repo_contents(book_name)
    
    for item in contents:
        if item["type"] == "file" and item["name"].endswith(".md"):
            file_path = book_dir / item["name"]
            content = download_file(item["download_url"])
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            downloaded.append({
                "book": book_name,
                "file": item["name"],
                "path": str(file_path),
                "url": item["html_url"],
                "size": len(content)
            })
            print(f"  Downloaded: {item['name']}")
    
    return downloaded


def fetch_all_docs(config=None):
    if config is None:
        config = load_config()
    
    output_dir = Path(config.raw_docs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_files = []
    
    print(f"Fetching YDKJS books from {REPO_OWNER}/{REPO_NAME}...")
    
    for book in BOOKS:
        print(f"\nBook: {book}")
        files = fetch_book_files(book, output_dir)
        all_files.extend(files)
    
    # preface
    print("\nFetching preface...")
    preface_url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/preface.md"
    preface_content = download_file(preface_url)
    preface_path = output_dir / "preface.md"
    with open(preface_path, "w", encoding="utf-8") as f:
        f.write(preface_content)
    all_files.append({
        "book": "preface",
        "file": "preface.md",
        "path": str(preface_path),
        "url": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/blob/{BRANCH}/preface.md",
        "size": len(preface_content)
    })
    
    manifest = {
        "repo": f"{REPO_OWNER}/{REPO_NAME}",
        "branch": BRANCH,
        "fetched_at": datetime.now().isoformat(),
        "files": all_files,
        "total_files": len(all_files),
        "total_size": sum(f["size"] for f in all_files)
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nTotal: {len(all_files)} files, {manifest['total_size']} bytes")
    print(f"Manifest saved to {manifest_path}")
    
    return manifest


if __name__ == "__main__":
    fetch_all_docs()


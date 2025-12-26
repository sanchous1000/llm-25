import requests
import os
import base64
from datetime import datetime

USERNAME = "karpathy"

os.makedirs('corpus', exist_ok=True)
repos_url = f"https://api.github.com/users/{USERNAME}/repos?per_page=100"
repos = requests.get(repos_url).json()

for repo in repos:
    repo_name = repo["name"]
    repo_url = repo["html_url"]
    repo_description = repo.get("description", "")
    created_at = repo.get("created_at", "")
    updated_at = repo.get("updated_at", "")

    readme_url = f"https://api.github.com/repos/{USERNAME}/{repo_name}/readme"
    readme_resp = requests.get(readme_url)

    if readme_resp.status_code != 200:
        print(f"[SKIP] No README in {repo_name}")
        continue

    readme_data = readme_resp.json()
    filename = readme_data.get("name", "README.md")
    readme_path = readme_data.get("path", "README.md")
    content_base64 = readme_data["content"]
    content_bytes = base64.b64decode(content_base64)
    content = content_bytes.decode("utf-8", errors="ignore")
    
    # Добавляем метаданные
    current_date = datetime.now().strftime('%Y-%m-%d')
    # Используем описание из GitHub
    description = repo_description if repo_description else "No description available"
    
    metadata = f"""---
source: {repo_url}
repository: {repo_name}
description: {description}
original_file: {filename}
date_processed: {current_date}
type: README
format: markdown
---

"""
    
    output_path = os.path.join('corpus', f"{repo_name}__{filename}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(metadata + content)

    print(f"[OK] Saved {output_path}")
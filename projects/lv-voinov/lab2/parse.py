import os
import re
from datetime import datetime
from pathlib import Path
from git import Repo
import tempfile
from dotenv import load_dotenv

load_dotenv()

class MarkdownRepoParser:
    def __init__(self, repo_url, output_dir="./output", github_token=None, ignore_dirs=None):
        self.repo_url = repo_url
        self.github_token = github_token
        self.output_dir = Path(output_dir)
        self.temp_dir = None
        self.repo = None
        
        if ignore_dirs is None:
            self.ignore_dirs = {'.git', '.idea', 'node_modules', '__pycache__', 'venv', 'dist', 'build'}
        else:
            self.ignore_dirs = set(ignore_dirs)

    def clone_repo(self):
        url_to_clone = self.repo_url
        
        if self.github_token and self.repo_url.startswith("https://"):
            url_to_clone = self.repo_url.replace("https://", f"https://{self.github_token}@", 1)

        print(f"Клонирование {url_to_clone}")
        try:
            self.temp_dir = tempfile.mkdtemp()
            self.repo = Repo.clone_from(url_to_clone, self.temp_dir)
            repo_name = self.repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            return repo_name
        except Exception as e:
            print(f"Ошибка при клонировании: {e}")
            raise

    def get_file_metadata(self, file_path, repo_name):
        commits = list(self.repo.iter_commits(paths=file_path, max_count=1))
        last_commit = commits[0] if commits else None
        
        date_str = None
        if last_commit:
            timestamp = last_commit.committed_date
            date_str = datetime.fromtimestamp(timestamp).isoformat()
        else:
            date_str = datetime.now().isoformat()
        
        rel_path = Path(file_path).relative_to(self.temp_dir)
        
        section = str(rel_path.parent) if str(rel_path.parent) != '.' else 'root'
        page = rel_path.name

        return {
            "source": self.repo_url, 
            "repo": repo_name,
            "path": str(rel_path),
            "section": section,
            "page": page,
            "date": date_str,
            "format": "markdown"
        }
    def format_metadata_as_comment(self, metadata):
        lines = ["<!-- METADATA START"]
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")
        lines.append("METADATA END -->")
        return "\n".join(lines)

    def normalize_markdown(self, content):
        lines = content.split('\n')
        normalized_lines = []
        
        for line in lines:
            line = line.rstrip()
            
            if re.match(r'^#{1,6}[^\s#]', line):
                line = re.sub(r'^(#+)([^\s#])', r'\1 \2', line)
            
            normalized_lines.append(line)
        
        text = '\n'.join(normalized_lines).strip()
        return text + '\n'

    def process(self):
        if not self.repo:
            repo_name = self.clone_repo()
        else:
            repo_name = self.repo_url.split('/')[-1]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for root, dirs, files in os.walk(self.temp_dir):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for file in files:
                if file.lower().endswith('.md'):
                    full_path = os.path.join(root, file)
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if not content.strip():
                            continue

                        clean_content = self.normalize_markdown(content)
                        
                        metadata = self.get_file_metadata(full_path, repo_name)
                        
                        meta_block = self.format_metadata_as_comment(metadata)
                        
                        final_content = f"{meta_block}\n\n{clean_content}"
                        
                        rel_path = Path(full_path).relative_to(self.temp_dir)
                        output_path = self.output_dir / rel_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(final_content)
                            
                        print(f"Обработан: {rel_path}")
                        count += 1
                        
                    except Exception as e:
                        print(f"Ошибка при чтении файла {full_path}: {e}")

        print(f"Результат сохранен в: {self.output_dir.absolute()}")

if __name__ == "__main__":
    md_parser = MarkdownRepoParser(
        repo_url="https://github.com/modal-labs/gpu-glossary", 
        output_dir="docs",
        github_token=os.environ["GITHUB_TOKEN"]
    )
    
    md_parser.process()
"""
Скрипт для парсинга README из репозиториев GitHub пользователя.
Выполняет задания 1 и 2: выбор источников данных и парсинг в Markdown.
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import base64
import time
from dotenv import load_dotenv

load_dotenv()

class GitHubRepoParser:
    """Парсер репозиториев GitHub для извлечения README файлов."""
    
    def __init__(self, username: str, output_dir: str = "parsed_docs", token: Optional[str] = None):
        """
        Инициализация парсера.
        
        Args:
            username: Имя пользователя GitHub
            output_dir: Директория для сохранения результатов
            token: GitHub Personal Access Token (опционально, увеличивает лимит запросов с 60 до 5000/час)
        """
        self.username = username
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.base_url = "https://api.github.com"
        self.session = requests.Session()
        
        # Настройка заголовков
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-Repo-Parser'
        }
        
        # Добавляем токен, если указан (увеличивает лимит с 60 до 5000 запросов/час)
        if token:
            headers['Authorization'] = f'token {token}'
        elif os.getenv('GITHUB_TOKEN'):
            headers['Authorization'] = f'token {os.getenv("GITHUB_TOKEN")}'
            
        self.session.headers.update(headers)
        
    def get_repositories(self) -> List[Dict]:
        """
        Получает список публичных репозиториев пользователя.
        
        Returns:
            Список словарей с информацией о репозиториях
        """
        repos = []
        page = 1
        per_page = 100
        
        print(f"Получение списка репозиториев пользователя {self.username}...")
        
        while True:
            url = f"{self.base_url}/users/{self.username}/repos"
                
            params = {
                'page': page,
                'per_page': per_page,
                'type': 'all',
                'sort': 'updated'
            }
            
            try:
                response = self.session.get(url, params=params)
                
                # Проверяем rate limit
                if response.status_code == 403:
                    rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', '0')
                    rate_limit_reset = response.headers.get('X-RateLimit-Reset', '0')
                    
                    if rate_limit_remaining == '0':
                        reset_time = int(rate_limit_reset)
                        wait_time = max(0, reset_time - int(time.time())) + 10
                        print(f"  Достигнут лимит запросов. Ожидание {wait_time} секунд...")
                        time.sleep(wait_time)
                        continue
                
                response.raise_for_status()
                
                page_repos = response.json()
                if not page_repos:
                    break
                    
                repos.extend(page_repos)
                print(f"  Получено {len(page_repos)} репозиториев (страница {page})")
                
                # Проверяем, есть ли следующая страница
                if len(page_repos) < per_page:
                    break
                    
                page += 1
                time.sleep(1)  # Увеличиваем задержку для соблюдения rate limit
                
            except requests.exceptions.RequestException as e:
                if "rate limit" in str(e).lower() or "403" in str(e):
                    print(f"  Ошибка rate limit. Ожидание 60 секунд...")
                    time.sleep(60)
                    continue
                print(f"Ошибка при получении репозиториев: {e}")
                break
                
        print(f"Всего найдено {len(repos)} репозиториев")
        return repos
    
    def get_readme(self, repo_name: str) -> Optional[Dict]:
        """
        Получает README файл из репозитория.
        
        Args:
            repo_name: Имя репозитория
            
        Returns:
            Словарь с содержимым README или None
        """
        # Пробуем разные варианты имени README файла
        readme_names = ['README.md', 'readme.md', 'README', 'readme']
        
        for readme_name in readme_names:
            url = f"{self.base_url}/repos/{self.username}/{repo_name}/contents/{readme_name}"
            
            try:
                response = self.session.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('type') == 'file':
                        # Декодируем содержимое
                        content = base64.b64decode(data['content']).decode('utf-8')
                        return {
                            'content': content,
                            'path': data['path'],
                            'sha': data['sha'],
                            'size': data['size']
                        }
                        
            except requests.exceptions.RequestException as e:
                continue
            except Exception as e:
                print(f"  Ошибка при обработке {readme_name}: {e}")
                continue
                
            time.sleep(0.3)
            
        return None
    
    def normalize_markdown(self, content: str, metadata: Dict) -> str:
        """
        Нормализует Markdown с добавлением метаданных.
        
        Args:
            content: Исходное содержимое README
            metadata: Метаданные документа
            
        Returns:
            Нормализованный Markdown с метаданными
        """
        # Формируем заголовок с метаданными
        metadata_block = f"""---
source: {metadata['source']}
repository: {metadata['repository']}
repository_url: {metadata['repository_url']}
path: {metadata['path']}
date_parsed: {metadata['date_parsed']}
date_updated: {metadata.get('date_updated', 'N/A')}
language: {metadata.get('language', 'N/A')}
description: {metadata.get('description', 'N/A')}
stars: {metadata.get('stars', 0)}
---

"""
        
        content = content.strip()
        
        normalized = metadata_block + content
        
        return normalized
    
    def parse_repositories(self) -> List[Dict]:
        """
        Парсит все репозитории и извлекает README.
        
        Returns:
            Список словарей с информацией о спарсенных документах
        """
        repos = self.get_repositories()
        parsed_docs = []
        
        print(f"\nПарсинг README из {len(repos)} репозиториев...")
        
        for i, repo in enumerate(repos, 1):
            repo_name = repo['name']
            print(f"\n[{i}/{len(repos)}] Обработка репозитория: {repo_name}")
            
            readme_data = self.get_readme(repo_name)
            
            if readme_data:
                # Формируем метаданные
                metadata = {
                    'source': 'GitHub',
                    'repository': repo_name,
                    'repository_url': repo['html_url'],
                    'path': readme_data['path'],
                    'date_parsed': datetime.now().isoformat(),
                    'date_updated': repo.get('updated_at', 'N/A'),
                    'language': repo.get('language', 'N/A'),
                    'description': repo.get('description', 'N/A') or 'N/A',
                    'stars': repo.get('stargazers_count', 0),
                    'size': readme_data['size']
                }
                
                # Нормализуем Markdown
                normalized_content = self.normalize_markdown(readme_data['content'], metadata)

                safe_repo_name = repo_name.replace('/', '_').replace('\\', '_')
                output_file = self.output_dir / f"{safe_repo_name}_README.md"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(normalized_content)
                
                print(f"  ✓ README сохранен: {output_file}")
                
                parsed_docs.append({
                    'repo_name': repo_name,
                    'file_path': str(output_file),
                    'metadata': metadata,
                    'content_length': len(normalized_content)
                })
            else:
                print(f"  ✗ README не найден")
            
            time.sleep(0.5)
        
        return parsed_docs

if __name__ == "__main__":
    """Основная функция для запуска парсинга."""
    
    usernames = [
        "IlyaDanilenko",
        "ADmangarakov",
        'sanchous1000',
        'ivanlabb'
    ]
    output_dir = "parsed_docs"
    github_token = os.getenv('GITHUB_TOKEN', '')
    
    print("=" * 60)
    print("Парсинг README из репозиториев GitHub")
    print("=" * 60)
    
    all_parsed_docs = []
    
    for username in usernames:
        print(f"\n{'='*60}")
        print(f"Обработка пользователя: {username}")
        print(f"{'='*60}\n")
        
        parser = GitHubRepoParser(username, output_dir, token=github_token)
        parsed_docs = parser.parse_repositories()
        all_parsed_docs.extend(parsed_docs)
    
    print("\n" + "=" * 60)
    print("Парсинг завершен!")
    print(f"Всего обработано пользователей: {len(usernames)}")
    print(f"Всего спарсено README: {len(all_parsed_docs)}")
    print("=" * 60)
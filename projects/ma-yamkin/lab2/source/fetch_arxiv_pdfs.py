import requests
import time
import os
from urllib.parse import quote
import xml.etree.ElementTree as ET


ARXIV_CATEGORY = "cs.AI"
MAX_PAPERS = 15
OUTPUT_DIR = "../data/raw/arxiv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

query = f"cat:{ARXIV_CATEGORY}"
url = f"http://export.arxiv.org/api/query?search_query={quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_PAPERS}"

response = requests.get(url)
response.raise_for_status()

# Извлекаем ID статей
root = ET.fromstring(response.content)
ns = {'arxiv': 'http://arxiv.org/OAI/2.0/'}

paper_ids = []
for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
    paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
    paper_ids.append(paper_id)

# Скачиваем PDF
for pid in paper_ids:
    pdf_url = f"https://arxiv.org/pdf/{pid}.pdf"
    out_path = os.path.join(OUTPUT_DIR, f"{pid}.pdf")
    if os.path.exists(out_path):
        continue
    print(f"Downloading {pid}...")
    with requests.get(pdf_url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    time.sleep(3)

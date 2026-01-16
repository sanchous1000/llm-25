import os
import json
import hashlib
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from tqdm import tqdm

class FaissVectorStore:
    def __init__(
        self, 
        store_dir: str, 
        dim: int,
        index_type: str = "HNSW", 
        M: int = 32, 
        efConstruction: int = 100, 
        efSearch: int = 64,
        use_quantization: bool = False
    ):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.dim = dim
        self.index_type = index_type
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.use_quantization = use_quantization
        
        self.index_path = self.store_dir / "faiss.index"
        self.meta_path = self.store_dir / "metadata.jsonl"
        self.id_map_path = self.store_dir / "id_mapping.json"
        self.config_path = self.store_dir / "config.json"
        
        self.index = None
        self.id_map = {}

    def get_config_hash(self):
        config_str = f"{self.dim}-{self.index_type}-{self.M}-{self.efConstruction}-{self.use_quantization}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def load_data(self, input_jsonl: Path) -> Tuple[np.ndarray, Dict[int, Dict]]:
        vectors = []
        metadata = {}
        original_ids = []
        
        print(f"Чтение данных из {input_jsonl}")
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    data = json.loads(line)
                    vec = np.array(data['vector'], dtype='float32')
                    
                    if len(vec) != self.dim:
                        raise ValueError(f"Размерность вектора не совпадает: {len(vec)} != {self.dim}")
                    
                    vectors.append(vec)
                    
                    faiss_id = len(vectors) - 1
                    original_id = data['id']
                    
                    self.id_map[faiss_id] = original_id
                    
                    metadata[faiss_id] = {
                        "text": data['text'],
                        "metadata": data['metadata']
                    }
                    
                    original_ids.append(original_id)
                except Exception as e:
                    print(f"Ошибка чтения строки: {e}")
                    continue
        
        vectors_np = np.vstack(vectors).astype('float32')
        print(f"Загружено {len(vectors_np)} векторов размерности {self.dim}")
        return vectors_np, metadata

    def create_index(self):
        print(f"Создание индекса {self.index_type} (M={self.M}, efConstr={self.efConstruction})")
        
        if self.index_type == "HNSW":
            if self.use_quantization:
                quantizer = faiss.ScalarQuantizer(self.dim, faiss.ScalarQuantizer.QT_8bit)
                self.index = faiss.IndexHNSWSQ(self.dim, quantizer, self.M)
            else:
                self.index = faiss.IndexHNSWFlat(self.dim, self.M)

            self.index.hnsw.efConstruction = self.efConstruction
            self.index.hnsw.efSearch = self.efSearch
            
        else:
            self.index = faiss.IndexFlatL2(self.dim)

    def save(self, metadata: Dict[int, Dict]):
        print(f"Сохранение индекса в {self.store_dir}")
        
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            for faiss_id, data in metadata.items():
                record = {"faiss_id": faiss_id, "payload": data}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        with open(self.id_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.id_map, f, indent=2, ensure_ascii=False)

        config = {
            "dim": self.dim,
            "index_type": self.index_type,
            "M": self.M,
            "efConstruction": self.efConstruction,
            "efSearch": self.efSearch,
            "use_quantization": self.use_quantization,
            "vector_count": self.index.ntotal,
            "config_hash": self.get_config_hash()
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
            
        print("Сохранение завершено.")

    def build_from_scratch(self, vectors: np.ndarray):
        self.create_index()
        
        if self.use_quantization and isinstance(self.index, (faiss.IndexHNSWSQ, faiss.IndexIVFPQ)):
            self.index.train(vectors)
        
        print(f"Добавление {vectors.shape[0]} векторов в индекс")
        self.index.add(vectors)
        print(f"Индекс построен, всего записей: {self.index.ntotal}")

    def reset(self):
        if self.store_dir.exists():
            print(f"Удаление старого хранилища в {self.store_dir}")
            shutil.rmtree(self.store_dir)
            self.store_dir.mkdir(parents=True, exist_ok=True)

    def check_status(self):
        if not (self.index_path.exists() and self.config_path.exists()):
            return "MISSING"
        
        with open(self.config_path, 'r') as f:
            stored_config = json.load(f)
        
        current_hash = self.get_config_hash()
        
        if stored_config['config_hash'] != current_hash:
            return "MISMATCH"
            
        return "OK"

def main():
    parser = argparse.ArgumentParser(description="Загрузка эмбеддингов в локальное FAISS хранилище")
    
    parser.add_argument("--input", required=True, help="Путь к JSONL файлу с эмбеддингами")
    parser.add_argument("--store_dir", default="./vector_store", help="Папка для хранения индекса Faiss")
    
    parser.add_argument("--dim", type=int, default=1024, help="Размерность вектора")
    parser.add_argument("--M", type=int, default=32, help="Параметр M для HNSW")
    parser.add_argument("--efConstruction", type=int, default=200, help="Параметр efConstruction для HNSW")
    parser.add_argument("--efSearch", type=int, default=100, help="Параметр efSearch для HNSW")
    
    parser.add_argument("--quantize", action="store_true", help="Включить скалярное квантование для экономии памяти")
    
    parser.add_argument("--rebuild", action="store_true", help="Принудительно пересоздать индекс (drop and reindex)")
    
    args = parser.parse_args()
    
    store = FaissVectorStore(
        store_dir=args.store_dir,
        dim=args.dim,
        M=args.M,
        efConstruction=args.efConstruction,
        efSearch=args.efSearch,
        use_quantization=args.quantize
    )
    
    if args.rebuild:
        print("Полное пересоздание индекса")
        store.reset()
        status = "MISSING"
    else:
        status = store.check_status()
        
        if status == "OK":
            print(f"Индекс существует и конфигурация актуальна. Загрузка не требуется. Используйте --rebuild для пересборки.")
            return
        elif status == "MISMATCH":
            print(f"Ошибка конфигурации! Индекс существует с параметрами, отличными от текущих.")
            print("Используйте --rebuild для обновления схемы (например, изменился dim или M).")
            return
        elif status == "MISSING":
            print("Индекс не найден. Создаем новый.")
    
    vectors, metadata = store.load_data(args.input)
    
    store.build_from_scratch(vectors)
    store.save(metadata)
    
    print("Успешно!")

if __name__ == "__main__":
    main()
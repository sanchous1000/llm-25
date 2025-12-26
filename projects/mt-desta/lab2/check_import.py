
import sys
import langchain
import langchain_community
import rank_bm25 # often needed for EnsembleRetriever

print(f"Python Executable: {sys.executable}")
print(f"LangChain Version: {langchain.__version__}")
print(f"LangChain Community Version: {langchain_community.__version__}")

try:
    from langchain.retrievers.ensemble import EnsembleRetriever
    print('Import 1 success: from langchain.retrievers import EnsembleRetriever')
except ImportError as e:
    print(f'Import 1 failed: {e}')

try:
    from langchain_community.retrievers import RRFEnsembleRetriever
    print('Import 2 success: from langchain_community.retrievers import EnsembleRetriever')
except ImportError as e:
    print(f'Import 2 failed: {e}')

try:
    from langchain.retrievers.ensemble import EnsembleRetriever
    print('Import 3 success: from langchain.retrievers.ensemble import EnsembleRetriever')
except ImportError as e:
    print(f'Import 3 failed: {e}')

"""
Check Langfuse connection and list datasets.
"""
from langfuse import Langfuse
import config

def check_connection():
    """Check Langfuse connection and list all datasets."""
    
    print(f"Connecting to Langfuse at {config.LANGFUSE_HOST}...")
    print(f"Using public key: {config.LANGFUSE_PUBLIC_KEY[:20]}...")
    
    langfuse = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST
    )
    
    # Get all datasets
    print("\nFetching datasets...")
    try:
        # Try to get dataset directly
        dataset = langfuse.get_dataset(config.DATASET_NAME)
        print(f"\n✓ Dataset '{config.DATASET_NAME}' found!")
        print(f"  Description: {dataset.description if hasattr(dataset, 'description') else 'N/A'}")
        
        # Get items count
        items = dataset.items
        print(f"  Items: {len(items) if items else 0}")
        
    except Exception as e:
        print(f"\n✗ Error getting dataset: {e}")
        print("\nTrying to fetch all available data...")
        
    langfuse.flush()
    print("\nConnection check completed!")

if __name__ == '__main__':
    check_connection()

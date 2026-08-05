import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve()))
import chromadb

client = chromadb.PersistentClient(path='data/chroma_db')
try:
    col = client.get_or_create_collection('manual_chunks')
    # try collection.count() if available
    count = None
    try:
        count = col.count()
    except Exception:
        # fallback: try to query with n_results=1 and read returned 'ids' length
        try:
            res = col.query(query_embeddings=[[0.0]*1536], n_results=1, include=['ids'])
            ids = res.get('ids', [])
            count = sum(len(batch) for batch in ids)
        except Exception:
            count = 'unknown'
    print('collection_count=', count)
except Exception as e:
    print('error checking chroma:', e)

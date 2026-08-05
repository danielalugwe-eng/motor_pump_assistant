import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve()))

import chromadb
import inspect

print('chromadb version:', chromadb.__version__)
client = chromadb.PersistentClient(path='data/chroma_db_debug')
print('client type:', type(client))
print('has get_or_create_collection:', hasattr(client, 'get_or_create_collection'))
print('collection methods:', [x for x in dir(client) if 'create' in x.lower() or 'collection' in x.lower()][:50])
print('get_or_create_collection sig:', inspect.signature(client.get_or_create_collection))

try:
    col = client.get_or_create_collection('test_noembed', embedding_function=None)
    print('collection created:', type(col))
    print('collection attrs contains embedding_function:', hasattr(col, 'embedding_function'))
    try:
        col.add(documents=['hello world'], metadatas=[{'a': 1}], ids=['doc1'])
        print('add succeeded without explicit embeddings')
    except Exception as e:
        print('add error:', type(e), e)
except Exception as e:
    print('collection creation error:', type(e), e)

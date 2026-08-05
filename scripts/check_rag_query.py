import json
import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve()))
from src.rag.search import search_manual

out = Path('scripts/rag_query_result.json')
try:
    results = search_manual('Where are the fuses located?', k=4)
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
except Exception as exc:
    out.write_text(json.dumps({'error': str(exc)}, indent=2), encoding='utf-8')
print('done')

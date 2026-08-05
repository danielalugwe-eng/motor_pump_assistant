import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve()))
from src.rag.ingest import ingest_manual

out = Path('scripts/ingest_result.txt')
try:
    count = ingest_manual('data')
    out.write_text(f'ingested_count={count}\n')
except Exception as e:
    out.write_text(f'error={e}\n')
print('done')

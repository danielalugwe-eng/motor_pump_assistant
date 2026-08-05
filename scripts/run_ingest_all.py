import sys
from pathlib import Path

# ensure project root is on sys.path for script execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag.ingest import ingest_manual


if __name__ == '__main__':
    count = ingest_manual('data/manuals')
    print('ingested_count=', count)

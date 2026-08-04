import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag.search import search_manual
from src.rag.llm import build_rag_context, route_query
from src.main import ask

print('cwd', os.getcwd())
print('root path', sys.path[0])
print('OPENAI_API_KEY', bool(os.getenv('OPENAI_API_KEY')))

query = 'How do I change the pump?'
print('route_query', route_query(query))

try:
    results = search_manual(query)
    print('search_manual ok', results.keys())
    context = build_rag_context(results)
    print('context length', len(context))
    print('context preview', context[:500].replace('\n', ' '))
except Exception as exc:
    print('search error', type(exc).__name__, exc)

try:
    class Q:
        def __init__(self, text):
            self.text = text

    response = ask(Q(query))
    print('ask response', response)
except Exception as exc:
    print('ask error', type(exc).__name__, exc)
    raise

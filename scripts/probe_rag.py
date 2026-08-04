from src.rag.search import search_manual
from src.rag.llm import build_rag_context

if __name__ == "__main__":
    query = "maintenance"
    results = search_manual(query, k=2)
    print("documents", len(results['documents'][0]))
    print(build_rag_context(results)[:1200])

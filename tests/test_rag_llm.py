from src.rag.llm import build_rag_context, route_query


def test_route_query_detects_predictive_requests():
    assert route_query("Will this pump break next week?") == "predictor"
    assert route_query("How do I maintain this pump?") == "rag"


def test_build_rag_context_joins_document_chunks():
    results = {"documents": [["First chunk", "Second chunk"]]}
    assert build_rag_context(results) == "First chunk\n\nSecond chunk"

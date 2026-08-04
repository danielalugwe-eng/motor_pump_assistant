import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv()

from src.rag.search import search_manual
from src.rag.llm import build_rag_context, build_rag_prompt, extract_answer_text
from openai import OpenAI

output_path = Path('terminal_llm_debug.txt')

with output_path.open('w', encoding='utf-8') as f:
    f.write(f'PWD={Path.cwd()}\n')
    f.write(f'OPENAI_API_KEY_SET={bool(os.getenv("OPENAI_API_KEY"))}\n')
    f.write('Starting search...\n')
    try:
        results = search_manual('How do I change the pump?', k=3)
        f.write(f'search results keys={list(results.keys())}\n')
        f.write(f'doc count={len(results.get("documents", [[]])[0]) if results.get("documents") else 0}\n')
        context = build_rag_context(results)
        f.write(f'context len={len(context)}\n')
        f.write(f'context preview={context[:500].replace("\n", " ")}\n')
    except Exception as exc:
        f.write(f'search error={type(exc).__name__}:{exc}\n')
        raise
    f.write('Starting prompt build...\n')
    prompt = build_rag_prompt('How do I change the pump?', context)
    f.write(f'prompt={prompt}\n')
    f.write('Creating OpenAI client...\n')
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    f.write(f'client has chat={hasattr(client, "chat")}\n')
    f.write(f'client has embeddings={hasattr(client, "embeddings")}\n')
    try:
        response = client.chat.completions.create(model='gpt-4o-mini', messages=prompt)
        f.write(f'response type={type(response)}\n')
        f.write(f'response repr={repr(response)}\n')
        answer = extract_answer_text(response)
        f.write(f'answer={answer}\n')
    except Exception as exc:
        f.write(f'llm error={type(exc).__name__}:{exc}\n')
        raise

import openai
from openai import OpenAI

with open('terminal_openai_debug.txt', 'w', encoding='utf-8') as f:
    f.write(f'openai={openai.__version__}\n')
    f.write(f'OpenAI={OpenAI}\n')
    f.write(f'has chat={hasattr(OpenAI, "chat")}\n')
    f.write(f'has responses={hasattr(OpenAI, "responses")}\n')
    f.write(f'has embeddings={hasattr(OpenAI, "embeddings")}\n')
    f.write(f'has completions={hasattr(OpenAI, "completions")}\n')

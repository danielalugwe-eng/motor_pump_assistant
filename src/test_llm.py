from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


def test_llm(prompt: str = "Say hello in one short sentence.") -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to the .env file first.")

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - diagnostics only
        raise RuntimeError(f"LLM request failed: {exc}") from exc


if __name__ == "__main__":
    try:
        print(test_llm())
    except Exception as exc:
        print(exc)

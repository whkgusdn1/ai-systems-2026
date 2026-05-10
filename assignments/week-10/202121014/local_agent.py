from __future__ import annotations

from openai import OpenAI


def ask_coder(task: str) -> str:
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-required",
    )
    response = client.chat.completions.create(
        model="deepseek-coder-v2",
        messages=[
            {"role": "system", "content": "You are an expert Python developer."},
            {"role": "user", "content": task},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def main() -> None:
    task = "Write a Python function that merges two sorted lists."
    print(ask_coder(task))


if __name__ == "__main__":
    main()


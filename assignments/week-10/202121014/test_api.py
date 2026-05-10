from __future__ import annotations

from openai import OpenAI


def main() -> None:
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-required",
    )

    response = client.chat.completions.create(
        model="deepseek-coder-v2",
        messages=[
            {"role": "system", "content": "You are an expert Python developer."},
            {
                "role": "user",
                "content": "Write a Python quicksort implementation with a short explanation.",
            },
        ],
        temperature=0.2,
        max_tokens=512,
    )

    message = response.choices[0].message.content if response.choices else ""
    print("=== Response ===")
    print(message)
    print("\n=== Usage ===")
    print(response.usage)


if __name__ == "__main__":
    main()


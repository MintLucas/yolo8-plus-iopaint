"""
测试 FastAPI + vLLM logprobs server

使用前先启动服务：
    bash start_server.sh

然后运行：
    python test_client.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import requests


def post(url: str, path: str, payload: dict) -> dict:
    resp = requests.post(f"{url.rstrip('/')}{path}", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("=== health ===")
    health = requests.get(f"{base_url}/health", timeout=10).json()
    print(json.dumps(health, indent=2, ensure_ascii=False))

    print("\n=== generate ===")
    gen = post(base_url, "/generate", {
        "prompt": "Question: What is the capital of France?\nAnswer:",
        "max_tokens": 16,
        "temperature": 0.0,
        "logprobs": 1,
    })
    print(json.dumps(gen, indent=2, ensure_ascii=False)[:2000])

    print("\n=== logprobs ===")
    lp = post(base_url, "/logprobs", {
        "prompt": "Question: What is the capital of France?\nAnswer:",
        "response": " Paris",
        "temperature": 0.0,
        "top_logprobs": 5,
    })
    print(json.dumps(lp, indent=2, ensure_ascii=False))

    print("\n=== compute_h_y_given_t ===")
    hy = post(base_url, "/compute_h_y_given_t", {
        "t_message": "France capital = Paris.",
        "y_answer": " Paris",
    })
    print(json.dumps(hy, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

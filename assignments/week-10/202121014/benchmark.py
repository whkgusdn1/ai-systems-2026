from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


BASE_URL = "http://localhost:8000/v1"
API_KEY = "not-required"
MODEL = "deepseek-coder-v2"
OUTPUT_PATH = Path(__file__).resolve().parent / "benchmark_results.json"
PROMPT = "Write a Python function that validates JSON input and returns parsed data."


@dataclass
class RequestMetrics:
    request_id: int
    success: bool
    ttft_ms: float | None
    total_latency_ms: float | None
    throughput_tps: float | None
    output_tokens: int
    error: str | None = None


@dataclass
class BenchmarkResult:
    concurrency: int
    total_requests: int
    completed_requests: int
    avg_ttft_ms: float | None
    p50_ttft_ms: float | None
    p99_ttft_ms: float | None
    avg_total_latency_ms: float | None
    avg_throughput_tps: float | None
    total_throughput_tps: float | None
    success_rate: float
    requests: list[RequestMetrics]


async def run_single_request(client: AsyncOpenAI, request_id: int) -> RequestMetrics:
    started = time.perf_counter()
    first_token_at: float | None = None
    output_chunks: list[str] = []

    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert Python developer."},
                {"role": "user", "content": PROMPT},
            ],
            max_tokens=256,
            temperature=0.1,
            stream=True,
        )

        async for chunk in stream:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                output_chunks.append(delta)

        finished = time.perf_counter()
        completion_text = "".join(output_chunks)
        output_tokens = len(completion_text.split())
        total_latency_ms = (finished - started) * 1000
        ttft_ms = ((first_token_at or finished) - started) * 1000
        throughput_tps = output_tokens / max(finished - (first_token_at or started), 1e-9)

        return RequestMetrics(
            request_id=request_id,
            success=True,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            throughput_tps=throughput_tps,
            output_tokens=output_tokens,
        )
    except Exception as exc:
        return RequestMetrics(
            request_id=request_id,
            success=False,
            ttft_ms=None,
            total_latency_ms=None,
            throughput_tps=None,
            output_tokens=0,
            error=str(exc),
        )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def summarize(concurrency: int, metrics: list[RequestMetrics]) -> BenchmarkResult:
    successes = [item for item in metrics if item.success]
    ttfts = [item.ttft_ms for item in successes if item.ttft_ms is not None]
    latencies = [item.total_latency_ms for item in successes if item.total_latency_ms is not None]
    throughputs = [item.throughput_tps for item in successes if item.throughput_tps is not None]

    total_tokens = sum(item.output_tokens for item in successes)
    total_seconds = sum((item.total_latency_ms or 0) / 1000 for item in successes)

    return BenchmarkResult(
        concurrency=concurrency,
        total_requests=len(metrics),
        completed_requests=len(successes),
        avg_ttft_ms=statistics.fmean(ttfts) if ttfts else None,
        p50_ttft_ms=percentile(ttfts, 50),
        p99_ttft_ms=percentile(ttfts, 99),
        avg_total_latency_ms=statistics.fmean(latencies) if latencies else None,
        avg_throughput_tps=statistics.fmean(throughputs) if throughputs else None,
        total_throughput_tps=(total_tokens / total_seconds) if total_seconds > 0 else None,
        success_rate=(len(successes) / len(metrics)) if metrics else 0.0,
        requests=metrics,
    )


async def run_benchmark_for_concurrency(concurrency: int) -> BenchmarkResult:
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    tasks = [run_single_request(client, request_id=index + 1) for index in range(concurrency)]
    metrics = await asyncio.gather(*tasks)
    return summarize(concurrency, metrics)


def to_export(result: BenchmarkResult) -> dict[str, Any]:
    data = asdict(result)
    data["requests"] = [asdict(item) for item in result.requests]
    return data


async def main() -> None:
    benchmark_payload: dict[str, Any] = {
        "note": "If this file contains sample or failure-heavy data, rerun on the DGX server after vLLM is up.",
        "results": {},
    }

    for concurrency in (1, 4, 8):
        result = await run_benchmark_for_concurrency(concurrency)
        benchmark_payload["results"][str(concurrency)] = to_export(result)

        print(f"[concurrency={concurrency}]")
        print(f"  success_rate={result.success_rate:.2%}")
        print(f"  avg_ttft_ms={result.avg_ttft_ms}")
        print(f"  p50_ttft_ms={result.p50_ttft_ms}")
        print(f"  p99_ttft_ms={result.p99_ttft_ms}")
        print(f"  avg_total_latency_ms={result.avg_total_latency_ms}")
        print(f"  avg_throughput_tps={result.avg_throughput_tps}")
        print(f"  total_throughput_tps={result.total_throughput_tps}")

    OUTPUT_PATH.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())


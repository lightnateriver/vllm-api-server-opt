#!/usr/bin/env python3

import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from bundle_paths import DATA_DIR, HOST, MODEL_DIR, PROBE_FILE, RESULTS_DIR, apply_request_model

RESULT_TAG = os.environ.get(
    "VLLM_BENCHMARK_TAG",
    f"phase{os.environ.get('VLLM_ASCEND_API_OPT_PHASE', 'unknown')}",
)
MAX_COMPLETION_TOKENS_OVERRIDE = os.environ.get(
    "VLLM_BENCHMARK_MAX_COMPLETION_TOKENS"
)
WARMUP_ROUNDS = list(range(3))
TEST_ROUNDS = list(range(3, 13))

TARGET_STAGE_ORDER = {
    "api_server": [
        "MediaConnector.fetch_image_async",
        "AsyncMicrobatchTokenizer.encode",
        "ProcessorInputs.get_mm_hashes",
        "Qwen3VLMultiModalProcessor._call_hf_processor",
        "ShmObjectStoreSenderCache.get_and_update_items_with_callback",
        "SingleWriterShmObjectStorage.batch_copy_to_buffer",
        "phase3_local_meta_url_to_path",
        "phase3_local_meta_realpath",
        "phase3_local_meta_stat",
        "phase3_local_meta_uuid",
        "phase3_local_meta_image_size",
        "phase3_local_meta_total",
    ],
    "engine_core": [
        "engine_core_time",
        "engine_core_decode_only_time",
    ],
    "tp_worker": [
        "Qwen3VLMultiModalProcessor._call_hf_processor",
        "ImageMediaIO.load_file",
        "MsgpackSerde.deserialize",
        "tp_worker_execute_model_decode_only",
    ],
}


@dataclass
class ProbeSlice:
    start_offset: int
    end_offset: int
    lines: list[dict[str, Any]]


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def load_payload(round_id: int) -> dict[str, Any]:
    with open(DATA_DIR / f"round_{round_id}" / "payload.json", encoding="utf-8") as fp:
        payload = json.load(fp)
    payload = apply_request_model(payload)
    if MAX_COMPLETION_TOKENS_OVERRIDE:
        payload["max_completion_tokens"] = int(MAX_COMPLETION_TOKENS_OVERRIDE)
    return payload


def round_payload_path(round_id: int) -> Path:
    return DATA_DIR / f"round_{round_id}" / "payload.json"


def read_probe_slice(start_offset: int) -> ProbeSlice:
    if not PROBE_FILE.exists():
        return ProbeSlice(start_offset=start_offset, end_offset=start_offset, lines=[])

    with open(PROBE_FILE, "r", encoding="utf-8") as fp:
        fp.seek(start_offset)
        chunk = fp.read()
        end_offset = fp.tell()

    lines = []
    for raw in chunk.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    return ProbeSlice(start_offset=start_offset, end_offset=end_offset, lines=lines)


def flatten_probe(lines: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    totals = {
        component: {stage: 0.0 for stage in stages}
        for component, stages in TARGET_STAGE_ORDER.items()
    }
    for line in lines:
        component = line.get("component")
        stage = line.get("stage")
        elapsed_ms = float(line.get("elapsed_ms", 0.0))
        if not component or not stage:
            continue
        if component not in totals:
            continue
        if stage not in totals[component]:
            continue
        totals[component][stage] += elapsed_ms
    return totals


def stream_request(
    payload: dict[str, Any],
    round_id: int,
) -> tuple[dict[str, Any], str, float, float, float, float]:
    start = time.perf_counter()
    first_token_at = None
    second_token_at = None
    last_token_at = None
    final_response = None
    content_parts: list[str] = []

    try:
        with requests.post(
            f"{HOST}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=1800,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"round_{round_id} received invalid SSE json: {data_str[:240]}"
                    ) from exc
                final_response = data
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    content_parts.append(content)
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    if second_token_at is None:
                        partial_text = "".join(content_parts)
                        partial_tokens = len(
                            get_tokenizer().encode(
                                partial_text,
                                add_special_tokens=False,
                            )
                        )
                        if partial_tokens >= 2:
                            second_token_at = now
                    last_token_at = now
    except requests.HTTPError as exc:
        response = exc.response
        status = response.status_code if response is not None else "unknown"
        body = ""
        if response is not None:
            body = response.text[:400].replace("\n", "\\n")
        raise RuntimeError(
            f"round_{round_id} request failed with status={status} body={body}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"round_{round_id} request failed: {exc}") from exc

    end = time.perf_counter()
    if first_token_at is None:
        first_token_at = end
    if second_token_at is None:
        second_token_at = last_token_at if last_token_at is not None else end
    if last_token_at is None:
        last_token_at = end
    return (
        final_response or {},
        "".join(content_parts),
        start,
        first_token_at,
        second_token_at,
        last_token_at,
    )


def run_round(round_id: int, probe_offset: int) -> tuple[dict[str, Any], int]:
    payload_path = round_payload_path(round_id)
    if not payload_path.exists():
        raise FileNotFoundError(f"round_{round_id} payload not found: {payload_path}")
    payload = load_payload(round_id)
    start_offset = probe_offset
    (
        response,
        completion_text,
        start,
        first_token_at,
        second_token_at,
        last_token_at,
    ) = stream_request(payload, round_id)
    end = time.perf_counter()
    probe_slice = read_probe_slice(start_offset)
    usage = response.get("usage") or {}
    completion_tokens = len(
        get_tokenizer().encode(completion_text, add_special_tokens=False)
    )
    probe_totals = flatten_probe(probe_slice.lines)
    first_to_second_gap_ms = max(second_token_at - first_token_at, 0.0) * 1000.0
    steady_token_denominator = max(completion_tokens - 2, 1)

    result = {
        "round": round_id,
        "payload_path": str(payload_path),
        "e2e_ms": (end - start) * 1000.0,
        "ttft_ms": (first_token_at - start) * 1000.0,
        "ttst_ms": (second_token_at - start) * 1000.0,
        "first_to_second_token_ms": first_to_second_gap_ms,
        "tpot_ms": (
            ((last_token_at - first_token_at) * 1000.0) / max(completion_tokens - 1, 1)
        ),
        "steady_tpot_ms": (
            ((last_token_at - second_token_at) * 1000.0) / steady_token_denominator
        ),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "completion_text_chars": len(completion_text),
        "probe_stage_ms": probe_totals,
    }
    return result, probe_slice.end_offset


def aggregate_stage_metrics(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for component, stages in TARGET_STAGE_ORDER.items():
        summary[component] = {}
        for stage in stages:
            values = [
                float(result["probe_stage_ms"].get(component, {}).get(stage, 0.0))
                for result in results
            ]
            summary[component][stage] = summarize(values)
    return summary


def print_phase(label: str, results: list[dict[str, Any]]) -> None:
    print(f"\n{label}")
    for result in results:
        print(
            f"round={result['round']} "
            f"e2e={result['e2e_ms']:.2f}ms "
            f"ttft={result['ttft_ms']:.2f}ms "
            f"ttst={result['ttst_ms']:.2f}ms "
            f"tpot={result['tpot_ms']:.4f}ms/token "
            f"steady_tpot={result['steady_tpot_ms']:.4f}ms/token "
            f"prompt_tokens={result['prompt_tokens']} "
            f"completion_tokens={result['completion_tokens']}"
        )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    probe_offset = 0
    if PROBE_FILE.exists():
        probe_offset = PROBE_FILE.stat().st_size

    warmup_results = []
    for round_id in WARMUP_ROUNDS:
        result, probe_offset = run_round(round_id, probe_offset)
        warmup_results.append(result)

    test_results = []
    for round_id in TEST_ROUNDS:
        result, probe_offset = run_round(round_id, probe_offset)
        test_results.append(result)

    print_phase("Warmup", warmup_results)
    print_phase("Test", test_results)

    summary = {
        "ttft_ms": summarize([r["ttft_ms"] for r in test_results]),
        "ttst_ms": summarize([r["ttst_ms"] for r in test_results]),
        "first_to_second_token_ms": summarize(
            [r["first_to_second_token_ms"] for r in test_results]
        ),
        "tpot_ms": summarize([r["tpot_ms"] for r in test_results]),
        "steady_tpot_ms": summarize([r["steady_tpot_ms"] for r in test_results]),
        "e2e_ms": summarize([r["e2e_ms"] for r in test_results]),
        "module_stage_ms": aggregate_stage_metrics(test_results),
    }

    print("\nSummary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    output = {
        "host": HOST,
        "probe_file": str(PROBE_FILE),
        "warmup_rounds": WARMUP_ROUNDS,
        "test_rounds": TEST_ROUNDS,
        "warmup_results": warmup_results,
        "test_results": test_results,
        "summary": summary,
    }
    output_path = RESULTS_DIR / f"multimodal_baseline_results_{RESULT_TAG}.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)
    print(f"\nresults_file={output_path}")


if __name__ == "__main__":
    main()

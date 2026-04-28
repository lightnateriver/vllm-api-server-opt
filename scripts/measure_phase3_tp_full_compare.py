#!/usr/bin/env python3

import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from bundle_paths import DATA_DIR, HOST, MODEL_DIR, PROBE_FILE, RESULTS_DIR, apply_request_model

RESULT_TAG = os.environ.get("VLLM_BENCHMARK_TAG", "phase3_tp_full_compare")
WARMUP_ROUNDS = list(range(3))
TEST_ROUNDS = list(range(3, 13))

WORKER_STAGES = [
    "tp_worker_local_mm_prepare_total",
    "tp_worker_local_mm_assign",
    "tp_worker_local_mm_load_images",
    "tp_worker_local_mm_parse_data",
    "tp_worker_local_mm_hf_process",
    "tp_worker_local_mm_build_kwargs",
    "tp_worker_local_mm_rebuild",
    "tp_worker_vit_inference",
]

ENGINE_STAGES = [
    "engine_core_time",
    "engine_core_decode_only_time",
]


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
    payload["stream"] = True
    return payload


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


def stream_request(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str, float, float, float, float]:
    start = time.perf_counter()
    first_token_at = None
    second_token_at = None
    last_token_at = None
    final_response = None
    content_parts: list[str] = []

    with requests.post(
        f"{HOST}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=1800,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data_str = raw_line[6:]
            if data_str == "[DONE]":
                break

            data = json.loads(data_str)
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


def stage_records_by_name(
    lines: list[dict[str, Any]],
    *,
    component: str,
    stage_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    result = {stage: [] for stage in stage_names}
    for line in lines:
        if line.get("component") != component:
            continue
        stage = line.get("stage")
        if stage in result:
            result[stage].append(line)
    return result


def analyze_worker_stage(stage_name: str, stage_records: list[dict[str, Any]]) -> dict[str, float]:
    elapsed_all = [float(item.get("elapsed_ms", 0.0)) for item in stage_records]

    local_image_counts = []
    active_elapsed = []
    active_tp_ranks = set()
    all_tp_ranks = set()
    for item in stage_records:
        extra = item.get("extra") or {}
        tp_rank = extra.get("tp_rank")
        if tp_rank is not None:
            all_tp_ranks.add(int(tp_rank))

        local_image_count = extra.get("local_image_count")
        if local_image_count is None:
            continue
        local_image_count = int(local_image_count)
        if local_image_count > 0:
            local_image_counts.append(float(local_image_count))
            active_elapsed.append(float(item.get("elapsed_ms", 0.0)))
            if tp_rank is not None:
                active_tp_ranks.add(int(tp_rank))

    result = {
        "cluster_sum_ms": sum(elapsed_all),
        "all_rank_record_count": float(len(elapsed_all)),
        "all_rank_mean_ms": statistics.mean(elapsed_all) if elapsed_all else 0.0,
        "unique_tp_ranks": float(len(all_tp_ranks)),
    }
    if local_image_counts:
        result.update(
            {
                "active_rank_record_count": float(len(active_elapsed)),
                "active_rank_mean_ms": statistics.mean(active_elapsed),
                "active_rank_max_ms": max(active_elapsed),
                "active_rank_local_image_mean": statistics.mean(local_image_counts),
                "active_rank_local_image_max": max(local_image_counts),
                "active_tp_ranks": float(len(active_tp_ranks)),
            }
        )
    return result


def analyze_engine_stage(stage_records: list[dict[str, Any]]) -> dict[str, float]:
    elapsed_all = [float(item.get("elapsed_ms", 0.0)) for item in stage_records]
    return {
        "cluster_sum_ms": sum(elapsed_all),
        "record_count": float(len(elapsed_all)),
        "mean_ms": statistics.mean(elapsed_all) if elapsed_all else 0.0,
        "max_ms": max(elapsed_all) if elapsed_all else 0.0,
    }


def run_round(round_id: int, probe_offset: int) -> tuple[dict[str, Any], int]:
    payload = load_payload(round_id)
    (
        response,
        completion_text,
        start,
        first_token_at,
        second_token_at,
        last_token_at,
    ) = stream_request(payload)
    end = time.perf_counter()

    time.sleep(0.8)
    probe_slice = read_probe_slice(probe_offset)
    usage = response.get("usage") or {}
    completion_tokens = len(
        get_tokenizer().encode(completion_text, add_special_tokens=False)
    )
    worker_records = stage_records_by_name(
        probe_slice.lines,
        component="tp_worker",
        stage_names=WORKER_STAGES,
    )
    engine_records = stage_records_by_name(
        probe_slice.lines,
        component="engine_core",
        stage_names=ENGINE_STAGES,
    )
    first_to_second_gap_ms = max(second_token_at - first_token_at, 0.0) * 1000.0
    steady_token_denominator = max(completion_tokens - 2, 1)

    result = {
        "round": round_id,
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
        "worker_stage_analysis": {
            stage: analyze_worker_stage(stage, records)
            for stage, records in worker_records.items()
        },
        "engine_stage_analysis": {
            stage: analyze_engine_stage(records)
            for stage, records in engine_records.items()
        },
    }
    return result, probe_slice.end_offset


def aggregate_worker_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    worker_metrics = [
        "cluster_sum_ms",
        "all_rank_mean_ms",
        "active_rank_mean_ms",
        "active_rank_max_ms",
        "active_rank_local_image_mean",
        "active_rank_local_image_max",
        "active_rank_record_count",
        "active_tp_ranks",
        "unique_tp_ranks",
    ]
    for stage in WORKER_STAGES:
        summary[stage] = {}
        for metric in worker_metrics:
            values = [
                float(result["worker_stage_analysis"][stage].get(metric, 0.0))
                for result in results
            ]
            summary[stage][metric] = summarize(values)
    return summary


def aggregate_engine_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    engine_metrics = [
        "cluster_sum_ms",
        "mean_ms",
        "max_ms",
        "record_count",
    ]
    for stage in ENGINE_STAGES:
        summary[stage] = {}
        for metric in engine_metrics:
            values = [
                float(result["engine_stage_analysis"][stage].get(metric, 0.0))
                for result in results
            ]
            summary[stage][metric] = summarize(values)
    return summary


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

    summary = {
        "ttft_ms": summarize([r["ttft_ms"] for r in test_results]),
        "ttst_ms": summarize([r["ttst_ms"] for r in test_results]),
        "first_to_second_token_ms": summarize(
            [r["first_to_second_token_ms"] for r in test_results]
        ),
        "tpot_ms": summarize([r["tpot_ms"] for r in test_results]),
        "steady_tpot_ms": summarize([r["steady_tpot_ms"] for r in test_results]),
        "e2e_ms": summarize([r["e2e_ms"] for r in test_results]),
        "worker_stage_metrics": aggregate_worker_metrics(test_results),
        "engine_stage_metrics": aggregate_engine_metrics(test_results),
    }

    output = {
        "host": HOST,
        "probe_file": str(PROBE_FILE),
        "warmup_rounds": WARMUP_ROUNDS,
        "test_rounds": TEST_ROUNDS,
        "warmup_results": warmup_results,
        "test_results": test_results,
        "summary": summary,
    }

    output_path = RESULTS_DIR / f"phase3_tp_full_compare_{RESULT_TAG}.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results_file={output_path}")


if __name__ == "__main__":
    main()

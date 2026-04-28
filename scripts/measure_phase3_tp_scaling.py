#!/usr/bin/env python3

import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import requests

from bundle_paths import DATA_DIR, HOST, PROBE_FILE, RESULTS_DIR, apply_request_model

RESULT_TAG = os.environ.get("VLLM_BENCHMARK_TAG", "phase3_tp_scaling")
WARMUP_ROUNDS = list(range(3))
TEST_ROUNDS = list(range(3, 13))

STAGES = [
    "tp_worker_local_mm_prepare_total",
    "tp_worker_local_mm_assign",
    "tp_worker_local_mm_load_images",
    "tp_worker_local_mm_parse_data",
    "tp_worker_local_mm_hf_process",
    "tp_worker_local_mm_build_kwargs",
    "tp_worker_local_mm_rebuild",
    "tp_worker_vit_inference",
]


@dataclass
class ProbeSlice:
    start_offset: int
    end_offset: int
    lines: list[dict[str, Any]]


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
    payload["stream"] = False
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


def extract_stage_records(lines: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {stage: [] for stage in STAGES}
    for line in lines:
        if line.get("component") != "tp_worker":
            continue
        stage = line.get("stage")
        if stage not in result:
            continue
        result[stage].append(line)
    return result


def analyze_stage_records(stage_records: list[dict[str, Any]]) -> dict[str, float]:
    elapsed_all = [float(item.get("elapsed_ms", 0.0)) for item in stage_records]

    active_records = []
    active_image_counts = []
    tp_ranks = set()
    active_tp_ranks = set()
    for item in stage_records:
        extra = item.get("extra") or {}
        tp_rank = extra.get("tp_rank")
        if tp_rank is not None:
            tp_ranks.add(int(tp_rank))
        local_image_count = extra.get("local_image_count")
        if local_image_count is None:
            continue
        local_image_count = int(local_image_count)
        if local_image_count > 0:
            active_records.append(float(item.get("elapsed_ms", 0.0)))
            active_image_counts.append(float(local_image_count))
            if tp_rank is not None:
                active_tp_ranks.add(int(tp_rank))

    return {
        "cluster_sum_ms": sum(elapsed_all),
        "all_rank_record_count": float(len(elapsed_all)),
        "all_rank_mean_ms": statistics.mean(elapsed_all) if elapsed_all else 0.0,
        "active_rank_record_count": float(len(active_records)),
        "active_rank_mean_ms": (
            statistics.mean(active_records) if active_records else 0.0
        ),
        "active_rank_max_ms": max(active_records) if active_records else 0.0,
        "active_rank_local_image_mean": (
            statistics.mean(active_image_counts) if active_image_counts else 0.0
        ),
        "active_rank_local_image_max": (
            max(active_image_counts) if active_image_counts else 0.0
        ),
        "unique_tp_ranks": float(len(tp_ranks)),
        "active_tp_ranks": float(len(active_tp_ranks)),
    }


def post_json(payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = requests.post(
        f"{HOST}/v1/chat/completions",
        json=payload,
        timeout=1800,
    )
    response.raise_for_status()
    e2e_ms = (time.perf_counter() - start) * 1000.0
    return response.json(), e2e_ms


def run_round(round_id: int, probe_offset: int) -> tuple[dict[str, Any], int]:
    payload = load_payload(round_id)
    response, e2e_ms = post_json(payload)
    time.sleep(0.8)
    probe_slice = read_probe_slice(probe_offset)
    usage = response.get("usage") or {}
    stage_records = extract_stage_records(probe_slice.lines)
    stage_analysis = {
        stage: analyze_stage_records(records)
        for stage, records in stage_records.items()
    }
    result = {
        "round": round_id,
        "e2e_ms": e2e_ms,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "tp_worker_stage_analysis": stage_analysis,
    }
    return result, probe_slice.end_offset


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    stage_summary: dict[str, dict[str, dict[str, float]]] = {}
    metrics = [
        "cluster_sum_ms",
        "all_rank_mean_ms",
        "active_rank_mean_ms",
        "active_rank_max_ms",
        "active_rank_local_image_mean",
        "active_rank_local_image_max",
        "active_rank_record_count",
        "active_tp_ranks",
    ]
    for stage in STAGES:
        stage_summary[stage] = {}
        for metric in metrics:
            values = [
                float(result["tp_worker_stage_analysis"][stage].get(metric, 0.0))
                for result in results
            ]
            stage_summary[stage][metric] = summarize(values)
    return {
        "e2e_ms": summarize([float(result["e2e_ms"]) for result in results]),
        "stage_metrics": stage_summary,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if PROBE_FILE.exists():
        PROBE_FILE.unlink()

    probe_offset = 0

    warmup_results = []
    for round_id in WARMUP_ROUNDS:
        result, probe_offset = run_round(round_id, probe_offset)
        warmup_results.append(result)

    test_results = []
    for round_id in TEST_ROUNDS:
        result, probe_offset = run_round(round_id, probe_offset)
        test_results.append(result)

    summary = aggregate(test_results)

    output = {
        "host": HOST,
        "probe_file": str(PROBE_FILE),
        "warmup_rounds": WARMUP_ROUNDS,
        "test_rounds": TEST_ROUNDS,
        "warmup_results": warmup_results,
        "test_results": test_results,
        "summary": summary,
    }
    output_path = RESULTS_DIR / f"phase3_tp_scaling_{RESULT_TAG}.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results_file={output_path}")


if __name__ == "__main__":
    main()

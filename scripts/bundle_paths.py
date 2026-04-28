#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BUNDLE_DIR = ROOT_DIR / "bundle"
DOCS_DIR = ROOT_DIR / "docs"
DATA_DIR = Path(
    os.environ.get("VLLM_BENCHMARK_DATA_DIR", str(ROOT_DIR / "data"))
).resolve()
RESULTS_DIR = Path(
    os.environ.get("VLLM_BENCHMARK_RESULTS_DIR", str(ROOT_DIR / "results"))
).resolve()
TEST_IMAGES_DIR = Path(
    os.environ.get("VLLM_BENCHMARK_TEST_IMAGES_DIR", str(DATA_DIR / "test_images"))
).resolve()
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.environ.get("VLLM_BENCHMARK_MODEL_DIR", "/mnt/sfs_turbo/models/Qwen3.5-4B"),
)
REQUEST_MODEL = os.environ.get("VLLM_REQUEST_MODEL", MODEL_DIR)
HOST = os.environ.get("VLLM_BENCHMARK_HOST", os.environ.get("VLLM_HOST", "http://127.0.0.1:8000"))
PROBE_FILE = Path(
    os.environ.get("VLLM_ASCEND_PERF_PROBE_FILE", "/tmp/vllm_ascend_perf_probe.jsonl")
)


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def round_payload_path(round_id: int) -> Path:
    return DATA_DIR / f"round_{round_id}" / "payload.json"


def load_payload(round_id: int) -> dict[str, Any]:
    payload = json.loads(round_payload_path(round_id).read_text(encoding="utf-8"))
    return apply_request_model(payload)


def apply_request_model(payload: dict[str, Any]) -> dict[str, Any]:
    if REQUEST_MODEL:
        payload["model"] = REQUEST_MODEL
    return payload


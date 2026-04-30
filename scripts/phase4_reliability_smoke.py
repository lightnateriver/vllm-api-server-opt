#!/usr/bin/env python3
"""Quick smoke tests for phase4 cache reliability hardening."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B"
DEFAULT_IMAGE = ROOT_DIR / "data" / "skill_l0" / "pics" / "720x1280" / "jpg" / "circle.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick phase4 reliability smoke tests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--http-port", type=int, default=9010)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--image-path", default=str(DEFAULT_IMAGE))
    parser.add_argument("--cache-max-gb", type=float, default=0.00005)
    parser.add_argument("--server-start-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--reuse-existing-server", action="store_true")
    parser.add_argument("--trace-path", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--keep-artifacts", action="store_true")
    return parser.parse_args()


def wait_http_ready(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Timeout waiting for {url}: {last_error!r}")


def start_static_server(http_port: int, image_path: Path, work_dir: Path) -> subprocess.Popen:
    root = work_dir / "http_root"
    root.mkdir(parents=True, exist_ok=True)
    image_dir = image_path.parent
    for source in sorted(image_dir.glob("*.jpg")):
        target = root / source.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
    cmd = [
        "python3",
        "-m",
        "http.server",
        str(http_port),
        "--directory",
        str(root),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_phase4_server(args: argparse.Namespace, work_dir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_DIR": args.model_dir,
            "HOST": args.host,
            "PORT": str(args.port),
            "ALLOWED_LOCAL_MEDIA_PATH": str(ROOT_DIR / "data"),
            "VLLM_ASCEND_HTTP_CACHE_DIR": str(work_dir / "cache"),
            "VLLM_ASCEND_HTTP_CACHE_TTL_S": "86400",
            "VLLM_ASCEND_HTTP_TIMEOUT_S": "5",
            "VLLM_ASCEND_HTTP_MAX_FILE_BYTES": str(64 * 1024 * 1024),
            "VLLM_ASCEND_HTTP_CACHE_MAX_GB": str(args.cache_max_gb),
            "VLLM_ASCEND_MM_FILE_MAP": str(work_dir / "mm_trace.jsonl"),
            "VLLM_ASCEND_PHASE2_VALIDATE_FILE": str(work_dir / "validate.jsonl"),
            "VLLM_ASCEND_PERF_PROBE": "0",
            "TENSOR_PARALLEL_SIZE": str(args.tensor_parallel_size),
            "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
            "MAX_MODEL_LEN": str(args.max_model_len),
        }
    )
    cmd = ["/bin/bash", str(ROOT_DIR / "bundle" / "start_phase4_server.sh")]
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=(work_dir / "server.stdout.log").open("w", encoding="utf-8"),
        stderr=(work_dir / "server.stderr.log").open("w", encoding="utf-8"),
        start_new_session=True,
    )


def stop_process_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def api_request(base_url: str, image_url: str, *, timeout_s: float) -> dict:
    import requests

    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with exactly one word: hello"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_completion_tokens": 8,
        "temperature": 0,
    }
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def load_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def new_trace_records(path: Path, start_idx: int) -> tuple[list[dict], int]:
    records = load_trace(path)
    return records[start_idx:], len(records)


def find_cache_entry(cache_dir: Path, image_url: str) -> tuple[Path | None, dict | None]:
    import hashlib

    key = hashlib.sha1(image_url.encode("utf-8")).hexdigest()
    meta_path = cache_dir / f"{key}.json"
    if not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    local_path = Path(meta["local_path"])
    return local_path, meta


def scenario_bad_cache(
    *,
    base_url: str,
    image_url: str,
    cache_dir: Path,
    trace_path: Path,
    timeout_s: float,
    trace_idx: int,
) -> tuple[dict, int]:
    api_request(base_url, image_url, timeout_s=timeout_s)
    local_path, _meta = find_cache_entry(cache_dir, image_url)
    if local_path is None or not local_path.exists():
        raise RuntimeError("Failed to populate cache for bad-cache scenario.")
    local_path.write_bytes(b"not-an-image")
    api_request(base_url, image_url, timeout_s=timeout_s)
    api_request(base_url, image_url, timeout_s=timeout_s)
    records, next_idx = new_trace_records(trace_path, trace_idx)
    invalid = [r for r in records if r.get("stage") == "phase4_http_cache_invalid"]
    fallback = [
        r
        for r in records
        if r.get("stage") == "phase4_http_fallback_stock"
        and r.get("fallback_reason") == "cache_invalid_local_ref"
    ]
    refilled = [
        r
        for r in records
        if r.get("stage") == "phase4_http_cache_fill"
        and r.get("canonical_http_url") == image_url
    ]
    return {
        "scenario": "bad_cache_self_heal",
        "passed": bool(invalid and fallback and refilled),
        "invalid_count": len(invalid),
        "fallback_count": len(fallback),
        "fill_count": len(refilled),
    }, next_idx


def scenario_capacity_gc(
    *,
    base_url: str,
    cache_dir: Path,
    trace_path: Path,
    timeout_s: float,
    http_port: int,
    trace_idx: int,
) -> tuple[dict, int]:
    image_names = ["circle.jpg", "cube.jpg", "cylinder.jpg", "rectangle.jpg"]
    for name in image_names:
        api_request(base_url, f"http://127.0.0.1:{http_port}/{name}", timeout_s=timeout_s)
    records, next_idx = new_trace_records(trace_path, trace_idx)
    gc_records = [r for r in records if r.get("stage") == "phase4_http_cache_gc"]
    evict_records = [r for r in records if r.get("stage") == "phase4_http_cache_evict"]
    cache_files = list(cache_dir.glob("*.json"))
    return {
        "scenario": "capacity_gc",
        "passed": bool(gc_records and evict_records),
        "gc_count": len(gc_records),
        "evict_count": len(evict_records),
        "remaining_meta_files": len(cache_files),
    }, next_idx


def scenario_fallback_observable(
    *,
    base_url: str,
    trace_path: Path,
    timeout_s: float,
    http_port: int,
    trace_idx: int,
) -> tuple[dict, int]:
    missing_url = f"http://127.0.0.1:{http_port}/missing-does-not-exist.jpg"
    try:
        api_request(base_url, missing_url, timeout_s=timeout_s)
        request_status = "success"
    except Exception as exc:  # noqa: BLE001
        request_status = type(exc).__name__
    records, next_idx = new_trace_records(trace_path, trace_idx)
    error_records = [r for r in records if r.get("stage") == "phase4_http_cache_error"]
    fallback = [
        r
        for r in records
        if r.get("stage") == "phase4_http_fallback_stock"
        and r.get("fallback_reason") == "materialize_failed"
    ]
    return {
        "scenario": "fallback_observable",
        "passed": bool(error_records and fallback),
        "request_status": request_status,
        "error_count": len(error_records),
        "fallback_count": len(fallback),
    }, next_idx


def main() -> int:
    args = parse_args()
    work_dir = Path("/tmp") / f"phase4_reliability_smoke_{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    static_proc = None
    server_proc = None
    results = []
    try:
        static_proc = start_static_server(args.http_port, image_path, work_dir)
        wait_http_ready(f"http://127.0.0.1:{args.http_port}/", 20.0)

        if args.reuse_existing_server:
            wait_http_ready(f"http://{args.host}:{args.port}/v1/models", args.server_start_timeout)
            trace_path = Path(args.trace_path) if args.trace_path else Path("/tmp/vllm_ascend_mm_file_map.jsonl")
            cache_dir = Path(args.cache_dir) if args.cache_dir else Path("/tmp/vllm_ascend_http_cache")
        else:
            server_proc = start_phase4_server(args, work_dir)
            wait_http_ready(f"http://{args.host}:{args.port}/v1/models", args.server_start_timeout)
            trace_path = work_dir / "mm_trace.jsonl"
            cache_dir = work_dir / "cache"

        base_url = f"http://{args.host}:{args.port}"
        image_url = f"http://127.0.0.1:{args.http_port}/{image_path.name}"

        trace_idx = 0
        scenario, trace_idx = scenario_bad_cache(
            base_url=base_url,
            image_url=image_url,
            cache_dir=cache_dir,
            trace_path=trace_path,
            timeout_s=args.request_timeout,
            trace_idx=trace_idx,
        )
        results.append(scenario)

        scenario, trace_idx = scenario_capacity_gc(
            base_url=base_url,
            cache_dir=cache_dir,
            trace_path=trace_path,
            timeout_s=args.request_timeout,
            http_port=args.http_port,
            trace_idx=trace_idx,
        )
        results.append(scenario)

        scenario, trace_idx = scenario_fallback_observable(
            base_url=base_url,
            trace_path=trace_path,
            timeout_s=args.request_timeout,
            http_port=args.http_port,
            trace_idx=trace_idx,
        )
        results.append(scenario)

        summary = {
            "work_dir": str(work_dir),
            "trace_path": str(trace_path),
            "cache_dir": str(cache_dir),
            "reuse_existing_server": args.reuse_existing_server,
            "passed": all(item.get("passed") for item in results),
            "results": results,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["passed"] else 1
    finally:
        stop_process_tree(server_proc)
        stop_process_tree(static_proc)
        if not args.keep_artifacts:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

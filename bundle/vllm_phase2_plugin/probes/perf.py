import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable


_LOCK = threading.Lock()
_ENABLE_ENV = "VLLM_ASCEND_PERF_PROBE"
_FILE_ENV = "VLLM_ASCEND_PERF_PROBE_FILE"
_DEFAULT_FILE = "/tmp/vllm_ascend_perf_probe.jsonl"


def perf_probe_enabled() -> bool:
    value = os.environ.get(_ENABLE_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _probe_file() -> str:
    return os.environ.get(_FILE_ENV, _DEFAULT_FILE)


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
    return repr(value)


def perf_probe_record(
    component: str,
    stage: str,
    elapsed_s: float,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    if not perf_probe_enabled():
        return

    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "component": component,
        "stage": stage,
        "elapsed_ms": elapsed_s * 1000.0,
    }
    if extra:
        record["extra"] = _make_json_safe(extra)

    line = json.dumps(record, ensure_ascii=True, sort_keys=True)
    with _LOCK:
        with open(_probe_file(), "a", encoding="utf-8") as fp:
            fp.write(line)
            fp.write("\n")


@contextmanager
def perf_probe_span(
    component: str,
    stage: str,
    *,
    extra: dict[str, Any] | Callable[[], dict[str, Any] | None] | None = None,
):
    if not perf_probe_enabled():
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        resolved_extra = extra() if callable(extra) else extra
        perf_probe_record(
            component,
            stage,
            time.perf_counter() - start,
            extra=resolved_extra,
        )


@contextmanager
def perf_probe_multi_span(
    component: str,
    stages: Iterable[str],
    *,
    extra: dict[str, Any] | Callable[[], dict[str, Any] | None] | None = None,
):
    if not perf_probe_enabled():
        yield
        return

    stage_names = tuple(stages)
    start = time.perf_counter()
    try:
        yield
    finally:
        resolved_extra = extra() if callable(extra) else extra
        elapsed_s = time.perf_counter() - start
        for stage in stage_names:
            perf_probe_record(component, stage, elapsed_s, extra=resolved_extra)

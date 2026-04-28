import os


def phase_value() -> int:
    phase = os.environ.get("VLLM_ASCEND_API_OPT_PHASE", "0").strip() or "0"
    try:
        return int(phase)
    except ValueError:
        lowered = phase.lower()
        if lowered in {"on", "true", "yes"}:
            return 1
        if lowered.startswith("phase") and lowered[5:].isdigit():
            return int(lowered[5:])
        return 0


def phase_at_least(target: int) -> bool:
    return phase_value() >= target


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

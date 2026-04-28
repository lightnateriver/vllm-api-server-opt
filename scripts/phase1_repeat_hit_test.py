import json
import os
import statistics
import time
from pathlib import Path
from urllib import request

from bundle_paths import DATA_DIR, PROBE_FILE, REQUEST_MODEL


API_URL = os.environ.get("PHASE1_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
BASE_PAYLOAD = Path(
    os.environ.get(
        "PHASE1_BASE_PAYLOAD",
        str(DATA_DIR / "round_3" / "payload.json"),
    )
)
OUTPUT_PAYLOAD = Path(
    os.environ.get(
        "PHASE1_REPEAT_PAYLOAD",
        str(DATA_DIR / "phase1_repeat_payload.json"),
    )
)


def build_repeat_payload() -> dict:
    payload = json.loads(BASE_PAYLOAD.read_text())
    payload["model"] = REQUEST_MODEL
    content = payload["messages"][0]["content"]

    first_image_url = None
    for part in content:
        if part.get("type") == "image_url":
            image_url = part.get("image_url")
            first_image_url = image_url["url"] if isinstance(image_url, dict) else image_url
            break

    if not first_image_url:
        raise RuntimeError("No image_url found in base payload")

    for part in content:
        if part.get("type") == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url["url"] = first_image_url
            else:
                part["image_url"] = first_image_url

    payload["stream"] = False
    OUTPUT_PAYLOAD.write_text(json.dumps(payload, ensure_ascii=False))
    return payload


def post_json(payload: dict) -> dict:
    req = request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=600) as resp:
        body = resp.read()
    return json.loads(body)


def summarize_probe(records: list[dict]) -> dict[str, float]:
    by_stage: dict[str, list[float]] = {}
    for record in records:
        if record.get("component") != "api_server":
            continue
        stage = record["stage"]
        by_stage.setdefault(stage, []).append(float(record["elapsed_ms"]))

    summary: dict[str, float] = {}
    for stage, values in by_stage.items():
        summary[f"{stage}.sum_ms"] = round(sum(values), 3)
        summary[f"{stage}.mean_ms"] = round(statistics.mean(values), 3)
        summary[f"{stage}.count"] = float(len(values))
    return summary


def run_once(payload: dict) -> tuple[dict, dict[str, float], float]:
    if PROBE_FILE.exists():
        PROBE_FILE.unlink()

    start = time.perf_counter()
    response = post_json(payload)
    e2e_ms = (time.perf_counter() - start) * 1000.0

    records = []
    if PROBE_FILE.exists():
        records = [json.loads(line) for line in PROBE_FILE.read_text().splitlines() if line]

    return response, summarize_probe(records), e2e_ms


def main() -> None:
    payload = build_repeat_payload()

    first_response, first_probe, first_e2e = run_once(payload)
    second_response, second_probe, second_e2e = run_once(payload)

    result = {
        "payload_file": str(OUTPUT_PAYLOAD),
        "first": {
            "e2e_ms": round(first_e2e, 3),
            "probe": first_probe,
            "response_preview": first_response["choices"][0]["message"]["content"][:120],
        },
        "second": {
            "e2e_ms": round(second_e2e, 3),
            "probe": second_probe,
            "response_preview": second_response["choices"][0]["message"]["content"][:120],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

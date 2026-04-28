#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from bundle_paths import DATA_DIR, HOST, apply_request_model


def stream_request(host: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    content_parts: list[str] = []
    final_response: dict[str, Any] = {}

    with requests.post(
        f"{host}/v1/chat/completions",
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

    return "".join(content_parts), final_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--rounds", type=int, default=13)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    results = []

    for round_id in range(args.rounds):
        payload_path = data_dir / f"round_{round_id}" / "payload.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload = apply_request_model(payload)
        text, response = stream_request(args.host, payload)
        results.append(
            {
                "round": round_id,
                "payload": str(payload_path),
                "text": text,
                "response": response,
            }
        )
        preview = text[:80].replace("\n", "\\n")
        print(f"round={round_id} chars={len(text)} preview={preview}")

    output = {
        "label": args.label,
        "host": args.host,
        "rounds": args.rounds,
        "results": results,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output_file={out_path}")


if __name__ == "__main__":
    main()

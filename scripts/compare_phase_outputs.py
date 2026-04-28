#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)
    left = load_results(left_path)
    right = load_results(right_path)

    left_results = left.get("results") or []
    right_results = right.get("results") or []
    round_count = min(len(left_results), len(right_results))

    mismatches = []
    for idx in range(round_count):
        left_item = left_results[idx]
        right_item = right_results[idx]
        left_text = left_item.get("text", "")
        right_text = right_item.get("text", "")
        if left_text != right_text:
            mismatches.append(
                {
                    "round": idx,
                    "left_chars": len(left_text),
                    "right_chars": len(right_text),
                    "left_preview": left_text[:160],
                    "right_preview": right_text[:160],
                }
            )

    output = {
        "left": str(left_path),
        "right": str(right_path),
        "left_label": left.get("label"),
        "right_label": right.get("label"),
        "left_rounds": len(left_results),
        "right_rounds": len(right_results),
        "compared_rounds": round_count,
        "all_equal": not mismatches and len(left_results) == len(right_results),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }

    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    print(rendered)

    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from bundle_paths import DATA_DIR, MODEL_DIR, REQUEST_MODEL

OUTPUT_DIR = DATA_DIR
NUM_IMAGES = 40
NUM_ROUNDS = 13
TARGET_TEXT_TOKENS = 10000
MAX_COMPLETION_TOKENS = 128


def build_sentence(round_id: int, sentence_id: int) -> str:
    code = f"r{round_id:02d}_s{sentence_id:05d}"
    adjectives = [
        "granular",
        "deterministic",
        "cacheless",
        "multimodal",
        "latency-aware",
        "token-precise",
        "vision-heavy",
        "nonrepeating",
    ]
    nouns = [
        "benchmark",
        "request",
        "payload",
        "profile",
        "timeline",
        "dataset",
        "session",
        "sample",
    ]
    adj = adjectives[sentence_id % len(adjectives)]
    noun = nouns[(sentence_id * 3) % len(nouns)]
    numbers = [str(round_id), str(sentence_id), str(round_id * 1000 + sentence_id)]
    return (
        f"Segment {code} records a {adj} {noun} for performance tracing. "
        f"It includes identifiers {'/'.join(numbers)} and unique markers "
        f"{code.upper()}::{sentence_id * 17 + round_id}. "
        f"This line is intentionally unique within and across rounds.\n"
    )


def fit_tail_exact(
    tokenizer: AutoTokenizer,
    prefix: str,
    round_id: int,
    sentence_id: int,
    target_tokens: int,
) -> str:
    current = len(tokenizer.encode(prefix, add_special_tokens=False))
    if current == target_tokens:
        return prefix

    def candidate_texts(seed: int) -> list[str]:
        return [
            f" tail{round_id:02d}_{sentence_id:05d}_{seed:05d}",
            f" item-{round_id:02d}-{sentence_id:05d}-{seed:05d}",
            f" note[{round_id:02d}:{sentence_id:05d}:{seed:05d}]",
            f" ref{seed:05d}",
            f" z{seed:05d}",
            f"\nextra_{round_id:02d}_{sentence_id:05d}_{seed:05d}",
            f" code={round_id * 100000 + sentence_id * 97 + seed}",
            f" flag{seed:05d}.",
        ]

    remaining = target_tokens - current
    stack: list[tuple[str, int, int]] = [(prefix, remaining, 1)]
    text = None

    while stack:
        base_text, base_remaining, seed = stack.pop()
        if base_remaining == 0:
            text = base_text
            break
        if base_remaining < 0 or seed > 4096:
            continue

        current_tokens = len(tokenizer.encode(base_text, add_special_tokens=False))
        options: list[tuple[int, str]] = []
        for fragment in candidate_texts(seed):
            merged = base_text + fragment
            merged_tokens = len(tokenizer.encode(merged, add_special_tokens=False))
            delta = merged_tokens - current_tokens
            if 0 < delta <= base_remaining:
                options.append((delta, merged))

        options.sort(key=lambda item: (-item[0], len(item[1])))
        stack.append((base_text, base_remaining, seed + 1))
        for delta, merged in reversed(options):
            stack.append((merged, base_remaining - delta, seed + 1))

    if text is None:
        raise RuntimeError(
            f"failed to fit exact token count: round={round_id} no solution for "
            f"remaining={remaining}"
        )

    final_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if final_tokens != target_tokens:
        raise RuntimeError(
            f"failed to fit exact token count: round={round_id} got={final_tokens} "
            f"target={target_tokens}"
        )
    return text


def build_text(
    tokenizer: AutoTokenizer,
    round_id: int,
    target_tokens: int = TARGET_TEXT_TOKENS,
) -> tuple[str, int]:
    random.seed(1000 + round_id)
    text = ""
    sentence_id = 0
    while True:
        candidate = text + build_sentence(round_id, sentence_id)
        candidate_tokens = len(tokenizer.encode(candidate, add_special_tokens=False))
        if candidate_tokens > target_tokens:
            text = fit_tail_exact(
                tokenizer,
                text,
                round_id,
                sentence_id,
                target_tokens,
            )
            break
        text = candidate
        sentence_id += 1

    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count != target_tokens:
        raise RuntimeError(
            f"unexpected token count for round {round_id}: {token_count}"
        )
    return text, token_count


def build_payload(
    round_dir: Path,
    text: str,
    num_images: int = NUM_IMAGES,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
) -> dict:
    image_paths = sorted((round_dir / "images").glob("*.png"))
    if len(image_paths) != num_images:
        raise RuntimeError(
            f"{round_dir} expected {num_images} images, found {len(image_paths)}"
        )

    content = [{"type": "text", "text": text}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_path.resolve().as_uri()},
            }
        )

    return {
        "model": REQUEST_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    for round_id in range(NUM_ROUNDS):
        round_dir = OUTPUT_DIR / f"round_{round_id}"
        payload_path = round_dir / "payload.json"
        text, token_count = build_text(tokenizer, round_id)
        payload = build_payload(round_dir, text)
        with open(payload_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"round_{round_id}: text_tokens={token_count} images={NUM_IMAGES}")


if __name__ == "__main__":
    main()

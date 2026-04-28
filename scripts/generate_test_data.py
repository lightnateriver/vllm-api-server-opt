#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import AutoTokenizer

from bundle_paths import DATA_DIR, MODEL_DIR, REQUEST_MODEL
from generate_data import (
    MAX_COMPLETION_TOKENS,
    NUM_IMAGES as DEFAULT_NUM_IMAGES,
    NUM_ROUNDS as DEFAULT_NUM_ROUNDS,
    TARGET_TEXT_TOKENS as DEFAULT_TARGET_TEXT_TOKENS,
    build_payload,
    build_text,
)


DEFAULT_IMAGE_SIZE = (288, 512)
DEFAULT_WARMUP_ROUNDS = 3
DEFAULT_SEED = 20260405


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a portable 13-round multimodal benchmark dataset with "
            "10k text tokens and 40 local file:// images per round."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument("--warmup-rounds", type=int, default=DEFAULT_WARMUP_ROUNDS)
    parser.add_argument("--images-per-round", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument(
        "--target-text-tokens",
        type=int,
        default=DEFAULT_TARGET_TEXT_TOKENS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rounds <= 0:
        raise ValueError("--rounds must be greater than 0")
    if args.warmup_rounds < 0 or args.warmup_rounds >= args.rounds:
        raise ValueError("--warmup-rounds must be in [0, rounds)")
    if args.images_per_round <= 0:
        raise ValueError("--images-per-round must be greater than 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be greater than 0")
    if args.target_text_tokens <= 0:
        raise ValueError("--target-text-tokens must be greater than 0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_text_uniqueness(text: str, round_id: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != len(set(lines)):
        raise RuntimeError(f"round_{round_id} generated duplicate text lines")
    return lines


def generate_unique_image(
    path: Path,
    size: tuple[int, int],
    round_id: int,
    image_id: int,
    seed: int,
) -> None:
    width, height = size
    image = Image.new("RGB", size)
    pixels = image.load()

    global_index = round_id * 1000 + image_id
    seed_mix = seed + round_id * 7919 + image_id * 1543

    for y in range(height):
        row_term = (seed_mix + y * (round_id + 11)) % 256
        for x in range(width):
            pixels[x, y] = (
                (x * 5 + y * 3 + row_term + global_index) % 256,
                (x * 7 + y * 11 + seed_mix // 7 + image_id * 17) % 256,
                ((x ^ y) + seed_mix // 13 + round_id * 19 + image_id * 29) % 256,
            )

    draw = ImageDraw.Draw(image)
    border_color = (
        (global_index * 53) % 256,
        (global_index * 97) % 256,
        (global_index * 149) % 256,
    )
    draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=3)

    step = max(width // 6, 24)
    for column in range(0, width, step):
        line_color = (
            (column + round_id * 31) % 256,
            (column * 3 + image_id * 47) % 256,
            (column * 5 + seed_mix) % 256,
        )
        draw.line((column, 0, width - 1 - (column % width), height - 1), fill=line_color, width=2)

    label = f"R{round_id:02d}-I{image_id:02d}"
    label_y = 12 + (round_id % 5) * 18
    draw.text((12, label_y), label, fill=(255, 255, 255))

    marker_bytes = global_index.to_bytes(4, byteorder="little", signed=False)
    for offset, value in enumerate(marker_bytes):
        pixels[offset, 0] = (
            value,
            (value * 53 + round_id) % 256,
            (value * 97 + image_id) % 256,
        )
    pixels[4, 0] = (round_id, image_id, (round_id + image_id) % 256)

    image.save(path, "PNG")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    seen_texts: set[str] = set()
    seen_image_hashes: dict[str, str] = {}
    round_summaries: list[dict[str, object]] = []

    for round_id in range(args.rounds):
        round_dir = output_dir / f"round_{round_id}"
        image_dir = round_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        text, token_count = build_text(
            tokenizer,
            round_id,
            target_tokens=args.target_text_tokens,
        )
        text_lines = validate_text_uniqueness(text, round_id)
        if text in seen_texts:
            raise RuntimeError(f"round_{round_id} duplicated a previous round text")
        seen_texts.add(text)

        image_paths: list[Path] = []
        image_hashes: list[str] = []
        for image_id in range(args.images_per_round):
            image_path = image_dir / f"img_{image_id:02d}.png"
            generate_unique_image(
                image_path,
                (args.width, args.height),
                round_id,
                image_id,
                args.seed,
            )
            image_hash = sha256_file(image_path)
            if image_hash in seen_image_hashes:
                raise RuntimeError(
                    "duplicate image bytes detected between "
                    f"{seen_image_hashes[image_hash]} and {image_path.resolve()}"
                )
            seen_image_hashes[image_hash] = str(image_path.resolve())
            image_paths.append(image_path.resolve())
            image_hashes.append(image_hash)

        payload = build_payload(
            round_dir,
            text,
            num_images=args.images_per_round,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        payload_path = round_dir / "payload.json"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        round_summaries.append(
            {
                "round": round_id,
                "text_tokens": token_count,
                "text_line_count": len(text_lines),
                "image_count": len(image_paths),
                "payload_path": str(payload_path.resolve()),
                "first_image": str(image_paths[0]),
                "last_image": str(image_paths[-1]),
                "image_sha256": image_hashes,
            }
        )
        print(
            f"round_{round_id}: text_tokens={token_count} "
            f"images={len(image_paths)} payload={payload_path.resolve()}"
        )

    manifest = {
        "model_dir": MODEL_DIR,
        "request_model": REQUEST_MODEL,
        "output_dir": str(output_dir),
        "rounds": args.rounds,
        "warmup_rounds": list(range(args.warmup_rounds)),
        "test_rounds": list(range(args.warmup_rounds, args.rounds)),
        "target_text_tokens": args.target_text_tokens,
        "images_per_round": args.images_per_round,
        "image_size": {"width": args.width, "height": args.height},
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "seed": args.seed,
        "unique_text_count": len(seen_texts),
        "unique_image_count": len(seen_image_hashes),
        "round_summaries": round_summaries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "done: "
        f"rounds={args.rounds} warmup={args.warmup_rounds} "
        f"test={args.rounds - args.warmup_rounds} "
        f"text_tokens={args.target_text_tokens} images_per_round={args.images_per_round}"
    )
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare and run the phase4 precision matrix for local_path/http/base64."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results" / "phase4_precision"
L0_ROOT = ROOT_DIR / "data" / "skill_l0"

SKILL_ROOT = Path("/root/.codex/skills/vllm-multimodal-precision-testing")
SKILL_SCRIPTS = SKILL_ROOT / "scripts"
SKILL_DATASET_DIR = SKILL_ROOT / "multi-pics-datasets" / "cases"

L0_SCRIPT = SKILL_SCRIPTS / "l0_multimodal_smoke.py"
L05_SCRIPT = SKILL_SCRIPTS / "multi_pics_eval.py"
MME_SCRIPT = SKILL_SCRIPTS / "mme_eval_local.py"

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_HOST = "http://127.0.0.1:8000"
DEFAULT_MODEL = "/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B"
DEFAULT_MME_TSV = "/tmp/MME.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run the 0428 phase4 precision test matrix."
    )
    parser.add_argument(
        "--transports",
        nargs="+",
        choices=["local_path", "http", "base64"],
        default=["local_path", "http", "base64"],
        help="Which media transports to include.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=["l0", "l05", "mme"],
        default=["l0", "l05", "mme"],
        help="Which test suites to include.",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mme-tsv", default=DEFAULT_MME_TSV)
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_DIR),
        help="Where to write phase4 precision artifacts.",
    )
    parser.add_argument(
        "--http-base-url",
        default="http://127.0.0.1:9000",
        help="Base URL for the local static server in http mode.",
    )
    parser.add_argument(
        "--http-root",
        default="",
        help="Optional prepared static root for http mode. Defaults under results-root.",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the commands now. Without this flag the script only prints the matrix.",
    )
    parser.add_argument(
        "--wait-ready",
        action="store_true",
        help="Pass readiness gating to the L0.5 evaluator when executing.",
    )
    parser.add_argument(
        "--mme-concurrency",
        type=int,
        default=16,
        help="MME concurrency override.",
    )
    return parser.parse_args()


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def ensure_symlink(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to overwrite existing path: {target}")
    target.symlink_to(source)


def prepare_http_root(http_root: Path) -> Path:
    ensure_symlink(http_root / "l0", L0_ROOT)
    ensure_symlink(http_root / "multi_pics" / "cases", SKILL_DATASET_DIR)
    (http_root / "mme").mkdir(parents=True, exist_ok=True)
    return http_root


def build_suite_command(
    *,
    suite: str,
    transport: str,
    endpoint: str,
    host: str,
    model: str,
    mme_tsv: str,
    http_base_url: str,
    http_root: Path,
    results_root: Path,
    wait_ready: bool,
    mme_concurrency: int,
) -> tuple[list[str], Path, str]:
    transport_dir = results_root / transport

    if suite == "l0":
        output_path = transport_dir / "l0_summary.json"
        if transport == "http":
            image_dir = (http_root / "l0" / "pics" / "720x1280" / "jpg").resolve()
            video_path = (http_root / "l0" / "video" / "720x1280" / "mp4" / "shapes.mp4").resolve()
            media_root = L0_ROOT.resolve()
        else:
            image_dir = L0_ROOT / "pics" / "720x1280" / "jpg"
            video_path = L0_ROOT / "video" / "720x1280" / "mp4" / "shapes.mp4"
            media_root = None
        cmd = [
            sys.executable,
            str(L0_SCRIPT),
            "--host",
            host,
            "--model",
            model,
            "--image-dir",
            str(image_dir),
            "--video-path",
            str(video_path),
            "--media-mode",
            transport,
            "--json",
        ]
        if transport == "http":
            cmd.extend(
                [
                    "--media-root",
                    str(media_root),
                    "--media-base-url",
                    http_base_url.rstrip("/") + "/l0",
                ]
            )
        return cmd, output_path, "json_stdout"

    if suite == "l05":
        output_dir = transport_dir / "l05"
        if transport == "http":
            dataset_dir = (http_root / "multi_pics" / "cases").resolve()
            media_root = SKILL_DATASET_DIR.resolve()
        else:
            dataset_dir = SKILL_DATASET_DIR
            media_root = None
        cmd = [
            sys.executable,
            str(L05_SCRIPT),
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--endpoint",
            endpoint,
            "--model",
            model,
            "--media-mode",
            transport,
            "--json",
        ]
        if wait_ready:
            cmd.append("--wait-ready")
        if transport == "http":
            cmd.extend(
                [
                    "--media-root",
                    str(media_root),
                    "--media-base-url",
                    http_base_url.rstrip("/") + "/multi_pics/cases",
                ]
            )
        return cmd, output_dir / "summary.json", "output_dir"

    if suite == "mme":
        out_prefix = transport_dir / "mme" / "mme_phase4"
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(MME_SCRIPT),
            "--tsv",
            mme_tsv,
            "--endpoint",
            endpoint,
            "--model",
            model,
            "--concurrency",
            str(mme_concurrency),
            "--out-prefix",
            str(out_prefix),
            "--media-mode",
            transport,
        ]
        if transport == "http":
            cmd.extend(
                [
                    "--media-root",
                    str(http_root / "mme"),
                    "--media-base-url",
                    http_base_url.rstrip("/") + "/mme",
                ]
            )
        elif transport == "local_path":
            cmd.extend(
                [
                    "--media-root",
                    str(out_prefix.parent / "_media_cache"),
                ]
            )
        return cmd, out_prefix.with_suffix(".score.csv"), "mme_prefix"

    raise ValueError(f"Unsupported suite: {suite}")


def build_matrix(args: argparse.Namespace) -> list[dict[str, str]]:
    results_root = Path(args.results_root).resolve()
    http_root = Path(args.http_root).resolve() if args.http_root else (results_root / "http_root")
    if "http" in args.transports:
        prepare_http_root(http_root)

    matrix = []
    for transport in args.transports:
        for suite in args.suites:
            cmd, expected_output, output_kind = build_suite_command(
                suite=suite,
                transport=transport,
                endpoint=args.endpoint,
                host=args.host,
                model=args.model,
                mme_tsv=args.mme_tsv,
                http_base_url=args.http_base_url,
                http_root=http_root,
                results_root=results_root,
                wait_ready=args.wait_ready,
                mme_concurrency=args.mme_concurrency,
            )
            matrix.append(
                {
                    "suite": suite,
                    "transport": transport,
                    "command": quote_command(cmd),
                    "expected_output": str(expected_output),
                    "output_kind": output_kind,
                }
            )
    return matrix


def write_stdout_artifact(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def run_matrix(matrix: list[dict[str, str]]) -> None:
    for item in matrix:
        command = item["command"]
        print(f"[RUN] {item['transport']} {item['suite']}")
        print(command)
        completed = subprocess.run(
            shlex.split(command),
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if item["output_kind"] == "json_stdout" and completed.stdout:
            write_stdout_artifact(Path(item["expected_output"]), completed.stdout)
        elif item["output_kind"] == "mme_prefix" and completed.stdout:
            summary_path = Path(item["expected_output"]).with_name("mme_phase4_summary.json")
            write_stdout_artifact(summary_path, completed.stdout)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    matrix = build_matrix(args)

    if args.print_commands or not args.execute:
        print(json.dumps(matrix, ensure_ascii=False, indent=2))

    if args.execute:
        run_matrix(matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

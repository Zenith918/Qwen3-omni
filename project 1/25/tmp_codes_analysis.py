#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from typing import Iterable, Optional

import torch


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_codes(tag: str, dump_dir: str):
    path = os.path.join(dump_dir, f"codes_{tag}.pt")
    codes = torch.load(path, map_location="cpu")
    if isinstance(codes, torch.Tensor):
        return codes.cpu().numpy()
    return codes


def _first_diff_frame(a, b) -> int:
    if a is None or b is None:
        return -1
    frames = min(a.shape[0], b.shape[0])
    if frames <= 0:
        return -1
    a = a[:frames]
    b = b[:frames]
    if a.ndim == 1:
        diff = a != b
        return int(diff.argmax()) if diff.any() else -1
    diff = (a != b).any(axis=1)
    return int(diff.argmax()) if diff.any() else -1


def _collect_tags(runs_json: Optional[str], tags: Optional[Iterable[str]]) -> list[str]:
    out: list[str] = []
    if runs_json:
        with open(runs_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data.get("rows", []):
            tag = row.get("tag", "")
            if tag:
                out.append(tag)
    if tags:
        out.extend([t for t in tags if t])
    seen = set()
    uniq = []
    for t in out:
        if t in seen:
            continue
        uniq.append(t)
        seen.add(t)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-json", default="")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--dump-dir", default="/workspace/project 1/25/output/code_dumps")
    parser.add_argument("--packet-tokens", type=int, default=4)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    tags = _collect_tags(args.runs_json, args.tags)
    full_hashes = []
    first_packet_hashes = []
    codes_cache = {}
    for tag in tags:
        codes = _load_codes(tag, args.dump_dir)
        codes_cache[tag] = codes
        full_hashes.append(_sha256_bytes(codes.tobytes()))
        first = codes[: max(0, args.packet_tokens)]
        first_packet_hashes.append(_sha256_bytes(first.tobytes()))

    first_diff = None
    if args.compare and len(tags) >= 2:
        a = codes_cache.get(tags[0])
        b = codes_cache.get(tags[1])
        first_diff = _first_diff_frame(a, b)

    result = {
        "count": len(tags),
        "packet_tokens": args.packet_tokens,
        "full_hash_unique": len(set(full_hashes)),
        "first_packet_hash_unique": len(set(first_packet_hashes)),
        "first_diff_frame": first_diff,
        "tags": tags,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

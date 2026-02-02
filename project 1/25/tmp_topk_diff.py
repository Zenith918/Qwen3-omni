#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict


LINE_RE = re.compile(
    r"\[CODEGEN_TOPK\]\s+req_id=(?P<req_id>\d+)\s+step=(?P<step>\d+)\s+"
    r"top1=(?P<top1_id>-?\d+)\s+(?P<top1_val>-?\d+\.\d+)\s+"
    r"top2=(?P<top2_id>-?\d+)\s+(?P<top2_val>-?\d+\.\d+)\s+"
    r"gap=(?P<gap>-?\d+\.\d+)\s+seed=(?P<seed>-?\d+)\s+"
    r"text_hash=(?P<text_hash>[0-9a-f]*)"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--req-ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    by_req = defaultdict(dict)
    meta = {}
    order = []

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            req_id = m.group("req_id")
            step = int(m.group("step"))
            if req_id not in meta:
                meta[req_id] = {
                    "seed": m.group("seed"),
                    "text_hash": m.group("text_hash"),
                }
                order.append(req_id)
            by_req[req_id][step] = {
                "top1_id": int(m.group("top1_id")),
                "top1_val": float(m.group("top1_val")),
                "top2_id": int(m.group("top2_id")),
                "top2_val": float(m.group("top2_val")),
                "gap": float(m.group("gap")),
            }

    req_ids = args.req_ids or order[-args.limit :]
    if len(req_ids) < 2:
        print("Need at least 2 req_ids. Found:", req_ids)
        return 1

    a, b = req_ids[0], req_ids[1]
    steps = sorted(set(by_req[a].keys()) & set(by_req[b].keys()))
    first_diff = None
    for step in steps:
        va = by_req[a][step]
        vb = by_req[b][step]
        if va != vb:
            first_diff = step
            break

    print("req_ids:", req_ids)
    print("meta:", {rid: meta.get(rid, {}) for rid in req_ids})
    print("first_diff_step:", first_diff)
    if first_diff is not None:
        print("A:", by_req[a][first_diff])
        print("B:", by_req[b][first_diff])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import hashlib
import os
import sys
import time

import torch


def _make_req(text: str):
    import tts_server as ts

    req = ts.TTSRequest(
        text=text,
        task_type="Base",
        language="Chinese",
        speaker="serena",
        instruct="",
        max_new_tokens=2048,
        non_streaming_mode=False,
    )
    return {
        "text": text,
        "speaker": req.speaker,
        "language": req.language,
        "instruct": req.instruct,
        "max_new_tokens": req.max_new_tokens,
        "non_streaming_mode": req.non_streaming_mode,
        "gen_kwargs": ts._deep_gen_kwargs(req),
        "seed": ts._stable_seed_from_text(req),
    }


def _run_once(worker, payload):
    worker.drain_out()
    req_id = worker.next_request_id()
    worker.send_generate(req_id, payload)
    codes_tensor = None
    while True:
        msg = worker.read_out(timeout=1.0)
        if msg is None:
            continue
        if msg.get("request_id") != req_id:
            continue
        if msg.get("type") == "error":
            raise RuntimeError(msg.get("error"))
        if msg.get("type") == "done":
            codes_list = msg.get("codes") or []
            codes_tensor = codes_list[0] if codes_list else None
            break
    if codes_tensor is None:
        raise RuntimeError("no codes")
    codes_np = codes_tensor.detach().cpu().numpy()
    sha256 = hashlib.sha256(codes_np.tobytes()).hexdigest()
    return sha256, codes_tensor.detach().cpu()


def main() -> int:
    os.environ.setdefault("TTS_DEEP_STREAM_CODEGEN_DEVICE", "cuda:0")
    os.environ.setdefault("TTS_DEEP_STREAM_MODEL_DIR", "/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    os.environ.setdefault("TTS_DEEP_STREAM_DETERMINISTIC", "1")
    os.environ.setdefault("TTS_DEEP_STREAM_DETERMINISTIC_POLICY", "seeded")
    os.environ.setdefault("TTS_DEEP_STREAM_SEED_MODE", "content")

    sys.path.append("/workspace/project 1/25/clients")
    import tts_server as ts

    text = (
        "如果系统在长句播放时出现越来越慢的情况，请先记录当时的时间戳、首包时间以及整体音频时长，并把日志打包。"
        "随后尝试相同文本重复三次，观察 RTF 是否呈线性上升，这能帮助我们定位是解码还是拼接导致的问题，同时也便于对比修复效果。"
    )
    payload = _make_req(text)
    worker = ts._DeepCodeWorker(ts.TTS_DEEP_STREAM_MODEL_DIR, ts.TTS_DEEP_STREAM_CODEGEN_DEVICE)
    try:
        sha1, c1 = _run_once(worker, payload)
        sha2, c2 = _run_once(worker, payload)
    finally:
        worker.shutdown(force=True)

    print("sha1", sha1)
    print("sha2", sha2)
    print("equal", sha1 == sha2)
    diff = (c1 != c2)
    count = int(diff.sum().item())
    print("diff_count", count)
    if count > 0:
        first = diff.nonzero()[0].tolist()
        print("first_diff", first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

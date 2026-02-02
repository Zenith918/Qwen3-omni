#!/usr/bin/env python3
import os
import sys
import multiprocessing as mp


def _collect_env(tag: str):
    import torch
    import tts_server as ts

    ts._set_global_seed(
        ts.TTS_DEEP_STREAM_SEED,
        strict=ts.TTS_DEEP_STREAM_CODEGEN_STRICT,
        strict_hard=ts.TTS_DEEP_STREAM_CODEGEN_STRICT_HARD,
        soft=ts.TTS_DEEP_STREAM_DETERMINISTIC_SOFT,
        single_thread=ts.TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD,
    )

    return {
        "tag": tag,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "default_dtype": str(torch.get_default_dtype()),
    }


def _worker(q):
    data = _collect_env("worker")
    q.put(data)


def main() -> int:
    sys.path.append("/workspace/project 1/25/clients")
    import tts_server as ts  # noqa: F401

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q,))
    p.start()
    main_data = _collect_env("main")
    worker_data = q.get()
    p.join(timeout=10)

    print("MAIN", main_data)
    print("WORKER", worker_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import json
import sys

import torch


def _summarize_cudnn_ops():
    if not torch.cuda.is_available():
        return {"error": "cuda_not_available"}

    device = torch.device("cuda:0")
    x = torch.randn(1, 64, 128, device=device)
    w = torch.randn(128, 64, 3, device=device)
    wt = torch.randn(64, 32, 3, device=device)

    if not torch.backends.cudnn.is_available():
        return {"error": "cudnn_not_available"}

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = torch.nn.functional.conv1d(x, w, padding=1)
        y2 = torch.nn.functional.conv_transpose1d(y, wt, padding=1)
        torch.cuda.synchronize()
        _ = y2.sum().item()

    names = [evt.key for evt in prof.key_averages()]
    cudnn_conv = [n for n in names if "cudnn_convolution" in n]
    cudnn_conv_t = [n for n in names if "cudnn_convolution_transpose" in n]
    return {
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        "cuda_matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "cudnn_convolution_events": cudnn_conv,
        "cudnn_convolution_transpose_events": cudnn_conv_t,
    }


def main() -> int:
    result = _summarize_cudnn_ops()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if "error" in result:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

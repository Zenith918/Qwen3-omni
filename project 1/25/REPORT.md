# Qwen3 主线切换进度报告

## 目标
- 主链路：upstream vLLM（非 Omni）作为 Thinker 引擎
- 语音：Qwen3-TTS 独立服务
- vLLM-Omni + AWQ 仅保留实验分支

## A) 冻结 Omni 分支
- 本地 Omni hack patch：`/workspace/project 1/25/artifacts/omni_hack.patch`
- 说明：vLLM-Omni + AWQ 在 L40S 输出异常（0000/....），主线已切回 vLLM

## B) LLM Server（vLLM OpenAI）
- 启动脚本：`/workspace/project 1/25/scripts/run_llm_server.sh`
- 模型路径：`/workspace/models/Qwen3-Omni-AWQ-4bit`
- 关键参数：
  - `max_model_len=2048`
  - `gpu_memory_utilization=0.6`
  - `quantization=compressed-tensors`
  - `kv_cache_dtype=fp8`
- 验收结果：
  - `GET /v1/models` 成功
  - `/v1/chat/completions` text-only 有语义
  - `stream=true` 有持续 delta

## C) Qwen3-TTS
- TTS 服务脚本：`/workspace/project 1/25/scripts/run_tts_server.sh`
- Smoke test：`/workspace/project 1/25/clients/tts_smoke_test.py`
- 模型默认：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- 低显存配置：`/workspace/project 1/25/artifacts/qwen3_tts_l40s.yaml`
- 验收结果：本地服务可稳定输出 WAV（chunked 响应）

## D) 桥接 Demo
- 脚本：`/workspace/project 1/25/scripts/run_demo_bridge.sh`
- 说明：LLM streaming 文本切分后逐段调用 TTS
- 验收结果：输出 6 段音频文件，首段音频生成可用

## E) 指标（待补全）
- LLM（`llm_smoke_test.py`）
  - TTFT: **0.055s**
  - tokens/s: **170.77**
  - tokens: **197**
  - peak VRAM: **约 32.9 GiB / 46.1 GiB**（运行中 nvidia-smi）
- TTS（`tts_smoke_test.py`）
  - 首包延迟: **2.265s**
  - 总耗时: **2.332s**
  - 音频时长: **2.457s**
  - RTF: **0.949**
  - 头信息：`x-gen-latency-ms=2247`, `sr=24000`
- Bridge（`bridge_demo.py`）
  - 首包可播放音频: **9.237s**
  - 总时长: **34.419s**
  - 音频总时长: **36.204s**

## 环境信息（待补全）
- Driver/CUDA：`550.127.05`（CUDA 12.x）
- vLLM version：`0.14.0`
- vLLM-Omni（TTS）commit：`578ed1967eb8596bf4993bcaa84735cb35ca39f3`
- GPU：`NVIDIA L40S`
- LLM server 日志：`/workspace/project 1/25/output/llm_server.log`
- TTS server 日志：`/workspace/project 1/25/output/tts_server.log`

## 已知风险与下一步
- 多模态输入（audio/image/video）暂不接入 Thinker 主线
- 后续补：speech_plan、插话与情绪控制
- TTS 依赖 SoX 未安装（当前不会阻塞，但功能受限）
- TTS tokenizer 有 regex warning，建议后续配置 `fix_mistral_regex=True`

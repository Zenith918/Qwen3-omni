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
  - `gpu_memory_utilization=0.6`（默认值，可用环境变量调低）
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
- 双段策略：
  - starter 段：2~6 字，优先首包
  - main 段：20~60 字，按标点/停顿切分，避免 1~2 字硬切
- 口径：
  - t_req_in：请求进入服务
  - t_first_audio_out：第一次写出音频 bytes
  - t_done：请求完成
  - warm：启动 warmup 后的第 2 个请求起算

## D) 桥接 Demo
- 脚本：`/workspace/project 1/25/scripts/run_demo_bridge.sh`
- 说明：LLM streaming 文本切分后逐段调用 TTS
- 验收结果：输出 6 段音频文件，首段音频生成可用
- Bridge 策略：
  - 生产者-消费者并行：LLM streaming 与 TTS 并行
  - flush：中文标点立即 flush；无标点时 8~12 字 flush
  - starter 段：2~6 字启动段优先送 TTS
  - stop/barge-in：支持 `/bridge/stop`（client-side stop）

## E) 指标（待补全）
- LLM（`llm_smoke_test.py`）
  - TTFT: **0.078s**
  - tokens/s: **171.49**
  - tokens: **221**
  - peak VRAM: **约 32.9 GiB / 46.1 GiB**（运行中 nvidia-smi）
- TTS（`tts_smoke_test.py`）
  - cold 首包延迟: **1.066s**
  - warm TTFA P50/P95: **1.013s / 1.309s**
  - warm 总耗时 P50/P95: **3.099s / 3.702s**
  - warm RTF P50/P95: **0.915 / 0.931**
  - 头信息：`x-sample-rate=24000`, `x-chunk-ms=30`, `x-warm-request`, `x-segments=2`
  - 备注：已避开 LLM 加载并发
- Bridge（`bridge_demo.py`）
  - 首包可播放音频 P50/P95: **1.401s / 1.401s**（n=1）
  - 总时长 P50/P95: **5.190s / 5.190s**（n=1）
  - 音频总时长: **1.417s**（n=1）
  - chunk 数 & flush 触发统计: **chunks=2, flush_punct=1, flush_len=0, starter=1**
  - 备注：连续运行出现 TTS read timeout（需重启 TTS 服务后恢复）；本次样本来自 `BRIDGE_MAX_TOKENS=8`、`BRIDGE_MAX_SEGMENTS=2` 的短答测试

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
- 深度流式结论：当前 vllm-omni 仅返回最终音频，无法获取中间 codes/decoder 状态
  - 阻塞点：`Omni.generate()` 不暴露 code predictor 输出
  - Plan-B：等官方 online serving / 修改 vllm-omni worker 暴露中间 codes
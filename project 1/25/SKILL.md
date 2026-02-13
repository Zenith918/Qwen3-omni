# Qwen3-TTS Deep Streaming — 工程 SOP

> 本文档面向新加入的工程师，目标：让你能在 30 分钟内启动服务、跑回归、理解 CUDA Graph 加速原理，并避免已知踩坑。

---

## 目录

1. [项目架构概览](#1-项目架构概览)
2. [环境 & 依赖](#2-环境--依赖)
3. [启动服务](#3-启动服务)
4. [环境变量速查表](#4-环境变量速查表)
5. [回归测试](#5-回归测试)
6. [CUDA Graph 加速](#6-cuda-graph-加速)
7. [关键指标定义](#7-关键指标定义)
8. [常见错误 & 修复](#8-常见错误--修复)
9. [踩坑经验 & 铁律](#9-踩坑经验--铁律)
10. [文件索引](#10-文件索引)

---

## 1. 项目架构概览

```
请求 → /tts/stream (FastAPI)
          │
          ├─ Codegen（Talker + Code Predictor）
          │    Talker: 自回归生成 codec token 的第 0 组
          │    Code Predictor (CP): 14 个 lm_head 并行预测剩余组
          │
          └─ Decoder（IncrementalDecoder）
               quantizer → pre_conv → pre_transformer
               → 2x (TransConv + ConvNeXt) upsample
               → decoder blocks (CausalConvNet, ResidualUnit, SnakeBeta)
               → PCM 音频输出
```

### 关键概念

| 概念 | 说明 |
|---|---|
| **Deep Streaming** | 增量 codegen + 增量 decode → 实时 PCM 输出 |
| **packet_tokens** | 每次 codegen 生成的 token 数（默认 2） |
| **left_context** | 解码器保留的上下文长度（默认 72） |
| **Incremental Decode** | 流式卷积，逐步生成 PCM，不需要完整 codes |
| **CUDA Graph** | 将 CUDA kernel 序列"录像+重放"，减少 CPU launch 开销 |

### 整体系统架构

```
用户 → LLM (vLLM OpenAI API)
         │ streaming text
         ├─ Bridge (文本切分)
         │    flush 策略：中文标点立即 flush；无标点时 8~12 字
         │    starter 段：2~6 字优先送 TTS 降低首包延迟
         │
         └─ TTS Server (/tts/stream)
              Deep Streaming: 增量 codegen → 增量 decode → PCM
```

---

## 2. 环境 & 依赖

```bash
# GPU & Driver
GPU:   NVIDIA L40S (48 GiB)
CUDA:  12.x
Driver: 550.127.05
vLLM:  0.14.0

# Python 路径
export PYTHONPATH="/workspace/vllm-omni"

# 模型路径
/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice    # ← TTS 当前主力模型
/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice    # TTS 1.7B 备选
/workspace/models/Qwen3-Omni-AWQ-4bit                # LLM（AWQ 4-bit 量化）

# vLLM-Omni（仅提供 TTS 模型加载能力，不走 vLLM 推理引擎）
# ⚠️ 有本地补丁（CUDA Graph 支持、decoder forward 签名等），不要 git checkout/reset
/workspace/vllm-omni
```

> LLM（gpu_memory_utilization=0.6）占用 ≤ 27 GiB，TTS 0.6B 实测占用 ~3.4 GiB。
> 合计 ~30.4 GiB，L40S (45 GiB) 可单卡共存，余量 ~14.6 GiB。
> 注意：如果调高 LLM 的 `gpu_memory_utilization`（如 0.9），则可能挤压 TTS 导致 OOM。

---

## 3. 启动服务

### 3.1 LLM Server（vLLM OpenAI API）

```bash
bash "/workspace/project 1/25/scripts/run_llm_server.sh"
```

关键参数（脚本内已配置）：
- `max_model_len=2048`
- `gpu_memory_utilization=0.6`
- `quantization=compressed-tensors`
- `kv_cache_dtype=fp8`

验证：
```bash
# 模型列表
curl http://localhost:8000/v1/models

# 文本生成
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Omni-AWQ-4bit","messages":[{"role":"user","content":"你好"}]}'
```

LLM 性能参考（L40S）：TTFT 0.078s, 171 tokens/s, VRAM ≤ 27 GiB (util=0.6)。

### 3.2 Bridge Demo（LLM → TTS 桥接）

```bash
bash "/workspace/project 1/25/scripts/run_demo_bridge.sh"
```

Bridge 策略：
- 生产者-消费者并行：LLM streaming 与 TTS 并行
- starter 段：2~6 字优先送 TTS 降低首包
- 中文标点立即 flush；无标点时 8~12 字 flush
- 支持 `/bridge/stop`（client-side barge-in）

### 3.3 TTS Server

### 方式 A：使用脚本（推荐）

```bash
cd "/workspace/project 1/25"
bash scripts/run_tts_server.sh
```

脚本已内置黄金基线配置（含 CUDA Graph CP=1, Decoder=1）。需要修改参数时通过环境变量覆盖：

```bash
# 例：关闭 Decoder Graph 做对比实验
TTS_DECODER_CUDAGRAPH=0 bash scripts/run_tts_server.sh
```

### 方式 B：直接启动（调试用）

```bash
cd "/workspace/project 1/25/clients"
export PYTHONPATH="/workspace/vllm-omni"

# === 必须参数 ===
export TTS_DEEP_STREAM_ENABLE=1
export TTS_DEEP_STREAM_PROCESS=0
export TTS_DEEP_STREAM_DECODE_MODE=incremental
export TTS_DEEP_STREAM_PACKET_TOKENS=2
export TTS_DEEP_STREAM_LEFT_CONTEXT=72
export TTS_DEEP_STREAM_DETERMINISTIC=1
export TTS_DEEP_STREAM_DETERMINISTIC_POLICY=greedy
export TTS_DEEP_STREAM_SEED_MODE=fixed
export TTS_DEEP_STREAM_SEED=42
export TTS_DEEP_STREAM_DEVICE=cuda:0
export TTS_DEEP_STREAM_CODEGEN_DEVICE=cuda:0
export TTS_DEEP_STREAM_MODEL_DIR="/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
export TTS_CODEGEN_GROUP_PARALLEL=0

python3 tts_server.py
```

### 验证服务就绪

```bash
# 等待 "Application startup complete"，然后：
curl -s http://localhost:9000/tts/stream \
  -X POST -H "Content-Type: application/json" \
  -d '{"text":"你好世界","speaker":"serena"}' \
  -o /dev/null -w '%{http_code}'
# 应返回 200
```

### 停止服务

```bash
pkill -f "python3.*tts_server"
# 确保无残留：
ps aux | grep tts_server
```

---

## 4. 环境变量速查表

### 核心配置（铁律——不可改变）

| 变量 | 值 | 说明 |
|---|---|---|
| `TTS_CODEGEN_GROUP_PARALLEL` | `0` | **禁止改为 auto**，会毁音质 |
| `TTS_DEEP_STREAM_DECODE_MODE` | `incremental` | 增量解码模式 |
| `TTS_DEEP_STREAM_PACKET_TOKENS` | `2` | 每包 token 数 |
| `TTS_DEEP_STREAM_LEFT_CONTEXT` | `72` | 解码器左上下文 |
| `TTS_DEEP_STREAM_PROCESS` | `0` | 同进程模式 |
| `TTS_DEEP_STREAM_DETERMINISTIC` | `1` | 确定性模式 |
| `TTS_DEEP_STREAM_DETERMINISTIC_POLICY` | `greedy` | 贪心采样 |
| `TTS_DEEP_STREAM_SEED_MODE` | `fixed` | 固定种子 |
| `TTS_DEEP_STREAM_SEED` | `42` | 种子值 |

### CUDA Graph 开关

| 变量 | 默认 | 说明 |
|---|---|---|
| `TTS_CODEGEN_CUDAGRAPH_TALKER` | `0` | Talker CUDA Graph（**保持 0**，浮点不 bit-exact） |
| `TTS_CODEGEN_CUDAGRAPH_CP` | **`1`** | Code Predictor CUDA Graph（**黄金配置默认开启**） |
| `TTS_DECODER_CUDAGRAPH` | **`1`** | Decoder CUDA Graph（**黄金配置默认开启**） |

### 调试开关

| 变量 | 默认 | 说明 |
|---|---|---|
| `TTS_DEEP_STREAM_METRICS` | `0` | 打印逐包耗时指标（**会影响性能**，仅调试用） |
| `TTS_DEEP_STREAM_PACKET_TRACE` | `0` | 打印逐包 trace |
| `TTS_CODE_DUMP_ENABLE` | `0` | 保存 codes 到磁盘 |
| `TTS_CODE_DUMP_DIR` | `output/code_dumps` | codes dump 目录 |

### 回归/质量

| 变量 | 说明 |
|---|---|
| `TTS_REGRESSION_BASELINE` | 黄金基线 summary.json 路径 |
| `TTS_GATE_SNR_BASELINE_DB` | SNR 质量门限（默认 15dB） |
| `TTS_DEEP_STREAM_SILENCE_PACKETS` | 静音包检测（回归时设为 0 避免时长偏差） |
| `TTS_DEEP_STREAM_SILENCE_PACKETS_P1` | 同上 |
| `TTS_DEEP_STREAM_OFFLINE_FROM_CODES` | stream/offline 一致性（回归时设为 1） |

---

## 5. 回归测试

### 5.1 黄金基线

当前黄金基线：`/workspace/project 1/25/output/regression/20260208_200725/summary.json`

基线配置：0.6B 模型，GP=0，incremental，packet=2，left_context=72，greedy seed=42，**CP Graph=1，Decoder Graph=1**。

> 历史基线（无 Graph）：`output/regression/20260207_192126/`，保留供参考。

### 5.2 运行回归

```bash
# Fast（约 2-3 分钟，不保存 wav）
bash "/workspace/project 1/25/scripts/run_ci_regression.sh" --mode fast

# Full（约 10 分钟，保存 wav 供试听）
bash "/workspace/project 1/25/scripts/run_ci_regression.sh" --mode full
```

### 5.3 质量 Gates（所有 gate 必须 PASS）

| Gate | 说明 | 阈值 |
|---|---|---|
| `determinism` | 多次运行 hash 一致 | hash_unique=1 |
| `abs_duration_diff_ms` | stream vs offline 时长差 | ≤ 500ms |
| `repeat` | 无重复片段 | 0 |
| `SNR_vs_baseline` | 与黄金基线信噪比 | ≥ 15dB |
| `TTFA` | 首音频包延迟 | ≤ 350ms |
| `stream_bad_audio` | 无空/损坏音频 | empty |

### 5.4 典型回归流程

```bash
# 1. 启动 server（黄金配置已内置 CUDA Graph）
bash scripts/run_tts_server.sh &

# 2. 等待就绪（看到 "Application startup complete"）

# 3. 跑 fast regression
bash scripts/run_ci_regression.sh --mode fast

# 4. 确认 PASS 后跑 full
SAVE_WAV=1 bash scripts/run_ci_regression.sh --mode full

# 5. 检查产物
ls output/regression/latest/
# summary.json, summary_brief.json, *.wav
```

---

## 6. CUDA Graph 加速

### 6.1 原理

TTS 推理瓶颈是 **CPU 端 cudaLaunchKernel 调用**（占 CPU 时间 91%+）。CUDA Graph 将一系列 GPU 操作"录制"为图，后续只需一次 graph launch 即可重放所有操作。

### 6.2 Code Predictor Graph（推荐开启）

```bash
export TTS_CODEGEN_CUDAGRAPH_CP=1
```

- 14 个 lm_head 各一个 graph，共享一个 frozen cache
- 仅 decode step（q_len=1）走 graph；prefill 保持 eager
- 形状不匹配自动 fallback eager

**效果**：codegen-only RTF 从 0.89 降至 0.45（1.97x 加速），100% bit-exact。

### 6.3 Decoder Graph（推荐开启）

```bash
export TTS_DECODER_CUDAGRAPH=1
```

- 对 conv/upsample 路径做 graph capture
- 服务启动时预捕获（pre-capture），避免运行时 `cudaErrorStreamCaptureUnsupported`
- 使用专用 CUDA stream 隔离 codegen 和 decode
- `kernel_size=1` 的卷积跳过 state 收集

**效果**：e2e RTF P50 从 ~0.93 降至 ~0.70。

### 6.4 Talker Graph（不推荐）

```bash
export TTS_CODEGEN_CUDAGRAPH_TALKER=0  # 保持关闭
```

由于 full-buffer attention 的浮点精度差异，Talker Graph 无法实现 bit-exact。

### 6.5 CUDA Graph 调试要点

1. **必须用 `torch.no_grad()`**，不能用 `torch.inference_mode()`（后者会报 "Inference tensors cannot be saved for backward"）

2. **graph capture 前必须做 warmup run**，否则 cuDNN/cuBLAS workspace 未初始化，输出全零

3. **capture 后返回 replay 的输出**，不能返回 capture-time 的输出（capture-time 输出只是占位符）

4. **CUDA Graph capture 是进程级全局状态**，在 `_init_deep_stream_backend` 启动时预捕获，不能在请求处理时捕获

5. **DecoderGraphAccelerator 必须全局缓存**，每个请求复用同一实例，否则每次重新 capture 导致 TTFA 暴增

6. **静态 buffer + copy_()**：所有动态输入必须先 `copy_()` 到预分配的静态 tensor，再 `graph.replay()`

7. **专用 CUDA stream**：decoder graph 需要专用 stream，避免与 codegen 的默认 stream 冲突
   ```python
   self._stream = torch.cuda.Stream()
   self._stream.wait_stream(torch.cuda.current_stream())  # 等输入就绪
   with torch.cuda.stream(self._stream):
       self.graph.replay()
   torch.cuda.current_stream().wait_stream(self._stream)  # 等结果就绪
   ```

8. **state buffer 只收集 kernel_size > 1 的卷积**，kernel_size=1 的卷积无状态，收集会报错

### 6.6 CUDA Graph Fallback

当输入 shape 与 capture 时不匹配，自动 fallback eager 并记录原因：

```python
# Server meta 输出：
{
  "cudagraph_cp_used": true,
  "decoder_graph_stats": {
    "graph_steps": 120,
    "eager_steps": 1,          # step 0 永远是 eager
    "fallback_count": 0,
    "fallback_reasons_topk": {}
  }
}
```

---

## 7. 关键指标定义

| 指标 | 定义 | 目标 |
|---|---|---|
| **RTF** (Real-Time Factor) | `wall_time / audio_duration` | < 0.7 |
| **TTFA** (Time to First Audio) | 从请求到第一个音频 chunk 输出 | ≤ 350ms |
| **codegen_iter_wall_ms** | 等待 `next(codes_iter)` 的时间 | — |
| **decode_wall_total_ms** | decoder 总耗时 | — |
| **glue_wall_total_ms** | 归一化/设备传输/PCM 分块等杂项 | — |
| **SNR** | stream vs offline/baseline 信噪比 | ≥ 15dB |
| **launches/step** | 每步 cudaLaunchKernel 调用数 | 越少越好 |

### 性能基线（L40S, 0.6B 模型）

| 配置 | RTF P50 | RTF P95 | TTFA P95 | 状态 |
|---|---|---|---|---|
| Baseline (no graph) | 0.93 | 0.98 | 330ms | 历史参考 |
| CP-only graph | 0.93 | 0.93 | 247ms | — |
| **CP + Decoder graph** ✅ | **0.70** | **0.76** | **244ms** | **当前黄金配置** |

> 黄金基线产物：`output/regression/20260208_200725/`（Full regression, 10-run determinism, SNR 120dB, 全 gate PASS）

---

## 8. 常见错误 & 修复

### 启动类

| 错误 | 原因 | 修复 |
|---|---|---|
| `KeyError: 'qwen3_tts'` | 模型未注册 | 确保用 `Qwen3TTSModel.from_pretrained` 加载 |
| `ValueError: Unsupported speakers` | speaker 名错误 | 使用 `serena`（不是 `Chelsie`） |
| `GPU free < 8000 MiB; aborting` | 显存不足 | `pkill -f python3` 清理残留进程 |
| 端口已占用 | 上次进程未彻底退出 | `pkill -f tts_server` |

### CUDA Graph 类

| 错误 | 原因 | 修复 |
|---|---|---|
| `Inference tensors cannot be saved for backward` | 用了 `inference_mode` | 改用 `torch.no_grad()` |
| `cudaErrorStreamCaptureUnsupported` | 请求处理时 capture | 启动时预捕获 |
| Graph 输出全零 | 没做 warmup | capture 前加 warmup run |
| Graph 输出不 bit-exact | capture-time 输出 vs replay 输出 | 返回 replay 后的 `static_out` |
| `operation not permitted when stream is capturing` | 默认 stream 上有并发操作 | 用专用 `torch.cuda.Stream()` |
| `IndexError: index 15 is out of range` (CP) | `n_q` 值错误 | 用 `decoder.quantizer.max_n_q`（=16） |

### 回归类

| 错误 | 原因 | 修复 |
|---|---|---|
| `stream_bad_audio=empty` | 没读取 stream 数据 | 确保 `iter_content` 在 `if save_wav` 外 |
| `baseline_missing_or_no_cases` | 基线路径无效 | 检查 `TTS_REGRESSION_BASELINE` 路径 |
| 时长偏差大 | 静音提前截断 | 设 `SILENCE_PACKETS=0` |
| SNR 负值 | stream 与 offline 输出不一致 | 检查 `OFFLINE_FROM_CODES=1` |

---

## 9. 踩坑经验 & 铁律

### 🚫 铁律（违反必出 bug）

1. **禁止 `GROUP_PARALLEL=auto`**：会导致 predictor 并行时音质崩溃，SNR 暴跌
2. **禁止 `TTS_CODEGEN_CUDAGRAPH_TALKER=1`**：浮点精度不一致，无法 bit-exact
3. **回归时必须关闭 `SILENCE_PACKETS`**：否则生成的音频被截短，abs_duration_diff 暴增
4. **Graph capture 必须在 `torch.no_grad()` 下**：不要用 `inference_mode()`
5. **DecoderGraphAccelerator 必须全局单例**：不要每个请求重建

### 💡 经验

1. **codegen 是吞吐瓶颈**：91%+ CPU 时间花在 `cudaLaunchKernel`，不是 GPU 计算
2. **decode 中 conv/upsample 占 90%+**，pre_transformer 只占约 10%
3. **`torch.cuda.synchronize()` 会严重干扰计时**：如需拆分计时，用 CUDA event 或 `time.monotonic()`
4. **vLLM-Omni 只提供模型加载**：实际推理走 HuggingFace `generate()`，不走 vLLM 引擎
5. **杀 server 后要确认进程彻底退出**：vLLM worker 可能变成僵尸进程占显存
6. **首次请求 TTFA 偏高**：cuDNN 和 CUDA runtime 需要初始化，第 2 次开始才是稳态
7. **CP Graph 的 14 个 lm_head 必须共享同一个 frozen cache**：分开的 cache 会导致不 bit-exact
8. **DynamicCache 的 `get_seq_length()` 返回 `MAX_SEQ_LEN`**：需要 monkey-patch 返回 `_seen_tokens`

### 🔍 调试技巧

1. **快速验证确定性**：同一输入跑 3 次，比较 PCM hash 是否一致
2. **用 `nsys profile` 抓 kernel launch**：`nsys profile -t cuda python3 your_script.py`
3. **用 code_dump 保存 codes**：`TTS_CODE_DUMP_ENABLE=1`，然后可以离线重放 decoder
4. **decoder-only microbench**：`python3 decoder_microbench.py` 可以隔离测试 decoder 性能
5. **codegen-only benchmark**：`python3 codegen_only_benchmark.py` 可以隔离测试 codegen 性能
6. **throughput benchmark**：`python3 throughput_benchmark.py --server http://localhost:9000 --duration 30`

---

## 10. 文件索引

### 核心服务

| 文件 | 说明 |
|---|---|
| `clients/tts_server.py` | TTS 主服务（FastAPI），包含 `/tts/stream` 和 `/synthesize` |
| `clients/tts_incremental_decoder.py` | 增量解码器，流式 conv/upsample |
| `clients/codegen_cudagraph.py` | Codegen CUDA Graph（Talker + CP） |
| `clients/decoder_cudagraph.py` | Decoder CUDA Graph |

### 测试 & 基准

| 文件 | 说明 |
|---|---|
| `clients/tts_regression_suite.py` | 回归测试套件（fast/full） |
| `clients/codegen_only_benchmark.py` | Codegen-only RTF 基准 |
| `clients/decoder_microbench.py` | Decoder-only 微基准 |
| `clients/throughput_benchmark.py` | 端到端吞吐基准 |
| `clients/tts_codes_dump.py` | Codes dump 工具 |
| `clients/tts_codes_eval.py` | Codes 质量评估 |
| `clients/tts_cancel_stress.py` | 取消请求压测 |
| `clients/tts_cached_decode_poc.py` | 缓存解码 PoC（参考用） |
| `clients/llm_smoke_test.py` | LLM 烟测 |

### Voice Agent 运行时 (D1–D5 新增)

| 文件 | 说明 |
|---|---|
| `runtime/livekit_agent.py` | **LiveKit Voice Agent** — VAD→STT(Omni)→LLM(Omni)→TTS, 含 TraceCollector 9点打点 |
| `runtime/token_server.py` | JWT Token API (:3000) + 前端静态文件托管 |
| `runtime/webrtc_test.html` | WebRTC 前端 UI（浏览器端 EoT 检测 + P50/P95 统计） |
| `runtime/duplex_controller.py` | 双工状态机（LISTENING/THINKING/SPEAKING/INTERRUPTING）+ 级联 cancel |
| `runtime/gpu_scheduler.py` | GPU 硬优先级调度器（fast lane 抢占, slow lane try_acquire） |
| `runtime/vad_silero.py` | Silero VAD 封装（CPU, 512 samples @16kHz） |
| `runtime/live_duplex.py` | 模拟 live 对话会话（用 WAV 文件模拟麦克风） |

### 脚本

| 文件 | 说明 |
|---|---|
| `scripts/run_tts_server.sh` | 启动 TTS 服务（黄金配置 + CUDA Graph + auto-restart） |
| `scripts/run_ci_regression.sh` | 运行 CI 回归（`--mode fast/full`） |
| `scripts/run_llm_server.sh` | 启动 LLM 服务（vLLM OpenAI API） |
| `scripts/start_all.sh` | **一键管理** `{start\|restart\|stop\|status}` 所有服务 |
| `scripts/supervisor_voice_agent.conf` | Supervisor 进程管理配置（备用） |
| `scripts/setup_tts_env.sh` | TTS 环境初始化 |
| `scripts/setup_llm_env.sh` | LLM 环境初始化 |
| `/post_start.sh` | RunPod Pod 重启后自动恢复所有服务 |

### 配置

| 文件 | 说明 |
|---|---|
| `clients/texts_p0_base.json` | 测试文本集（含 short_01, long_03 等） |
| `clients/voices_base.json` | 测试语音配置 |
| `artifacts/qwen3_tts_l40s.yaml` | L40S 低显存配置 |

### 输出

| 目录 | 说明 |
|---|---|
| `output/regression/20260208_200725/` | **当前黄金基线**（CP+Decoder Graph, 全 PASS） |
| `output/regression/latest/` | 最新回归的符号链接 |
| `output/day5_e2e_traces.jsonl` | D5 端到端延迟 trace（22 轮） |
| `output/day3_stress_cancel_report.json` | D3 压测报告（200轮, cancel P95=7.5ms） |
| `output/day3_vad_eval.json` | D3 VAD 评估结果 |

### Voice Agent 环境变量速查

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VAD_SILENCE_MS` | 200 | VAD hangover（静音判定），越小越灵敏 |
| `TTS_FRAME_MS` | 20 | TTS 发布帧粒度 ms |
| `MIN_ENDPOINTING` | 0.3 | LiveKit 最小 endpointing delay |
| `ENABLE_CONTINUATION` | 1 | LLM 延续句机制（先短后长） |
| `LLM_MAX_TOKENS` | 150 | LLM 最大 token 数 |
| `LLM_TEMPERATURE` | 0.3 | LLM 温度 |
| `LIVEKIT_URL` | wss://...livekit.cloud | LiveKit Cloud 地址 |
| `LIVEKIT_API_KEY` | — | LiveKit API Key |
| `LIVEKIT_API_SECRET` | — | LiveKit API Secret |

### Voice Agent 快速启动

```bash
# 1. 确保 LLM + TTS 已运行
bash scripts/start_all.sh status

# 2. 启动全部服务（含 Agent + Token Server）
bash scripts/start_all.sh start

# 3. 浏览器访问（通过 Jupyter proxy）
# https://POD_ID-8888.proxy.runpod.net/proxy/3000/?token=JUPYTER_TOKEN

# 4. 查看延迟 trace
cat output/day5_e2e_traces.jsonl | python3 -m json.tool

# 5. 重启 Agent（改代码后）
bash scripts/start_all.sh restart
```

### LiveKit Agent 踩坑经验（v1.4 API）

| 坑 | 解决方案 |
|---|---|
| `JobContext` 没有 `participant` 属性 | 用 `ctx.connect()` + `ctx.wait_for_participant()` |
| `AgentSession.start()` 不接受 `participant` | 只传 `agent` 和 `room` |
| `LLMStream.__init__()` 缺参数 | 必须传 `tools=[]` 和 `conn_options` |
| `ChatChunk` 缺 `id` 字段 | 必须传 `id="omni"` |
| `ChunkedStream._run()` 签名变化 | 必须接受 `output_emitter` 参数 |
| `AudioEmitter isn't started` | **必须在 `_run()` 开头就调 `initialize()`**，即使没音频也推静音帧 |
| `start_segment()` 仅限 stream=True | 非流式 ChunkedStream 不用 segment 管理 |
| Omni audio 格式 | 用 `{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}`，不是 `input_audio` |
| 同步 HTTP 阻塞事件循环 | 所有 requests.post 必须 `run_in_executor()` |
| 音频编码阻塞主线程 | base64 编码也要 offload 到线程 |

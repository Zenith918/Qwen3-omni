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

**AutoRTC 黄金基线（D11 冻结）**：`golden/d10_baseline/summary.json`

基线配置：0.6B 模型，GP=0，incremental，packet=2，left_context=72，greedy seed=42，**CP Graph=1，Decoder Graph=1**。

| 指标 | 基线值 |
|------|--------|
| PRIMARY_KPI (EoT→FirstAudio P95) | 17.23 ms |
| BASELINE_VERSION | D10_R4 |
| total_cases | 16 (12 P0 + 4 P1) |
| ok_cases | 16/16 |

**TTS 回归基线**：`output/regression/20260208_200725/summary.json`

> 历史基线（无 Graph）：`output/regression/20260207_192126/`，保留供参考。

### 5.2 运行回归

```bash
# Fast（约 2-3 分钟，不保存 wav）
bash "/workspace/project 1/25/scripts/run_ci_regression.sh" --mode fast

# Full（约 10 分钟，保存 wav 供试听）
bash "/workspace/project 1/25/scripts/run_ci_regression.sh" --mode full
```

### 5.3 质量 Gates

#### TTS 回归 Gates（所有 gate 必须 PASS）

| Gate | 说明 | 阈值 |
|---|---|---|
| `determinism` | 多次运行 hash 一致 | hash_unique=1 |
| `abs_duration_diff_ms` | stream vs offline 时长差 | ≤ 500ms |
| `repeat` | 无重复片段 | 0 |
| `SNR_vs_baseline` | 与黄金基线信噪比 | ≥ 15dB |
| `TTFA` | 首音频包延迟 | ≤ 350ms |
| `stream_bad_audio` | 无空/损坏音频 | empty |

#### AutoRTC Gates（9 gates，D11 校准后）

| Gate | 阈值 | 说明 |
|---|---|---|
| `EoT→FirstAudio P95` | ≤ 650ms | 端到端响应延迟 |
| `tts_first→publish P95` | ≤ 120ms | TTS 首帧到发布延迟 |
| `audible_dropout (P0 reply)` | == 0 | 可听断裂次数 |
| `max_gap (P0 reply)` | **< 350ms** | reply 段内最大静音间隙（D11 从 200→350） |
| `clipping_ratio` | < 0.1% | 削波比例 |
| `fast lane TTFT P95` | ≤ 80ms | LLM 快车道首 token |
| `P0 audio valid rate` | 100% | 有声比例 |
| `inter_arrival P95` | ≤ 30ms | 帧间到达时间 |
| `PRIMARY_KPI regression` | ≤ 30ms | D11 新增：主线 KPI 不恶化超过 30ms |

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
| `clients/tts_regression_suite.py` | TTS 回归测试套件（fast/full） |
| `clients/codegen_only_benchmark.py` | Codegen-only RTF 基准 |
| `clients/decoder_microbench.py` | Decoder-only 微基准 |
| `clients/throughput_benchmark.py` | 端到端吞吐基准 |
| `clients/tts_codes_dump.py` | Codes dump 工具 |
| `clients/tts_codes_eval.py` | Codes 质量评估 |

### AutoRTC 回归系统（D9-D11）

| 文件 | 说明 |
|---|---|
| `tools/autortc/run_suite.py` | AutoRTC 测试编排器（fast/nightly 模式） |
| `tools/autortc/audio_metrics.py` | 三层指标分析 + gate 判定 + PRIMARY_KPI |
| `tools/autortc/baseline_stability.py` | D11 波动统计工具 |
| `tools/autortc/user_bot.py` | 用户音频推送 bot |
| `tools/autortc/probe_bot.py` | Agent 音频录制 bot |
| `tools/autortc/common.py` | 通用工具（wav I/O、JSON I/O） |
| `tools/autortc/cases/all_cases.json` | 全部 16 个测试用例（fast suite） |
| `tools/autortc/cases/mini_cases.json` | 4 个代表性用例（日常迭代） |
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
| `golden/d10_baseline/` | **D11 冻结黄金基线**（16 case, PRIMARY_KPI=17.23ms） |
| `output/baseline_stability/` | D11 波动统计报告 + mini runs |
| `output/regression/20260208_200725/` | TTS 回归黄金基线（CP+Decoder Graph, 全 PASS） |
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

---

## 11. AutoRTC 自动回归系统

### 11.1 架构概览

```
run_suite.py                    ← 编排层
  ├── user_bot.py  (推用户音频)   ← LiveKit participant "user-bot"
  ├── probe_bot.py (录 Agent 音频) ← LiveKit participant "probe-bot"
  └── audio_metrics.py (质量分析)  ← 三层指标 + gates
```

每个 test case：
1. `user_bot` + `probe_bot` 加入同一 LiveKit room
2. `probe_bot` 订阅 Agent 音轨，确认收到首帧后发 `autortc.probe_ready`
3. Agent 收到 `probe_ready` 后确认 track published，回发 `autortc.agent_ready`（D10 双向 ACK）
4. `user_bot` 收到 `probe_ready` + `agent_ready` 双 ACK 后才推音频
5. Agent 处理后发 `autortc.reply_start` / `autortc.reply_end` 标记回复段
6. `probe_bot` 录制 `post_rtc_full.wav`（全段）和 `post_rtc_reply.wav`（回复段）
7. `audio_metrics.py` 对比 `pre_rtc.wav`（Agent TTS 直出）与 `post_rtc_reply.wav`（经 WebRTC 后）
8. `audio_metrics.py` 输出 Suggested Fixes：每个 FAIL/WARN 映射到具体排查动作

### 11.2 关键设计决策

| 决策 | 原因 |
|------|------|
| **reply 段切片而非全段测量** | 全段包含 welcome→等待→reply 的自然间隔，会导致 max_gap/dropout 假阳性 |
| **probe_ready barrier** | 不加 barrier 时 probe 可能还没订阅就开始录音，导致静音 |
| **trace_id 确定性路径** | 按 case_id/修改时间猜文件不可复现，必须用 `trace_id` 做唯一主键 |
| **capture_status 分类** | 区分"传输导致静音"(POST_SILENT) 和"音质差异"(mel_distance)，不混淆 |
| **P1 WARN 不 FAIL** | 异常指纹（boom/speed/distort）阈值未校准前先监控，不计入 PASS/FAIL |

### 11.3 三层指标体系

| 层 | 指标 | 说明 |
|----|------|------|
| Ring0 (传输) | `post_rms`, `max_gap`, `audible_dropout`, `clipping_ratio` | 音频是否完整到达 |
| Ring1 (音质) | `mel_distance`(pre vs post), `hf_ratio_drop` | 经 WebRTC 后音质有无劣化 |
| Ring2 (延迟) | `eot_to_first_audio`, `tts_first_to_publish`, `fast_lane_ttft` | 端到端响应速度 |

### 11.4 AutoRTC 踩坑经验

| 坑 | 说明 | 解决方案 |
|----|------|---------|
| **Agent 进程池耗尽** | WorkerType.ROOM 下每个 room 消耗一个子进程，连续测 16 case 时进程池被占满 | case 之间等 15s 让进程回收；每 case 用独立 room name |
| **probe 录到静音** | probe 在 Agent 发送前就开始录，或 Agent 音轨未就绪 | 实现 `probe_ready` barrier 握手 |
| **pre_rtc 文件找不到** | Agent 保存路径与 run_suite 查找路径不一致 | 统一用 `output/pre_rtc/<trace_id>/pre_rtc.wav` |
| **dropout 假阳性** | probe 帧间隔抖动被当作音频断裂 | 改为基于音频能量帧检测 gap，而非时间戳推测 |
| **subprocess 超时挂死** | bot 进程卡住导致 suite 整体终止 | 用 `try/except TimeoutExpired` 包裹 `wait()`，超时后 `kill()` |
| **max_gap 全段 vs reply 段** | 全段自然有 welcome→silence→reply 间隔 | 只在 reply 段（`reply_start` 到 `reply_end`）测 max_gap |
| **mel_distance = -1** | pre_rtc 或 post_rtc 文件缺失 | 用 `capture_status` 标记，仅 `OK` 时计算 mel |

---

## 12. Cursor Remote SSH 连接稳定性

### 12.1 断连根因分析

Cursor 通过 SSH 连接远程服务器时频繁 "Connection Error"，经排查有三层原因：

| 层级 | 根因 | 影响 |
|------|------|------|
| **🔴 最直接** | Cursor 工具调用中执行 `sleep 90-180s`，超过 tool call 无输出超时(60-120s) | Cursor 判定命令超时，报 Connection Error |
| **🟡 加重** | 高系统负载（Agent+LLM+TTS+测试进程同时跑，load avg>30） | SSH 响应变慢，加剧超时 |
| **🟡 加重** | SSH 未配置 keepalive（`ClientAliveInterval 0`） | 网络波动时无心跳保活 |
| **🟢 次要** | Cursor fileWatcher 扫描大量 `.wav` 文件导致 CPU 高 | 占用系统资源 |

### 12.2 修复方案

| 修复 | 做法 |
|------|------|
| **避免长 sleep** | 工具调用中 sleep 不超过 30s；长时间任务用 `nohup` 后台执行 |
| **nohup 后台跑** | `nohup python3 run_suite.py ... > /tmp/log.txt 2>&1 &`，用 `tail` 检查进度 |
| **SSH keepalive** | `/etc/ssh/sshd_config` 设 `ClientAliveInterval 15`, `ClientAliveCountMax 20` |
| **.cursorignore** | 排除 `output/`, `*.wav`, `models/` 减少 fileWatcher CPU |
| **定期 tail 检查** | 不阻塞等结果，而是每 30s `tail -20 /tmp/log.txt` 看进展 |

### 12.3 铁律

1. **禁止在 Cursor 工具调用中执行超过 30s 的阻塞命令**（包括 `sleep`、`wait`）
2. **长时间任务必须用 `nohup` 后台执行**，通过 `tail` 查看日志
3. **大目录（output/, models/）必须加入 `.cursorignore`**
4. **SSH keepalive 必须开启**：`ClientAliveInterval 15` + `ClientAliveCountMax 20`

## 13. D9 AutoRTC 回归调试经验

### 13.1 DataChannel 事件匹配三要素

Agent 通过 DataChannel 发送 `reply_start`/`reply_end` 事件时，必须保证：

| 要素 | 说明 | 错误案例 |
|------|------|---------|
| **reply_seq 一致性** | start 和 end 必须用同一个 seq | Agent 在 start 前递增 seq → start=0, end=1，probe 无法匹配 |
| **trace_id 过滤** | probe 只处理当前 trace_id 的事件 | 旧 Agent 进程残留的 reply_end (trace_id=None) 被错误匹配 |
| **三字段匹配** | reply_start↔end 按 trace_id + case_id + reply_seq 匹配 | 仅用 seq 匹配会被跨 case 的 stale 事件污染 |

### 13.2 Agent 进程池与 case 间隔

LiveKit Agent (WorkerType.ROOM) 每个 room 独占一个子进程。连续跑 case 时：

- 删除 room 后子进程不会立即退出（有 graceful shutdown 延迟）
- **最小间隔 18s**，否则后续 case 拿不到空闲进程 → 录到静音
- **Case 级重试**是必要保底：如果 probe 录到 rms < 0.01，自动用新 room 重跑一次

### 13.3 audio_valid 判定逻辑

```
# 正确：reply 或 full 任一有声即算有效
valid = max(reply_rms, full_rms) >= 0.01

# 错误：只看 reply_rms（reply 段切片可能错误，但 agent 确实出了声）
valid = reply_rms >= 0.01
```

reply_wav 切片依赖 DataChannel 事件时间戳，事件丢失/延迟时切片可能为空，但 full 录音证明 Agent 确实产生了音频。

### 13.4 回归门控设计原则

| 原则 | 做法 | 反面教材 |
|------|------|---------|
| **不靠放宽阈值过关** | 改测量口径（reply 段）而非调大阈值 | max_gap 从 200→1000ms 能 PASS 但无意义 |
| **分层判定** | capture_status 先判采集成功，再看音质 | mel_distance 对 POST_SILENT 无意义 |
| **重试消除非确定性** | 静音时自动重试一次 | 每次跑结果不同，gate 形同虚设 |
| **透明化** | report 中写明 reply_wav_count、capture_status 分布 | 笼统 PASS/FAIL 无法定位问题 |

---

## 14. Cursor IDE 长任务监控防断连（D10 教训）

### 14.1 根因：Cursor Cloud AI API 超时（对话 context 过长）

Cursor AI 对话走 Cursor Cloud API。Connection Error 有两个触发条件：
1. **tool call 里 sleep** → 阻塞响应 → API Gateway 超时
2. **对话 context 累积过长** → 即使命令秒级返回，AI 处理/生成时间也变长 → 超时

实测：D10 即使完全不用 sleep（命令都秒级返回），长对话仍然频繁断连。

**关键区分**：SSH 隧道始终正常（服务端日志无断连记录），断的是 AI 对话层。

### 14.2 禁止做法 + 缓解策略

```bash
# ❌ 禁止：在 tool call 里 sleep
sleep 60 && check_status

# ❌ 禁止：长循环监控
for i in $(seq 1 25); do sleep 28; check; done
```

**缓解长 context 断连**：当对话累积大量内容（跨多天工作），建议开新对话，
用 Summary 传递上下文。这是 Cursor Cloud 的限制，非代码问题。

### 14.3 正确做法：后台跑 + 即时查

```bash
# ✅ 步骤1: 后台启动长任务
python3 -u run_suite.py ... > /tmp/suite.log 2>&1 &
echo $! > /tmp/suite_pid.txt

# ✅ 步骤2: 用即时命令查进度（每次 < 5秒）
grep -c '^\[' /tmp/suite.log        # 已完成case数
tail -3 /tmp/suite.log                # 最近输出
ps -p $(cat /tmp/suite_pid.txt) -o pid=  # 是否还在跑

# ✅ 步骤3: 完成后查结果
grep -E "PASS|FAIL|RESULT" /tmp/suite.log
```

### 14.4 后台监控哨兵（可选）

如需自动通知，用**后台哨兵脚本**写结果到文件：

```bash
# 后台哨兵（is_background=true 启动）
while ps -p $PID > /dev/null 2>&1; do sleep 30; done
echo "DONE $(date)" > /tmp/suite_done.txt
```

AI 只需读 `/tmp/suite_done.txt` 是否存在，0 秒返回。

### 14.5 retry room 命名必须匹配 Agent prefix

LiveKit Agent 用 `room_prefix` 匹配 room。retry 创建的新 room 必须与原始
room 用相同前缀，否则 Agent 不会 dispatch worker 到 retry room：

```python
# ❌ retry room 前缀不匹配
case_room = f"autortc-{run_id}-{case_id}-r{attempt}"

# ✅ 保持与原始room相同的前缀
case_room = f"{args.room}-{case_id}-{run_id[-6:]}-r{attempt}"
```

### 14.6 双向 ACK Barrier 同步

单向 `probe_ready` 不够（agent 可能还没 publish track）。D10 升级为双向 ACK：

```
probe_bot → autortc.probe (probe_ready)
agent    → autortc.agent (agent_ready)   ← 确认 track published + session ready
user_bot  等 probe_ready + agent_ready 都收到后才推音频
```

Agent 侧需监听 probe 的 topic（`autortc.probe`，不是 `autortc.probe_ready`），
注意 topic 命名必须与 probe 实际发送的一致。

### 14.7 pre_rtc 必须在 TTS finally 块中保存

TTS 可能被中断（room disconnect、probe 提前离开），`pre_rtc` 必须在
`finally` 块中保存，否则中断场景下丢失：

```python
try:
    # TTS synthesis loop
    async for chunk in tts_stream:
        pre_rtc_chunks.append(chunk)
        yield chunk
finally:
    # 即使中断也保存 pre_rtc
    if pre_rtc_chunks and trace_id:
        save_pre_rtc(trace_id, pre_rtc_chunks)
```

### 14.8 recording pad 要覆盖完整链路

probe 的录音窗口 = `wav_duration + record_pad`。pad 必须覆盖：
`welcome TTS + STT处理 + LLM推理 + TTS生成 + 网络传输`

| pad 值 | 效果 |
|--------|------|
| 6s | 不够：agent 回复可能被截断，pre_rtc 来不及保存 |
| 10s | 足够：覆盖典型 welcome(3s) + 处理(3s) + 回复(4s) |

### 14.9 Nightly 必须用 per-turn room（不能同 room 复用）

LiveKit Agent (WorkerType.ROOM) 每个 room 绑定一个 worker 进程。
同 room 复用时，上一轮的 worker 可能仍在 graceful shutdown，
下一轮的 user_bot/probe_bot 重新连入时遇到 stale agent state。

| 策略 | retry_rate | 原因 |
|------|-----------|------|
| 同 room 复用, 3s wait | **50%** | agent 进程 stale，首次尝试录到静音 |
| per-turn room, 18s wait | **10%** | 大幅改善但仍有边界 case |
| **per-turn room, 20s wait** | **5%** | ✅ 达标 |

```python
# ❌ nightly 同 room 复用
case_room = nightly_room  # 所有 turn 共享一个 room

# ✅ 每个 turn 用独立 room + 统一删 room + 等回收
case_room = f"{args.room}-{case_id}-{run_id[-6:]}"
# turn 结束后: delete_room(case_room) + sleep(20)
```

---

## 15. 测试分级策略（快 vs 全）

### 15.1 原则：日常迭代 ≤ 3 分钟，阶段验收 ≤ 15 分钟

| 级别 | 用途 | cases | 预计耗时 | 何时跑 |
|------|------|-------|---------|--------|
| **mini** | 日常改代码后快速验证 | 4 个代表性 P0 | **~3 分钟** | 每次代码改动后 |
| **fast** | 完整 P0+P1 验证 | 16 全部 case | **~15 分钟** | 冻结基线/阶段交付 |
| **nightly** | 稳定性压测 | 20 turns | **~17 分钟** | 阶段交付前跑一次 |
| **stability** | 波动采样（mini×5） | 4 case × 5 runs | **~15 分钟** | 初始化基线时 |

### 15.2 Mini Cases（`tools/autortc/cases/mini_cases.json`）

4 个代表性 case，覆盖核心场景：

| case_id | 覆盖场景 |
|---------|---------|
| `endpoint_short_hello` | 短句端到端延迟 |
| `endpoint_long_sentence` | 长句 TTS 稳定性 |
| `interrupt_once` | 打断处理 |
| `noise_background` | 噪音鲁棒性 |

```bash
# 日常迭代用这个（~3分钟）
python3 -u tools/autortc/run_suite.py \
  --cases_json tools/autortc/cases/mini_cases.json \
  --token_api http://127.0.0.1:9090/api/token \
  --output_root output/autortc --ring0 0 --with_metrics 1

# 阶段验收用这个（~15分钟）
python3 -u tools/autortc/run_suite.py \
  --cases_json tools/autortc/cases/all_cases.json \
  --token_api http://127.0.0.1:9090/api/token \
  --output_root output/autortc --ring0 0 --with_metrics 1
```

### 15.3 铁律

1. **Take data 不超过 30 分钟**：如果一个采样计划超过 30 分钟，必须用 mini cases 或减少重复次数
2. **日常迭代用 mini**（~3 min），只在**阶段性交付**时才跑 full（~15 min）
3. **波动采样用 mini×5**（~15 min），不用 full×5（~75 min）
4. **Nightly 只在交付前跑一次**，不用于日常验证

---

### 14.10 P1 异常指纹检测要点

| 异常类型 | 检测位置 | 指标 | 说明 |
|---------|---------|------|------|
| **boom (爆音)** | **用户输入 wav** | `input_spike_count`, `input_max_abs_peak` | spike 在用户输入里，不在 agent 输出里 |
| **speed drift** | agent 输出 reply 段 | `drift_ratio = samples_actual / samples_expected` | >2% 偏离视为异常 |
| **distortion** | pre_rtc vs post_rtc | `hf_ratio_drop` (4-8kHz 衰减) | 高频掉 = 发闷/失真 |

关键教训：boom_trigger 的 spike 必须在 **input wav** 上检测（`_audio_quality_metrics(input_wav)`），
不能只查 agent 输出——因为 agent 的 TTS 生成的是全新音频，不会包含用户输入的 spike。

---

## 16. PRIMARY KPI 与基线校准（D11）

### 16.1 PRIMARY KPI 定义

**主线优化指标**：`eot_to_probe_first_audio_p95_ms`

含义：从用户说完最后一个字（End-of-Turn），到 probe 第一次收到 Agent 音频的 P95 延迟。
这是用户最直接感受到的"等了多久才听到回复"。

- 基线值（D10）：**17.23 ms**
- 每次跑 suite 时，report.md 顶部自动显示当前值 + baseline + Δ
- 如果 PRIMARY_KPI 比 baseline 恶化超过 30ms，自动 FAIL

### 16.2 使用方法

```bash
# 日常迭代（自动对比 golden baseline）
python3 -u tools/autortc/run_suite.py \
  --cases_json tools/autortc/cases/mini_cases.json \
  --token_api http://127.0.0.1:9090/api/token \
  --output_root output/autortc --ring0 0 --with_metrics 1 \
  --baseline_summary golden/d10_baseline/summary.json

# 也可通过环境变量指定
export TTS_REGRESSION_BASELINE_SUMMARY=golden/d10_baseline/summary.json
```

### 16.3 黄金基线目录结构

```
golden/d10_baseline/
├── summary.json          # BASELINE_VERSION + PRIMARY_KPI_VALUE
├── metrics.csv           # 全量指标 CSV
├── report.md             # Gate 报告
└── <case_id>/            # 16 个 case 各自的产物
    ├── pre_rtc.wav       # Agent TTS 直出音频
    ├── post_rtc_reply.wav # 经 WebRTC 后的回复段
    ├── probe_result.json  # probe 采集结果
    └── user_result.json   # user bot 结果
```

### 16.4 建议 Gate 阈值（基于 D11 波动统计，6 runs / 32 P0 samples）

| Gate | 当前阈值 | 统计 median | 统计 P95 | 统计 σ | 建议阈值 | 方法 |
|------|---------|-----------|---------|--------|---------|------|
| EoT→FirstAudio P95 | ≤ 650ms | 8.2ms | 18.4ms | 5.9ms | **≤ 25ms** | P95 × 1.2 + margin |
| TTS First→Publish P95 | ≤ 120ms | 0.3ms | 1.0ms | 0.3ms | **≤ 2ms** | P95 × 1.2 |
| Max Gap (P0 reply) | < 200ms | 0.0ms | 289ms | 98.6ms | **< 350ms** | P95 × 1.2 |
| Clipping Ratio | < 0.1% | 0.0 | 0.0 | 0.0 | **< 0.1%** | 保持不变 |
| Fast Lane TTFT P95 | ≤ 80ms | 62.9ms | 71.3ms | 8.6ms | **≤ 86ms** | median + 2σ |
| Audible Dropout | == 0 | 0 | 0 | 0 | **== 0** | 保持不变 |
| Audio Valid Rate | 100% | 100% | 100% | — | **100%** | 保持不变 |
| PRIMARY_KPI regression | ≤ 30ms | — | — | — | **≤ 30ms** | 硬限 |

> **关键发现**：`max_gap` 当前阈值 200ms 太紧（P95=289ms），建议放宽到 350ms。
> `interrupt_once` 案例天然有 reply 内间隙，导致 max_gap 波动大。

### 16.5 波动分析工具

```bash
# 生成波动统计报告
python3 tools/autortc/baseline_stability.py \
  --run_dirs output/baseline_stability/mini_runs/run_*/*/  \
  --output_dir output/baseline_stability
# 输出: output/baseline_stability/baseline_stability.md
```

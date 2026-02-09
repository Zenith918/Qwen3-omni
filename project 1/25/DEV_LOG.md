# 研发日志

---

## Phase 1: Deep Streaming 基础建设 (2026-01-29)

### 1.1 参数调优与基线确立

| 日期 | 变更 | TTFA P50 | RTF P50 | MAE | 状态 |
| --- | --- | --- | --- | --- | --- |
| 01-29 03:26 | 初始 deep-stream, packet=4, left_context=25 | 364ms | 1.087 | >1e-3 | ❌ FAIL |
| 01-29 03:45 | +确定性 `DETERMINISTIC=1` | 345ms | 1.040 | 2.7e-05 | ✅ PASS |
| 01-29 06:40 | +offline 直接波形 `OFFLINE_FROM_CODES=0` | 347ms | 1.029 | — | ✅ PASS |
| 01-29 06:46 | +左上下文 `LEFT_CONTEXT=72` | 356ms | 1.048 | — | ✅ PASS |
| 01-29 07:06 | packet=8（实验） | 645ms | 0.994 | — | ❌ TTFA 过高 |

**结论**：最终基线 = `packet_tokens=4, LEFT_CONTEXT=72, DETERMINISTIC=1`。packet=8 虽降 RTF 但 TTFA 不可接受。

### 1.2 爆音(Pop Noise)根因定位

| 实验 | 结论 |
| --- | --- |
| 不同 packet_tokens 爆音对齐率 | 仅 13%，爆音随 packet 变化 → 非固定音源缺陷 |
| chunk_ms=40 vs 默认 | 爆音 100% 对齐 → chunk 大小不是根因 |
| code_offline vs direct_offline | 对齐率 95.7% → **爆音来自模型输出本身（codes）** |
| stream vs offline 爆音 | streaming-style 解码与 true-offline 爆音位置不一致，B/C 高度一致 → "解码窗口/拼接策略"是强影响因素 |

### 1.3 Codes 确定性验证

- `seed=42, deterministic=1` 下同配置 codes hash 100% 一致，跨 packet_tokens 也一致。
- **codes 生成可复现，与 packet_tokens 无关。**

### 1.4 增量解码器实现 (M1)

- 新增 `tts_incremental_decoder.py`：causal conv streaming + transposed conv streaming。
- A_full vs B_incremental MAE = 0.000238（达标）。
- 接入 `/tts/stream` 路径，环境变量 `TTS_DEEP_STREAM_INCREMENTAL=1` 启用。

### 1.5 Codes 漂移问题

- stream vs offline codes hash 不一致，多次 stream 也不一致（常差 1 帧）。
- 严格确定性 (`DETERMINISTIC_STRICT=1`) 可消除漂移，但触发 **CUDA device-side assert**（长文本更易触发），不可用。
- soft 确定性可保证 stream 内部可复现，但 stream vs offline 仍有差异。

---

## Phase 2: 漂移根因定位 (2026-02-01 ~ 2026-02-02)

### 2.1 进程/并行形态 (Q1-Q3)

- `PROCESS=1` 使用常驻 worker 进程（spawn），codegen 与 decoder 默认同一 GPU。
- 主线程和 worker 线程使用 **同一个 default CUDA stream (0x0)**，GPU 上严格串行。

### 2.2 无 overlap 基线 (Q31-Q32)

| 配置 | TTFA P50 | TTFA P95 | RTF P50 | hash_unique |
| --- | --- | --- | --- | --- |
| process=0, packet=4 | 660ms | 696ms | 1.567 | 1（稳定） |
| process=0, **packet=2** | **350ms** | **401ms** | 1.556 | 1（稳定） |
| process=0, packet=1 | 40s+ | 92s+ | 7.11 | 23（异常） |

**结论**：packet=2 是 TTFA 接近 350ms 的最优选择。packet=1 路径异常（codegen 阶段卡死）。

### 2.3 漂移触发源 (Q34, Q38)

| 实验 | hash_unique | 结论 |
| --- | --- | --- |
| codegen-only | 1（稳定） | codegen 本身不漂移 |
| pre_transformer only | 1（稳定） | — |
| **conv/upsample only** | **2（漂移）** | **漂移触发源** |
| full decoder | 2（漂移） | — |
| decoder 走 noop | 1（稳定） | conv/upsample 路径参与时才触发 |

### 2.4 精度/cuDNN 实验 (Q35, Q37)

- bf16 / codegen fp32 / decoder fp32 / 全 fp32 → **全部仍漂移**。
- cudnn benchmark/deterministic 切换 → **不消除漂移**。
- 关闭 TF32 → TTFA 飙升到秒/十秒级，**漂移仍在**。
- **结论：漂移对精度不敏感，更像并行/调度级非确定性。**

### 2.5 同步方案 (Q39)

| 方案 | hash_unique | TTFA P50 | 适用性 |
| --- | --- | --- | --- |
| sync（显式同步） | 1 ✅ | 609-633ms | 消除漂移但 TTFA 高 |
| event（CUDA event 等待） | 1 ✅ | 619-621ms | 同进程有效，跨进程不适用 |
| 无同步（默认 overlap） | >1 ❌ | 435-464ms | 有漂移 |

**结论**：同步可消除漂移，但 TTFA 代价过高。最终选择 `process=0 + packet=2 + phase sync` 作为产品配置。

---

## Phase 3: 性能瓶颈分析与路线裁决 (2026-02-06 ~ 2026-02-07)

### 3.1 项目口径确立 (Q-A~Q-F)

- 合法 baseline：gp=0。gp=auto 不承认（codec frame=0 分叉 + 听感崩）。
- attention backend 必须用 profiler 证据，不能靠代码推断。
- 低开销 always-on 计时拆分（不使用 METRICS=1 避免污染性能）。
- `codegen_wall_ms` 定义明确拆分：纯 codegen-only / 纯 decode-only / glue。

### 3.2 端到端计时拆分 (Q13, 1.7B)

实现了 `codegen_iter_wall_ms` / `decode_wall_ms` / `glue_wall_ms` / `loop_wall_ms` / `tail_wall_ms` / `total_wall_ms` 六桶拆分。

| 分量 | short_01 (%) | long_03 (%) |
| --- | --- | --- |
| codegen_iter | 43.5% | 33.6% |
| decode | 53.8% | 65.4% |
| glue | 2.7% | 1.0% |

> ⚠️ 此拆分后被 D1-D6 修正（`cuda.synchronize()` 导致 decode 桶膨胀）。

### 3.3 Kernel 分析 (Q-C, 1.7B)

| 指标 | 值 |
| --- | --- |
| `pytorch_flash::flash_fwd_kernel` | 0.24% CUDA 时间（仅 prefill） |
| `gemvx`（eager GEMV） | 47% CUDA 时间（decode 主体） |
| **`cudaLaunchKernel` 次数** | **661,416（6,614/frame）** |
| **CPU launch 时间占比** | **75.9%** |

**结论**：CPU kernel launch overhead 是性能瓶颈，不是 attention 计算本身。

### 3.4 SDPA/flash 裁决 (Q21, D3) ❌ 放弃

- 实验显示 eager vs sdpa codegen RTF 差异 +21%。
- **D3 修正**：monkey-patch `F.scaled_dot_product_attention` 计数发现两种模式调用次数**完全相同（31752 次）**。模型内部 attention module `config._attn_implementation` 始终为 `'sdpa'`，**无论顶层设置什么都走 SDPA 路径**。
- **结论：模型始终使用 SDPA，"eager" vs "sdpa" 差异为测量噪声。此路线不可评估，放弃。**

### 3.5 torch.compile 裁决 (Q22, D4) ❌ 放弃

- 实验显示 compile 后 codegen RTF +17%，kernel launches 零减少。
- **D4 修正**：TorchDynamo 追踪的 frame 数 = **0**，Inductor/Triton kernel = **0**。Dynamo 从 `generate()` 入口遇到 `while` 循环、stopping criteria、dynamic KV cache 即 graph break，**什么都没编译**。
- **结论：torch.compile 在 HF generate() 框架下完全不适用，放弃。**

### 3.6 D1-D6 关键修正

> ⚠️ 以下修正了 Q13-Q23 的多个关键计量错误。

**D1: cuda.synchronize() 计时偏差**

隔离测量（long_03, 1.7B, 308 frames）:

| 组件 | 独立 wall 时间 | RTF | 真实占比 |
| --- | --- | --- | --- |
| **codegen** | 22407ms | **0.909** | **69.3%** |
| decode | 9922ms | **0.403** | 30.7% |

原 Q19 报告 "decode 占 71.1%" 是由 `cuda.synchronize()` 捕获 codegen kernel 导致的假值。

**D5: decode-only 真实 RTF**

| 模式 | RTF | 结论 |
| --- | --- | --- |
| decode-only (incremental) | **0.398** | ✅ < 0.7，**decode 不是瓶颈** |
| decode-only (batch) | 0.357 | 增量开销 11.5% |

**核心修正**：

| 原始结论 | 修正后 |
| --- | --- |
| "decode 占 71.1%，是瓶颈" | **codegen 占 69.3%，是瓶颈** |
| "decode RTF=1.065 > 0.7，单卡不可行" | **decode RTF=0.398 < 0.7，瓶颈在 codegen** |
| "SDPA 退化 21%" | 两种模式走同一路径，无法评估 |
| "compile 退化 17%" | compile 零 tracing，什么都没做 |

### 3.7 修正后的可行性分析

**单卡 RTF < 0.7 = 有条件可行：**
1. codegen/decode 并行化（双 CUDA stream 或双卡）
2. codegen RTF 从 0.91 降至 < 0.7（≥23% 优化）
3. 主攻方向：kernel launch 开销（CPU 占 89%）

---

## Phase 4: 模型修正 + 优化路线评估 (2026-02-07)

### 4.1 🔴 严重修正：模型从 1.7B 改回 0.6B (19:15 UTC)

用户试听发现"语气太怪"，核对历史启动命令发现：**从 Q13 以来一直用 1.7B，用户正确基线是 0.6B**。

- D1-D6 所有实验结果**仅对 1.7B 有效**，需用 0.6B 重做。
- 修复 `tts_regression_suite.py` `run_stream()` bug：fast 模式不读取 stream 数据。
- 修复脚本文件污染（重复追加 19 份 `if __name__` 块）。

### 4.2 黄金基线 v2 (0.6B) ✅

**产物**: `output/regression/20260207_192126/`（已被 v3 取代，保留供参考）

| 指标 | P50 | P95 | 目标 | 状态 |
| --- | --- | --- | --- | --- |
| **TTFA** | 332ms | 335ms | ≤350ms | ✅ |
| **RTF** | 1.510 | 1.538 | <0.7 | ❌ 需 2.16x |
| MAE | 2.6e-05 | 2.7e-05 | — | ✅ |
| SNR | 64.2dB | 64.8dB | — | ✅ |
| 确定性 | 10 runs bit-exact | — | hash_unique=1 | ✅ |

### 4.3 P1 Benchmark: 0.6B 三路分解

| Component | long_03 RTF P50 | Launches/Frame |
| --- | --- | --- |
| stream (端到端) | **1.476** | — |
| codegen-only | **0.893** | **6,624** |
| decode-only | **0.442** | — |

分解 (long_03):
```
stream RTF = 1.476
├── codegen-only RTF = 0.893 (21.78s) → 60.5%
├── decode-only  RTF = 0.442 (10.78s) → 29.9%
└── glue+HTTP    RTF ≈ 0.141 (3.34s)  →  9.6%
```

### 4.4 三条优化路线评估

| 路线 | 理论收益 | 工程量 | 风险 | 建议 |
| --- | --- | --- | --- | --- |
| **1. vLLM/TRT-LLM** | 高(2-3×) | 极高(2-4周) | 致命阻碍：嵌套 generate | ⏸️ 暂缓 |
| **2. CUDA Graph per-step** | 极高(30× launch↓) | 中(1-2周) | 中：StaticCache 兼容性 | 🟢 **P0 优先** |
| **3. INT8/FP8 量化** | 低(5%单独) | 极低(1-2天) | 低 | 🟢 **P1 补刀** |

**路线 2 核心思路**：不用 torch.compile，手动将 talker/code_predictor 的单步 forward 捕获为 CUDA Graph，在 Python generate 循环中以 `graph.replay()` 替代逐 kernel 发射。

**路线 2 关键技术难点**：
- DynamicCache 每步 `torch.cat()` 导致地址变化 → 需 StaticCache 或 monkey-patch
- mRoPE 动态 ops → 需 pre-compute
- 模型声明 `_supports_static_cache = False` → 需验证

**路线 3 关键论点**：当前瓶颈是 kernel launch overhead（89%），量化只减少 kernel compute time（11%），单独使用仅改善 ~5%。但 CUDA Graph 后瓶颈转为 compute → 量化可叠加 15-20%。

---

## Phase 5: CUDA Graph 实现与验收 (2026-02-07 ~ 2026-02-08)

### 5.1 P2: 最小可行性验证 ✅

**核心创新**：用 monkey-patched `DynamicCache`（预分配静态缓冲区 + in-place `copy_()` 的 `update()`）绕过 `_supports_static_cache=False`。

**(A) Talker 单步 forward ✅**

| 指标 | Eager | Graph | 改善 |
| --- | --- | --- | --- |
| Hash | — | ✅ bit-exact | — |
| Kernel launches | 1,754 | 56 | **31.3x** |
| 单步时延 | 21.68ms | 3.76ms | **5.77x** |

**(B) Code Predictor 单步 forward ✅**

| 指标 | Eager | Graph | 改善 |
| --- | --- | --- | --- |
| Hash | — | ✅ bit-exact | — |
| Kernel launches | 299 | 10 | **29.9x** |
| 单步时延 | 3.68ms | 0.65ms | **5.65x** |

**(C) CP 14-步 decode 批量 ✅**

| 指标 | Eager | Graph | 改善 |
| --- | --- | --- | --- |
| 总时延 | 54.49ms | 10.39ms | **5.24x** |
| 总 launches | 4,469 | 140 | **31.9x** |

**技术关键发现**：
1. `torch.inference_mode()` 不兼容 CUDA Graph，必须用 `torch.no_grad()`
2. DynamicCache 可通过预分配 buffer + in-place `copy_()` 的 monkey-patch 兼容 graph capture
3. CP 有 15 组 embedding/lm_head (0..14)，需 per-step 独立 graph
4. Prefill 仍需 eager（输入形状不同），但仅占总时间 ~7%

### 5.2 P3: 工程化集成

**核心实现 (`codegen_cudagraph.py`)**：
- 两个独立 flag：`TTS_CODEGEN_CUDAGRAPH_TALKER=0|1`, `TTS_CODEGEN_CUDAGRAPH_CP=0|1`
- **CPGraphAccelerator**：14 个 per-step CUDA Graph，共享同一 frozen cache（关键修复：独立 cache 时 graph N 写入的 KV 对 graph N+1 不可见）
- **TalkerGraphAccelerator**：使用 `GraphFriendlyCache`，但存在 bit-exact 问题
- 安全机制：形状不匹配自动 fallback → eager

**Codegen-Only 端到端 Benchmark**：

| Group | RTF | Launches/Frame | BitExact | Speedup |
| --- | --- | --- | --- | --- |
| baseline (eager) | 0.893 | 6,669 | ✅ | 1.00x |
| talker=1, cp=0 | 0.815 | 4,923 | ❌ | 1.10x |
| **talker=0, cp=1** | **0.454** | **2,219** | **✅** | **1.97x** |
| talker=1, cp=1 | 0.244 | 473 | ❌ | 3.66x |

**决策：✅ PROCEED — CP-only CUDA Graph**
- CP-only: RTF 0.45, **bit-exact**, 100% graph used rate
- Talker Graph 不 bit-exact（frame count 变化 305→309），暂不启用

### 5.3 Talker Graph Bit-Exactness 调查

**根本原因**（两个独立 bug）：

1. `DynamicLayer.get_seq_length()` 对全尺寸 buffer 报告错误长度 → causal mask 大小错误
2. 全 buffer attention 有固有数值差异（IEEE 754 浮点舍入，不可消除）

| Test | 方法 | vs Baseline |
| --- | --- | --- |
| frozen_cache eager (sliced, WITH gsl fix) | 切片返回 + get_seq_length 修复 | ✅ bit-exact |
| frozen_cache eager (full buf, WITH gsl fix) | 全 buffer + get_seq_length 修复 | ❌ 数值差异 |

**结论**：CUDA Graph 要求固定大小张量 → 必须全 buffer → 固有数值差异 → **Talker CUDA Graph 无法 bit-exact**。保持 Talker eager。

### 5.4 P3.4/P3.5: Regression 验收 ✅ ALL PASS

**Fast Regression (CP-only)**:

| Gate | Value | Threshold | Status |
| --- | --- | --- | --- |
| TTFA P95 | 204ms | ≤350ms | ✅ |
| SNR vs Baseline | **120.0 dB** | ≥15 dB | ✅ |
| Determinism | hash_unique=1, 3 runs | =1 | ✅ |
| Duration Diff P95 | 23.1ms | ≤50ms | ✅ |
| Repeat Count | 0 | ≤0 | ✅ |

**Full Regression (CP-only, 10 runs)**:

| Metric | P50 | P95 |
| --- | --- | --- |
| TTFA | 212ms | 230ms |
| RTF (端到端) | 0.887 | 0.980 |
| SNR vs Baseline | 120.0 dB | 120.0 dB |
| Determinism | hash_unique=1 (long_03 + short_01, 10 runs each) | ✅ |

**SNR 120dB** = 波形与 gold baseline 近乎 bit-exact（MAE ≈ 浮点精度噪底）。

**推荐配置**：
```bash
TTS_CODEGEN_CUDAGRAPH_CP=1
TTS_CODEGEN_CUDAGRAPH_TALKER=0  # 待 bit-exact 修复
```

---

## 关键结论汇总

### 性能瓶颈
1. **Codegen 是吞吐瓶颈**（RTF 0.89，占 stream wall 60%），decode 不是（RTF 0.44）
2. 瓶颈类型 = **kernel launch overhead**（6,624 launches/frame, CPU 89% 时间在 `cudaLaunchKernel`）
3. SDPA/flash 无法切换（模型内部始终走 SDPA），torch.compile 不适用（dynamo 零 tracing）

### 已否定路线
- ❌ SDPA/flash_attn 切换：模型始终走 SDPA，无法评估
- ❌ torch.compile：HF generate() 框架导致 dynamo 零 tracing
- ❌ vLLM/TRT-LLM 原生集成：嵌套 generate 是致命架构障碍

### 已验证路线
- ✅ **CUDA Graph CP-only**：codegen RTF 0.89→0.45（1.97x），bit-exact，全 gates PASS
- 🟡 CUDA Graph Talker：3.66x（两者都开），但不 bit-exact
- 🟡 INT8/FP8 量化：单独 ~5%，与 CUDA Graph 组合可叠加 15-20%

### 漂移问题
- 漂移触发源 = conv/upsample 路径（GPU 调度非确定性）
- 对精度不敏感（bf16/fp32 均漂移）
- `process=0 + greedy + fixed seed` 可保证确定性
- event/sync 同步可消除漂移但 TTFA 代价过高

### 爆音问题
- 爆音来自模型输出本身（codes），非 streaming 造成
- 解码窗口/拼接策略对爆音位置有强影响

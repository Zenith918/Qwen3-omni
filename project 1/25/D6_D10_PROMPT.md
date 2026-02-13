# D6–D10 新聊天 Prompt

> 复制下面全部内容粘贴到新聊天中。

---

## 项目背景

我在 `/workspace/project 1/25/` 有一个 **Qwen3 实时语音通话系统**，运行在 RunPod L40S 单卡上。

### 当前架构（D5 已跑通）

```
浏览器 🎤 →WebRTC→ LiveKit Cloud → Agent Server (RunPod)
                                      │
                                 Silero VAD (CPU, hangover 200ms)
                                      │
                                 OmniSTT (Qwen3-Omni 转写, P50=104ms)
                                      │
                                 OmniLLM (Qwen3-Omni 流式回复, streaming)
                                      │
                                 QwenTTS (TTS Server :9000, 24kHz PCM)
                                      │
浏览器 🔊 ←WebRTC← LiveKit Cloud ←───┘
```

### 服务栈

| 服务 | 端口 | 说明 |
|------|------|------|
| LLM (vLLM-Omni) | :8000 | Qwen3-Omni-AWQ-4bit, ~27 GiB VRAM |
| TTS Server (FastAPI) | :9000 | Qwen3-TTS-0.6B, ~3.4 GiB VRAM, CP CUDA Graph |
| LiveKit Agent | :8089 | `runtime/livekit_agent.py`, LiveKit Agents SDK v1.4 |
| Token Server | :3000 | `runtime/token_server.py`, JWT 生成 + 前端托管 |
| Nginx | :9091 | 统一入口（但 RunPod proxy 只暴露 8888/Jupyter） |

### 关键文件

| 文件 | 说明 |
|------|------|
| `runtime/livekit_agent.py` (~678行) | **核心**: LiveKit Voice Agent, 含 OmniSTT + OmniLLM + QwenTTS + TraceCollector |
| `runtime/token_server.py` | Token API + 前端托管 |
| `runtime/webrtc_test.html` | 前端（浏览器端 EoT 检测 + P50/P95 统计） |
| `runtime/duplex_controller.py` | 状态机 + 级联 cancel |
| `runtime/gpu_scheduler.py` | 硬优先级 GPU 调度器 |
| `runtime/vad_silero.py` | Silero VAD 封装 |
| `clients/tts_server.py` | TTS FastAPI 服务（per-request cancel + crash dump + auto-restart） |
| `clients/codegen_cudagraph.py` | CUDA Graph 加速（CP 5.7x） |
| `scripts/start_all.sh` | 一键启动/重启/状态 |
| `tools/autortc/` | D6 新建的 AutoRTC 自动化测试框架 |
| `output/day5_e2e_traces.jsonl` | D5 延迟 trace（22 轮实测数据） |
| `PROJECT_BRIEFING.md` | 项目总文档 |
| `DEV_LOG.md` | 研发日志（Phase 1-6） |
| `SKILL.md` | 工程 SOP + LiveKit v1.4 踩坑经验 |

### LiveKit 配置

```
LIVEKIT_URL=wss://renshenghehuoren-mpdsjfwe.livekit.cloud
LIVEKIT_API_KEY=API7fj35wGLumtc
LIVEKIT_API_SECRET=WK8k8fUhhsHoa2R2qfO076lyuDHgJubwemQuY4nk398B
```

### Agent identity 注意

- Agent 在 LiveKit 中的 identity 是 **`agent-{JOB_ID}`** 格式（如 `agent-AJ_db4eqHkfcpYs`），不是固定的 `agent`
- probe_bot 订阅时应匹配前缀 `agent-` 或订阅"非自己"的参与者

### 环境变量速查（Agent 可配参数）

```bash
VAD_SILENCE_MS=200       # VAD hangover ms
TTS_FRAME_MS=20          # 发布帧粒度 ms
MIN_ENDPOINTING=0.3      # LiveKit endpointing delay
ENABLE_CONTINUATION=1    # LLM 先短后长
LLM_MAX_TOKENS=150       # LLM 最大 token
LLM_TEMPERATURE=0.3
LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET  # LiveKit 连接
PYTHONPATH=/workspace/vllm-omni
```

### 启动/重启 Agent

```bash
# 一键管理
bash scripts/start_all.sh status   # 查看
bash scripts/start_all.sh restart  # 重启 Agent + Token Server
bash scripts/start_all.sh start    # 启动全部

# 手动启动 Agent（需要先 export 上面的环境变量）
cd "/workspace/project 1/25"
python3 runtime/livekit_agent.py start > /tmp/livekit_agent.log 2>&1 &
```

---

## D6 之前的工程师做了什么（D6 第一轮，已部分完成）

上一位工程师完成了 D6 的初步框架搭建：

### ✅ 已完成

1. **P0-1 AutoRTC 框架** — `tools/autortc/` 包含 `user_bot.py`, `probe_bot.py`, `run_suite.py`, `common.py`
2. **P0-2 Trace DataChannel** — Agent 新增 `_on_data_received` 监听 DataChannel，`apply_trace` 方法
3. **P0-3 音质指标** — `tools/autortc/audio_metrics.py` 实现了 dropout/jitter/clipping/rms
4. **P0-4 用例库** — `tools/autortc/cases/p0_cases.json` 定义了 12 个用例
5. **P0-5 TTS 边收边推** — `_stream_tts_frames_worker` + `asyncio.Queue` 实现了流式推帧，`tts→pub` 从 P50=1481ms 降到 ~3ms ✅

### 🔴 D6 审核发现的严重问题（必须修复）

#### 问题 1：AutoRTC 12 个 case 中 11 个录到的是静音（最严重）

```
endpoint_short_hello    rms=0.0398  ✅（唯一有声音的）
其他 11 个 case          rms=0.000004  🔴 近零静音
```

**原因**：Agent 是 `WorkerType.ROOM`，**只有第一个 case 触发了 job 并得到了回复**。后续 case 在同一个 room/session 内跑，Agent 的工作进程被第一个 job 占用，后续全静音。

**修复要求**：`run_suite.py` 必须为**每个 case 创建独立的 room**（每个 case 用不同 room name），确保每个 case 都能触发 Agent 新 job，录到真实音频。每个 case 结束后删掉 room。

#### 问题 2：LLM 没有对话历史（Agent 回复很笨）

当前 `_stream_omni()` 只发一条消息：

```python
"messages": [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": user_msg},  # ← 只有当前一句！
]
```

导致 Agent 完全没有上下文：
- 用户说 "为什么不错？" → LLM 不知道之前说了什么"不错"
- 用户说 "哪个产品啊？" → LLM 反问 "你说的是哪个产品？"

**修复要求**：把 LiveKit 的 `chat_ctx.items`（已有的聊天历史）转成 Omni 的 messages 格式传给 LLM。`_stream_omni` 方法签名改为接受完整 messages list 而不仅仅是 user_msg string。最近 N 轮（比如 10 轮）保留即可，避免 token 超限。

#### 问题 3：TTS "Response ended prematurely" 错误

日志出现：`ERROR:voice-agent:[TTS] Error: Response ended prematurely`

新的"边收边推"代码在 cancel/中断场景下，TTS HTTP 连接被提前关闭但 worker 线程还在读 → 报错。

**修复要求**：`_stream_tts_frames_worker` 中，当 `stop_event.is_set()` 或 HTTP 响应异常时，安全关闭连接并向 queue 发送 `None` sentinel，不要抛异常到主协程。

#### 问题 4：AutoRTC 测试会耗尽 Agent 进程池，导致真人连接时 Agent 不说话

**现象**：跑完 AutoRTC 后，Agent 的 4 个子进程全部 `process exiting`，但 Worker 没有自动恢复。用户连接时 JOB 无法被处理。

**修复要求**：
1. `run_suite.py` 结束后**必须**检查 Agent 健康状态，如果进程池耗尽就自动执行 `bash scripts/start_all.sh restart`
2. 或者在 `run_suite.py` 每个 case 结束后主动删掉 room（`livekit.api.DeleteRoomRequest`），让 Agent 子进程正常退出回池

#### 问题 5：Trace 大量 None 值（手动浏览器连接时）

手动连接浏览器时，没有 DataChannel 发送 trace_id，导致 trace 打点全是 None。新代码的 trace 防污染逻辑**过于严格**（要求 `t_stt_done` 存在才记录后续时间戳），连非 AutoRTC 的正常通话都受影响。

**修复要求**：当没有 DataChannel trace 时，回退到原来的"自动创建 trace"逻辑。防污染只对有 DataChannel trace 的 AutoRTC 场景生效。

#### 问题 6：用例库全用同 2 个 wav 文件，没有真实多样性

`p0_cases.json` 12 个 case 全部使用 `day1_test_input.wav` 和 `day2_reply.wav` 两个文件轮换，没有对应场景的真实音频（比如"低声"用的是正常音量 wav）。

**修复要求**：
- 短句类：用 TTS 生成短句 wav（"你好"、"嗯"、"再见"）
- 长句类：用 TTS 生成 >6s 的长段落
- 低音量：对 wav 做 volume attenuation（乘以 0.1）
- 噪声类：给 wav 叠加白噪声/粉噪声
- 打断类：在 wav 中间加入 2s 静音再继续说（模拟 barge-in 场景）
- 可以写一个 `tools/autortc/gen_cases.py` 脚本批量生成

---

## 当前需要继续完成的 D6–D10 任务

### 紧急修复（Day 6 剩余，今天必须完成）

| # | 任务 | 优先级 |
|---|------|--------|
| **F1** | LLM 加对话历史 | 🔴 P0 |
| **F2** | AutoRTC 每个 case 用独立 room | 🔴 P0 |
| **F3** | 修 TTS "Response ended prematurely" | 🟡 P1 |
| **F4** | Agent 进程池耗尽保护 | 🟡 P1 |
| **F5** | Trace 回退逻辑（非 AutoRTC 也能打点）| 🟡 P1 |
| **F6** | 用例 wav 文件多样性 | 🟡 P1 |

### D7–D10 任务（原计划继续）

#### P0-2 完善：Trace 贯穿验证

- 跑 AutoRTC 100 轮（每 case 独立 room），确认 `traces.jsonl` 中每段延迟都有有效数值（非 None）
- 输出 `output/autortc/latency_breakdown.md`，包含每段 P50/P95/P99

#### P0-3 完善：音质指标验证

- 在真实音频（非静音）上跑 `audio_metrics.py`，确认指标可靠
- 新增 Gate：`rms >= 0.01`（确保录到了真实音频，不是静音数据上的假 PASS）

#### P0-4 完善：用例库

- `tools/autortc/gen_cases.py`：批量生成不同场景的 wav 文件
- 压力测试：连续 20 turns 在同一 room（测内存泄漏/延迟漂移）
- 并发测试：2 个 room 同时跑（测 GPU 调度器）

#### 新增 P0-6：多轮对话质量回归

LLM 加了对话历史后，需要验证：
- 5 轮以上的对话中，Agent 回复是否连贯（人工抽查 3 条链路的 STT→LLM 对）
- 上下文记忆长度是否合理（最近 10 轮 vs 5 轮）
- Token 不超限（max_tokens + 历史 < context window）

---

## D5 已定位的性能数据

| 延迟段 | D5 P50 | D6 改进后 | 说明 |
|--------|--------|-----------|------|
| vad→stt | 104ms | ~100ms | ✅ 无退化 |
| llm_first→tts_first | 322ms | — | TTS TTFA |
| **tts_first→publish** | **1481ms** | **~3ms** ✅ | D6 边收边推已生效 |
| Cancel→silence | 7.5ms | — | ✅ |

---

## D1-D5 核心成果

- ✅ WebRTC 全双工通话（浏览器⇄LiveKit⇄Agent）
- ✅ VAD→STT(Omni)→LLM(Omni)→TTS 全链路
- ✅ 0 Error 运行（22 轮多轮对话）
- ✅ Cancel→silence P95=7.5ms
- ✅ TTS 回归 PASS（SNR 120dB bit-exact）
- ✅ GPU 硬优先级调度器
- ✅ 9 个时间戳端到端打点
- ✅ 持久化部署（start_all.sh + post_start.sh）

---

## 研发常见 bug 清单（写进回归）

1. TTS 流式被"收完再推" → 体感 1-2s (**D6 已修**)
2. 事件循环被同步 IO 阻塞 → 延迟尖刺 (**D5 已修**)
3. VAD/endpointer 过保守 → 反应慢 (**D5 已调**)
4. 短文本/空文本触发 CUDA assert → 崩溃 (**D3 已防御**)
5. sample rate / frame size 不一致 → 音高怪
6. publish 帧粒度太大 → 浏览器缓存才播 (**D5 已改20ms**)
7. GPU 争抢（slow lane 排队） (**D3 已做调度器**)
8. trace_id 时序错位 → 数据不可用 (**D6 部分修，仍有 None**)
9. barge-in 取消链路不完整
10. 内存泄漏/队列积压
11. **AutoRTC 跑完后 Agent 进程池耗尽** → 真人连接无回复 (**D6 新发现**)
12. **TTS 边收边推 HTTP 连接提前关闭** → "Response ended prematurely" (**D6 新发现**)
13. **LLM 无对话历史** → Agent 回复不连贯、像傻子 (**D6 新发现**)
14. **AutoRTC 在同一 room 跑多 case** → 只有第一个有音频 (**D6 新发现**)

---

## LiveKit Agent 踩坑经验（v1.4 API，之前已踩过的坑）

| 坑 | 解决方案 |
|---|---|
| `JobContext` 没有 `participant` 属性 | 用 `ctx.connect()` + `ctx.wait_for_participant()` |
| `AgentSession.start()` 不接受 `participant` | 只传 `agent` 和 `room` |
| `LLMStream.__init__()` 缺参数 | 必须传 `tools=[]` 和 `conn_options` |
| `ChatChunk` 缺 `id` 字段 | 必须传 `id="omni"` |
| `ChunkedStream._run()` 签名变化 | 必须接受 `output_emitter` 参数 |
| **AudioEmitter 必须先 initialize** | **在 `_run()` 开头就调 `initialize()`**，即使没音频也推静音帧 |
| `start_segment()` 仅限 stream=True | 非流式 ChunkedStream 不用 segment 管理 |
| Omni audio 格式 | 用 `{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}`，不是 `input_audio` |
| 同步 HTTP 阻塞事件循环 | 所有 requests.post 必须 `run_in_executor()` 或 `asyncio.to_thread()` |
| Agent identity 是动态的 | `agent-{JOB_ID}` 格式，不能硬匹配 `agent` |
| **AutoRTC 后 Agent 不响应** | Agent 进程池 (默认 4 个) 被 AutoRTC 耗尽，必须 restart |

---

**请先阅读 `PROJECT_BRIEFING.md`、`DEV_LOG.md`、`SKILL.md` 了解完整上下文，然后阅读 `runtime/livekit_agent.py` 理解当前实现，最后从紧急修复 F1（LLM 加对话历史）开始做。**

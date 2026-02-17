# D13 Cloud Agent Task: WYSIWYG 从"能跑"升级到"听即所得"

## 你是谁

你是接手一个实时语音通话项目的 AI 工程师。项目用 Qwen3-Omni + Qwen3-TTS + LiveKit 构建了一套浏览器端实时语音通话系统，并建立了自动化回归测试体系（AutoRTC + AutoBrowser）。

## 必读文件（按优先级）

1. **`PROJECT_BRIEFING.md`** — 项目全貌、架构、里程碑
2. **`SKILL.md`** — 工程 SOP、所有技术决策、踩坑经验、Gates 定义
3. **`DEV_LOG.md`** — 每个阶段的详细开发记录和验收结果
4. **`tools/autobrowser/run_suite.py`** — D12 AutoBrowser 编排器（你要改的主文件之一）
5. **`runtime/webrtc_test.html`** — 产品网页（浏览器端打点，你要改的主文件之一）
6. **`tools/autortc/audio_metrics.py`** — 指标分析 + Gates

## 当前状态（D12 已完成，D13 代码已部分修改但未验证）

D12 成果：
- AutoBrowser 16/16 cases PASS，USER_KPI P50=201ms P95=207ms
- 但 USER_KPI 数值"太整齐"——因为用了 30ms 轮询 + mic mute + resetForMeasurement 的作弊方式

D13 已改但未验证的代码：
- `run_suite.py`: `_prepare_chromium_wav()` 已替换 `_convert_wav_for_chromium()`，追加 10s 静音
- `run_suite.py`: 已移除 mic mute + resetForMeasurement 调用
- `webrtc_test.html`: `finalizeTrace()` 已改为保留 raw + clamped + is_talk_over
- `webrtc_test.html`: 轮询精度已改为 PLAYOUT_POLL_MS=5ms, MIC_POLL_MS=10ms
- `run_suite.py`: summary 已改为输出 raw/clamped/talk_over 统计

## D13 目标

### P0-1：修正 USER_KPI 口径（代码已改，需验证）

- `user_kpi_raw_ms` = 原始值（可为负，表示 talk-over）
- `user_kpi_ms` = max(0, raw)（clamped，用于 turn-taking gate）
- `is_talk_over` = raw < 0
- report.md 新增：`talk_over_count`、`talk_over_ms_p95`
- 报告顶部显示 Turn-taking KPI 和 Duplex KPI 两张表

**验收**：跑 mini 4 cases，确认 browser_trace.json 里有 raw/clamped/is_talk_over 字段

### P0-2：padded wav 替代 mic mute（代码已改，需验证）

- `_prepare_chromium_wav()` 在原始 speech 后追加 10s 静音（48kHz 零值）
- 不再调用 `setMicrophoneEnabled(false)` 和 `resetForMeasurement()`
- `monitorMic` 通过能量检测自然发现 EoT

**验收**：
- 跑 mini 4 cases，确认 `t_user_eot_browser` 来自能量自然落点（不是 mute 时刻）
- Turn-taking cases 的 raw USER_KPI 不应该全是 ~200ms（应该有更多方差）
- 不应该出现大量负值（如果出现，说明 silence padding 不够或 WAV 在静音段内 loop 了）

### P0-3：playout 检测精度（代码已改，需验证）

- `PLAYOUT_POLL_MS` 从 30ms → 5ms
- `MIC_POLL_MS` 从 30ms → 10ms
- `agentAnalyser.fftSize` 从 512 → 256（更快处理）
- 每个 trace 记录 `playout_resolution_ms` 和 `mic_resolution_ms`

**验收**：USER_KPI 值的方差应该比 D12 更大（不再全卡在 200±8ms）

### P0-4：USER_KPI gate 从 WARN 升为 FAIL-ready

- 跑 3 次 mini suite（repeat 3），收集 USER_KPI 波动数据
- 用 `tools/autortc/baseline_stability.py` 的思路对 USER_KPI 做统计（min/med/P95/P99/max/σ）
- 建议阈值：FAIL gate = baseline_P95 + 50ms
- 在 `audio_metrics.py` 里把 WARN 升级为 FAIL（或至少准备好开关）

**验收**：有 USER_KPI 的波动统计数据和建议阈值

### P1-1：听感校准 calibration_report

- 随机抽 4 个 case，对比 AutoBrowser USER_KPI vs 人工预估
- 由于无法真正人工听测，改为：对比 `user_kpi_raw_ms`（浏览器端）vs `eot_to_first_audio_ms`（probe 端，从 autortc summary）
- 输出 `output/autobrowser/calibration_report.md`

### P1-2：netem 真正生效

- 检查容器是否有 NET_ADMIN（`tc qdisc add dev eth0 root netem delay 1ms` 测试）
- 如果没有，尝试用 `toxiproxy` 或 Python socket proxy 做应用层扰动
- 如果都不行，在报告里标注清楚

## GPU 服务器远程访问（零干预执行）

你有一台远程 GPU 服务器（RunPod L40S），上面跑着完整的服务栈：
- TTS Server (:9000)
- LLM Server (vLLM)
- LiveKit Agent (:8089)
- Token Server (:9090)
- Playwright + Chromium

**通过 SSH 访问**：
```bash
ssh gpu   # 已配置在 ~/.ssh/config
```

**项目路径**：`/workspace/project 1/25/`

**在 GPU 服务器上执行命令的方式**：
```bash
# 单条命令
ssh gpu 'cd "/workspace/project 1/25" && python3 -u tools/autobrowser/run_suite.py --cases_json tools/autortc/cases/mini_cases.json --token_api http://127.0.0.1:9090/api/token --record_s 25 --p0_only 1 --output_root /tmp/d13_test --inter_case_wait_s 10 2>&1'

# 读文件
ssh gpu 'cat "/workspace/project 1/25/output/autobrowser/latest/report.md"'

# 检查服务状态
ssh gpu 'curl -s http://127.0.0.1:9090/api/token?room=health\&identity=test | head -1'
```

**重要**：你的代码修改在本地（Cloud Agent VM）的 git repo 里。要让 GPU 服务器用最新代码，需要：
```bash
# 1. 在本地 commit + push
git add -A && git commit -m "..." && git push origin main

# 2. 在 GPU 服务器上 pull
ssh gpu 'cd "/workspace/project 1/25" && git pull origin main'

# 3. 然后在 GPU 服务器上跑测试
ssh gpu 'cd "/workspace/project 1/25" && python3 -u tools/autobrowser/run_suite.py ...'
```

## 执行步骤

1. **验证 SSH 连通性**：`ssh gpu 'echo OK && hostname'`
2. **验证服务可用**：`ssh gpu 'curl -s http://127.0.0.1:9090/api/token?room=test\&identity=test | python3 -m json.tool | head -3'`
3. **在本地改代码**（如果需要修改）
4. **push 到 GitHub → SSH pull → 在 GPU 上跑测试**
5. **读取结果**：`ssh gpu 'cat /tmp/d13_test/*/report.md'`
6. **分析结果，如果有问题则修改代码重复 3-5**
7. **跑 full 16 cases**
8. **做 P0-4 stability 采样（repeat 3）**
9. **更新 report 模板、SKILL.md、DEV_LOG.md**
10. **最终 commit + push**

## 关键注意事项

- **不要**用 `requestAnimationFrame`，headless Chromium 里不触发，必须用 `setInterval`
- **不要**重新引入 mic mute + resetForMeasurement（D13 的核心就是去掉它）
- `user_kpi_ms=0` 是合法值（Python falsy 坑），必须用 `is not None` 判断
- Token server 在 GPU 服务器上 `http://127.0.0.1:9090/api/token`
- WAV 文件路径相对于 `/workspace/project 1/25/`
- 每个 case 之间等 10-15s 让 LiveKit Agent 进程池回收
- SSH 命令中路径有空格，必须用引号：`"/workspace/project 1/25/"`

## 完成后

1. 更新 `DEV_LOG.md` 添加 D13 记录（做了什么、结果、踩坑）
2. 更新 `SKILL.md` 如有新的经验教训
3. git commit + push

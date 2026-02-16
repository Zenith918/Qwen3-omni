# Week 2 · Day 11 — 工程师指令

## 你需要知道的背景（2 分钟速读）

我们有一个**实时语音通话系统**：用户浏览器通过 WebRTC 连接到 LiveKit Agent，Agent 内部跑 VAD→STT→LLM→TTS 链路，生成语音回复。整个系统跑在单张 L40S GPU 上。

过去 10 天（D1-D10），我们做了两件大事：

1. **TTS 引擎加速**（CUDA Graph）：RTF 从 1.48 降到 0.70，首音频延迟 244ms
2. **AutoRTC 自动回归系统**：一个全自动的"质量守门员"——16 个测试 case 自动跑完，8 个 gate 硬判 PASS/FAIL，不需要人耳听测

现在系统的状态是：
- Fast Suite **8/8 gates PASS**
- Nightly 20 turns **5% retry, 100% audio valid, 0 crash**
- P1 异常指纹（爆音/语速漂移/失真）**全部可验证**

**从 D11 开始，目标变了**：不再是"让回归跑通"，而是**用回归系统作为加速器，持续压缩延迟**。

---

## D11 总目标

> 把 D10 的结果"冻结"为黄金基线，测出系统自然波动范围，定义好主线优化指标。
> 这一天做的是"校准仪器"——之后每次改代码，都能用这个基线判断"变好了还是变差了"。

---

## P0-1：冻结黄金基线（~1 小时）

### 为什么要做
现在每次跑 `run_suite.py` 都会生成新的 `output/autortc/<run_id>/` 目录。但我们没有一个"官方参考点"。如果你改了代码，你怎么知道是变好了还是变差了？需要一个冻结的基线来对比。

### 具体步骤

1. **创建 golden 目录**：
```bash
mkdir -p golden/d10_baseline/
```

2. **跑一次 Fast Suite**（确认当前代码 8/8 PASS）：
```bash
cd "/workspace/project 1/25"
python3 -u tools/autortc/run_suite.py \
  --cases_json tools/autortc/cases/all_cases.json \
  --token_api http://127.0.0.1:9090/api/token \
  --output_root output/autortc \
  --ring0 0 --with_metrics 1 \
  > /tmp/d11_baseline.log 2>&1 &
```

3. **把结果固化到 golden/**：
```bash
RUN_DIR=output/autortc/<刚跑出的run_id>
# 对每个 case 保存三件套
for case_dir in $RUN_DIR/*/; do
  case_id=$(basename $case_dir)
  dst=golden/d10_baseline/$case_id
  mkdir -p $dst
  cp $case_dir/pre_rtc.wav $dst/ 2>/dev/null
  cp $case_dir/post_rtc_reply.wav $dst/ 2>/dev/null
  # 保存 reply_text（从 probe_result.json 提取）
  cp $case_dir/probe_result.json $dst/
  cp $case_dir/user_result.json $dst/
done
# 保存全局产物
cp $RUN_DIR/metrics.csv golden/d10_baseline/
cp $RUN_DIR/summary.json golden/d10_baseline/
cp $RUN_DIR/report.md golden/d10_baseline/
```

4. **在 summary.json 里标记基线版本**：
```python
# 在 summary.json 里加一个字段
import json
with open("golden/d10_baseline/summary.json", "r+") as f:
    s = json.load(f)
    s["BASELINE_VERSION"] = "D10_R4"
    f.seek(0); json.dump(s, f, indent=2, ensure_ascii=False); f.truncate()
```

### 验收标准
- [ ] `golden/d10_baseline/` 下 16 个 case 目录，每个有 pre_rtc.wav / probe_result.json
- [ ] `golden/d10_baseline/metrics.csv` + `summary.json` + `report.md` 存在
- [ ] summary.json 里有 `BASELINE_VERSION` 字段
- [ ] 8/8 gates PASS

---

## P0-2：测量自然波动区间（~3 小时，后台跑）

### 为什么要做
现在 gate 阈值是拍脑袋定的（比如 max_gap < 200ms）。但系统每次跑的结果有波动——即使代码完全没改，max_gap 可能是 120ms 也可能是 180ms。如果不知道"正常波动范围"，就无法区分"真的变差了"和"只是运气不好"。

### 具体步骤

1. **Mini 稳定性采样（5 次，~15 分钟总计）**：

用 `mini_cases.json`（4 个代表性 P0 case）而非全部 16 个，每次跑 ~3 分钟：

```bash
mkdir -p output/baseline_stability/mini_runs
for i in $(seq 1 5); do
  python3 -u tools/autortc/run_suite.py \
    --cases_json tools/autortc/cases/mini_cases.json \
    --token_api http://127.0.0.1:9090/api/token \
    --output_root output/baseline_stability/mini_runs/run_$i \
    --ring0 0 --with_metrics 1 \
    > /tmp/d11_stability_mini_$i.log 2>&1
  echo "Mini run $i done at $(date)"
done
```

> 4 cases × ~33s/case = ~2 min/run × 5 = **~10 分钟总计**。后台跑。

2. **Full 验收（1 次，仅在阶段性完成时跑）**：

```bash
# 只在冻结基线/阶段交付时跑一次完整 16 case
python3 -u tools/autortc/run_suite.py \
  --cases_json tools/autortc/cases/all_cases.json \
  --token_api http://127.0.0.1:9090/api/token \
  --output_root output/baseline_stability/full_validation \
  --ring0 0 --with_metrics 1 \
  > /tmp/d11_full_validation.log 2>&1
```

> **测试分级原则**：日常迭代用 mini（~3 min），阶段验收用 full（~15 min）。详见 SKILL.md §15。

3. **统计输出**：写一个脚本 `tools/autortc/baseline_stability.py`

输入：5 个 fast run + 3 个 nightly run 的 metrics.csv

输出：`output/baseline_stability/baseline_stability.md`

每个指标算 min / median / P95 / P99 / max：

| 类别 | 指标 | 含义 |
|------|------|------|
| **延迟** | `eot_to_first_audio_ms` | 用户说完→听到第一声 |
| **延迟** | `fast_lane_ttft_ms` | LLM 快车道首 token |
| **延迟** | `tts_first_to_publish_ms` | TTS 首帧→发布 |
| **音质** | `reply_max_gap_ms` | reply 段内最大静音间隙 |
| **音质** | `reply_audible_dropout_count` | 可听断裂次数 |
| **音质** | `clipping_ratio` | 削波比例 |
| **音质** | `mel_distance` | pre vs post mel 距离 |
| **音质** | `hf_ratio_drop` | 高频能量下降 |
| **可靠性** | `audio_valid_rate` | 有声比例 |
| **可靠性** | `retry_rate` | 需要重试的比例 |

4. **给出建议阈值**：

```
建议阈值 = median + 2σ（或 P95 + 安全余量）
```

写入 SKILL.md 的 gate 阈值建议表。

### 验收标准
- [ ] 5 次 mini run 全部完成（~15 min），metrics.csv 都存在
- [ ] 1 次 full validation 完成（~15 min），8/8 PASS
- [ ] `baseline_stability.md` 生成，每个指标有 min/median/P95/max
- [ ] SKILL.md 里更新了"建议阈值"表

---

## P0-3：定义主线优化指标（~30 分钟）

### 为什么要做
从 D11 开始，我们的目标从"让系统跑通"变成"持续优化"。需要定义清楚：**优化什么？怎么衡量？什么算"变好了"？**

### 主线指标选择

**推荐主线指标**：`eot_to_probe_first_audio_p95_ms`

含义：从用户说完最后一个字（End-of-Turn），到 probe 第一次收到 Agent 音频的时间。这是用户最直接感受到的"等了多久才听到回复"。

### 具体步骤

1. **在 `audio_metrics.py` 的 report 顶部新增 PRIMARY_KPI**：

```python
# 在 report.md 开头加一行
primary_kpi = _pct(p0_eot, 95) if p0_eot else None
f.write(f"## PRIMARY KPI\n\n")
f.write(f"- **EoT→FirstAudio P95**: `{primary_kpi}` ms\n")
f.write(f"- Baseline (D10): `{baseline_value}` ms\n")
f.write(f"- Δ: `{primary_kpi - baseline_value if primary_kpi and baseline_value else 'N/A'}` ms\n\n")
```

2. **在 summary.json 里写入**：

```json
{
  "PRIMARY_KPI_NAME": "eot_to_probe_first_audio_p95_ms",
  "PRIMARY_KPI_VALUE": 14.3,
  "PRIMARY_KPI_BASELINE": 14.3,
  "PRIMARY_KPI_DELTA_MS": 0
}
```

3. **新增一个 gate**（可选，建议做）：

```python
# 如果 PRIMARY_KPI 比 baseline 恶化超过 30ms，直接 FAIL
"PRIMARY_KPI regression <= 30ms": (primary_kpi - baseline) <= 30.0
```

### 验收标准
- [ ] report.md 顶部有 PRIMARY KPI 显示（当前值 + baseline + Δ）
- [ ] summary.json 有 PRIMARY_KPI 相关字段
- [ ] 如果 PRIMARY_KPI 恶化超过 30ms，report 里显示 FAIL

---

## 快速上手指南

### 环境准备
```bash
# 确认服务在跑
cd "/workspace/project 1/25"
bash scripts/start_all.sh status

# 如果服务没跑，启动
bash scripts/start_all.sh start
```

### 关键文件位置
| 文件 | 作用 |
|------|------|
| `tools/autortc/run_suite.py` | 测试编排器（改这个加 --repeat） |
| `tools/autortc/audio_metrics.py` | 指标分析 + gate 判定（改这个加 PRIMARY_KPI） |
| `tools/autortc/cases/all_cases.json` | 16 个测试用例定义 |
| `SKILL.md` | 工程 SOP（你的"操作手册"） |
| `DEV_LOG.md` | 研发日志（所有历史记录） |

### 跑一次测试验证环境

```bash
# 后台跑 Fast Suite（~15 分钟）
python3 -u tools/autortc/run_suite.py \
  --cases_json tools/autortc/cases/all_cases.json \
  --token_api http://127.0.0.1:9090/api/token \
  --output_root output/autortc \
  --ring0 0 --with_metrics 1 \
  > /tmp/test_run.log 2>&1 &

# 查进度（不用 sleep！）
grep -c '^\[' /tmp/test_run.log   # 已完成 case 数
tail -5 /tmp/test_run.log          # 最新输出
```

### ⚠️ 注意事项

1. **永远不要在 Cursor 工具调用里用 `sleep`**（会导致 Connection Error，详见 SKILL.md §14）
2. **长任务必须后台跑**（`> /tmp/log.txt 2>&1 &`），用即时命令查进度
3. **改代码后重启 Agent**：`pkill -f livekit_agent && sleep 5 && python3 -u runtime/livekit_agent.py start ...`
4. **Token Server 端口是 9090**（不是 3000）

---

## D11 结束后，你应该有

1. ✅ `golden/d10_baseline/` 冻结基线（16 case + metrics + summary + report）
2. ✅ `output/baseline_stability/baseline_stability.md` 波动区间报告（mini×5，~15 min 采集）
3. ✅ SKILL.md 里更新了基于统计的"建议阈值"
4. ✅ report.md 顶部有 PRIMARY_KPI 显示
5. ✅ 每次跑 suite 都能看到"比 baseline 好了/差了多少 ms"
6. ✅ `mini_cases.json` 就绪（日常迭代用，~3 min/次）

这些做完，D12 就可以开始真正优化延迟了（方向：INT8 量化、Talker 近似 Graph、音频直送 Omni 跳过 STT）。

> **时间预算提醒**：D11 全部任务应在 **2-3 小时内**完成（P0-1: 30min, P0-2: 30min 后台, P0-3: 30min）。不要超过半天。


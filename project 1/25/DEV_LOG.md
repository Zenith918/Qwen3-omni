# 研发日志

## 2026-01-29 03:26 UTC
步骤
- 删除深度流式“滑窗重解”分支，仅保留增量解码（paper 路径）。
- 增加服务端指标头：model_ttfp_ms / model_ttf_ms / server_ttfa_ms。
- 回归对齐论文口径：记录 model_ttf & e2e_ttfa，强制 paper + packet_tokens=4，写入定义说明。
- offline 超时默认提升到 600s；回归集缩为 base + 5 条（2 long、2 medium、1 short）。
- 新增 `clients/texts_p0_base.json`、`clients/voices_base.json`，更新 `run_ci_regression.sh` 默认值。
- 修复 `_synthesize_stream_deep` 里 first_chunk 变量未定义的异常。
- 跑一轮回归（deep-stream + packet_tokens=4 + left_context=25）。

结果
- 回归结果：`output/regression/20260129_032345/summary.json`
- 状态：FAIL（5/5 case 的 mae_waveform > 1e-3）
- E2E TTFA P50：364.28 ms；Model TTF P50：359.03 ms；RTF P50：1.087

## 2026-01-29 03:45 UTC
步骤
- 开启确定性生成：`TTS_DEEP_STREAM_DETERMINISTIC=1`。
- 复跑回归（同样的 base-only 5 条集）。

结果
- 回归结果：`output/regression/20260129_034340/summary.json`
- 状态：PASS（波形一致性恢复到 2.7e-05 量级）
- E2E TTFA P50：344.78 ms；Model TTF P50：339.71 ms；RTF P50：1.040

## 2026-01-29 06:40 UTC
步骤
- offline 改为直接波形生成：`TTS_DEEP_STREAM_OFFLINE_FROM_CODES=0`。
- 保持确定性与 paper 路径，复跑回归。

结果
- 回归结果：`output/regression/20260129_063825/summary.json`
- 状态：PASS
- E2E TTFA P50：347.32 ms；Model TTF P50：342.09 ms；RTF P50：1.029

## 2026-01-29 06:46 UTC
步骤
- 增大左上下文：`TTS_DEEP_STREAM_LEFT_CONTEXT=72`（同时保持 offline 直接波形）。
- 复跑回归。

结果
- 回归结果：`output/regression/20260129_064415/summary.json`
- 状态：PASS
- E2E TTFA P50：356.42 ms；Model TTF P50：350.94 ms；RTF P50：1.048

## 2026-01-29 06:47 UTC
回滚基线（用于后续排查对比）
- 启动参数：`TTS_DEEP_STREAM_DETERMINISTIC=1`，`TTS_DEEP_STREAM_OFFLINE_FROM_CODES=0`
- `TTS_DEEP_STREAM_LEFT_CONTEXT=72`，`TTS_DEEP_STREAM_PACKET_TOKENS=4`
- `TTS_STREAM_CHUNK_MS` 未设置（默认 30ms）

## 2026-01-29 07:06 UTC
步骤
- 提高包大小：`TTS_DEEP_STREAM_PACKET_TOKENS=8`（其余保持 `LEFT_CONTEXT=72`、offline 直接波形、确定性）。
- 复跑回归。

结果
- 回归结果：`output/regression/20260129_070412/summary.json`
- 状态：FAIL（回归规则要求 packet_tokens=4，且 TTFA 回归超阈）
- E2E TTFA P50：644.99 ms；Model TTF P50：638.84 ms；RTF P50：0.994
- pop_click_score P50：0.0333（相较基线无明显改善）

## 2026-01-29 07:18 UTC
步骤
- 回滚到基线参数：`TTS_DEEP_STREAM_PACKET_TOKENS=4`，`TTS_DEEP_STREAM_LEFT_CONTEXT=72`，`TTS_DEEP_STREAM_OFFLINE_FROM_CODES=0`。
- 对比爆音位置：基线（20260129_064415） vs packet_tokens=8（20260129_070412）。

结果
- 爆音位置不稳定：基线与 packet_tokens=8 的尖峰对齐率（2ms 内）约 13%。
- 结论：爆音位置随 packet size 改变，说明并非“固定音源缺陷”，更像解码过程的瞬态/数值不稳。

## 2026-01-29 07:24 UTC
步骤
- 只改 chunk 大小：`TTS_STREAM_CHUNK_MS=40`（其余保持基线）。
- 复跑回归并对比爆音位置。

结果
- 回归结果：`output/regression/20260129_072241/summary.json`（PASS，指标与基线近似）
- 爆音位置完全一致（基线 vs chunk_ms=40 对齐率 100%），说明 chunk 大小不是根因。

## 2026-01-29 07:46 UTC
步骤
- 关闭 deep-stream（`TTS_DEEP_STREAM_ENABLE=0`），尝试生成 non-deep offline。
- long_03（read timeout=600）与 medium_01（read timeout=300）均超时。

结果
- non-deep offline 太慢，无法获得对比音频。
- 记录：`output/debug/non_deep_long_03.wav` 未生成。

## 2026-01-29 07:49 UTC
步骤
- 回滚到基线 deep-stream。
- 生成爆音时间戳清单用于人工比对。

结果
- `output/debug/pop_times_long03_stream.txt`
- `output/debug/pop_times_long03_offline.txt`

## 2026-01-29 08:00 UTC
步骤
- 对比“直接波形 offline”（20260129_064415）与“codes→decode offline”（20260129_034340）的爆音位置。
- 同时对比 stream vs offline 的爆音位置对齐度。

结果
- 爆音位置高度一致：code_offline vs direct_offline 对齐率约 95.7%（2ms 内）。
- direct_offline vs stream 对齐率 100%。
- 结论：爆音来自模型输出本身（code 生成/codec 输出），不是 streaming/packet/chunk 造成。

## 2026-01-29 18:58 UTC
步骤（P0）
- 为 deep-stream 增加 codes 落盘与 hash：`TTS_CODE_DUMP_ENABLE=1` 时保存 `codes_*.pt` 与 `meta_*.json`。
- 固定 `seed=42`、`deterministic=1`，同文本/voice 各跑 3 次：
  - packet_tokens=4：`output/code_dumps/manifest_packet4.json`
  - packet_tokens=8：`output/code_dumps/manifest_packet8.json`

结果（P0）
- 同配置 hash 100% 一致，跨配置 hash 也一致：
  - sha256=`113984ff57fc128b702a07f2b1ad688aeb8b59a4f01bbb4cc583f8f09c5f7f4f`
- 说明：codes 生成可复现，且与 packet_tokens 无关（在 deterministic + 固定 seed 下）。

## 2026-01-29 18:58 UTC
步骤（P1/P2）
- 使用同一份 codes 做三种解码：
  - A TRUE-OFFLINE：`tokenizer.decode(...)` 全量一次性 decode
  - B CURRENT-STREAM：`decode_streaming` + left_context=72 + packet_tokens=4
  - C WINDOW-ABLATION：`decode_streaming` + left_context=0
- 输出文件：
  - `output/debug/codes_1769711907843_aaca9fb4_A_full.wav`
  - `output/debug/codes_1769711907843_aaca9fb4_B_stream.wav`
  - `output/debug/codes_1769711907843_aaca9fb4_C_ctx0.wav`
  - `output/debug/codes_1769711907843_aaca9fb4_metrics.json`
  - `output/debug/codes_1769711907843_aaca9fb4_pop_code_stats.json`

结果（P1）
- pop_click_score：A=0.03285，B=0.03430，C=0.03568（数量级接近）。
- 爆音位置对齐率（2ms）：
  - A→B 约 10.6%
  - A→C 约 10.9%
  - B→C 约 90.6%
- 初步结论：当前 streaming-style 解码与 true-offline 的爆音位置不一致，B/C 高度一致，指向“解码窗口/拼接策略”对爆音位置有强影响。

结果（P2）
- 12Hz frames_per_sec=12.5，已输出 top-20 爆音点的 code 跳变统计：
  - `output/debug/codes_1769711907843_aaca9fb4_pop_code_stats.json`
- 全局 code jump 分布（|c[t]-c[t-1]|，跨 codebook 汇总）：p95≈1570，p99≈1822，max≈2032。

## 2026-01-29 20:51 UTC
步骤（缓存增量解码 PoC）
- 阅读 tokenizer/decoder 代码，确认 transformer 层支持 past_key_values，但 decode_streaming 未暴露。
- 实现最小 PoC：在 transformer 使用 cache（past_key_values），conv/upsample 保持 windowed。
- 脚本：`clients/tts_cached_decode_poc.py`
  - 输出：full / cached incremental / current windowed 三个 wav
  - 评估：pop_click_score + decode_ms（仅 decoder 侧）
- 使用同一份 codes：`output/code_dumps/codes_1769711907843_aaca9fb4.pt`

结果（PoC）
- 输出文件：
  - `output/debug/codes_1769711907843_aaca9fb4_A_full.wav`
  - `output/debug/codes_1769711907843_aaca9fb4_B_cached.wav`
  - `output/debug/codes_1769711907843_aaca9fb4_C_windowed.wav`
  - `output/debug/codes_1769711907843_aaca9fb4_cached_metrics.json`
- 指标（pop_click_score / decode_ms）：
  - A_full: 0.03285 / 607ms
  - B_cached: 0.03429 / 1472ms
  - C_windowed: 0.03430 / 1643ms
- 结论：cached incremental（仅 transformer 缓存）比 current windowed 略快（~10%），爆音指标相近。

## 2026-01-29 21:54 UTC
步骤（M1：stateful conv/upsample streaming）
- 新增 stateful 流式解码器：`clients/tts_incremental_decoder.py`
  - 1D causal conv streaming（buffer=kernel_size-1）
  - transposed conv streaming（k=s 直接；k=2s 用前一帧配对输出）
  - 状态结构：transformer cache/context + conv buffers + trans_prev
- 更新 PoC 脚本：`clients/tts_cached_decode_poc.py`
  - A_full_true：一次性 full decode（不走 chunked_decode）
  - B_incremental：stateful conv/upsample + packet=4
  - C_windowed：现有 windowed re-decode
  - 输出 `output/debug/codes_1769711907843_aaca9fb4_incremental_metrics.json`

结果（M1）
- A_full_true vs B_incremental（packet=4）：
  - pop_click_score：0.03286 vs 0.03292
  - MAE：0.000238（达标）
- B_incremental vs C_windowed：
  - pop_click_score 有所下降（0.03292 < 0.03430）
- 修正点：去掉“每包实时裁剪”，改为“流式输出 + 结束后裁剪尾部”，避免中途截断导致 MAE 偏大。

## 2026-01-29 22:18 UTC
步骤（M2 接入准备）
- 将 stateful incremental 解码接入 `/tts/stream` deep 路径（默认关闭）：
  - 新环境变量：`TTS_DEEP_STREAM_INCREMENTAL=1`
  - transformer 模式可选：`TTS_DEEP_STREAM_INCREMENTAL_TRANSFORMER=cache|window|full`
  - holdback 保护尾部：`TTS_DEEP_STREAM_INCREMENTAL_HOLDBACK`（默认用 decode_upsample_rate）
- 流式输出策略：
  - 每 packet 调用 `decode_incremental`，累计 tail
  - 仅输出 `expected_samples - holdback`，结束后 flush 尾部
  - 断连取消时不 flush

结果（M2）
- 代码已接入，默认不影响现有线上（需显式启用 env）。
- 头信息新增：`X-Deep-Decode-Mode`（windowed / incremental:mode）。

## 2026-01-29 22:52 UTC
步骤（M2 回归对齐）
- offline 端改为复用 streaming 路径收集 codes：
  - `_generate_audio_np_deep` 在 incremental 模式下走 `_collect_codes_streaming`
  - 取消 incremental 模式下 offline 的尾部静音裁剪（避免长度差）
- streaming 端 max_new_tokens 对齐 offline 的 cap（`_estimate_max_frames`）
- 增加 offline codes dump（便于与 streaming 对比）

结果（M2 回归）
- base-only 5 条回归仍 FAIL，MAE P50≈0.009，duration_diff 已归零。
- 结论：差异来自 codes 不一致（非解码器）。

验证（codes hash）
- 同文案、同设置、deterministic=1 下，stream vs offline codes hash 不一致。
- 同文案多次 stream 也不一致（差异 16 elements，通常为 1 帧）。
- 推断：code 生成仍有 GPU 非确定性（cuBLAS/GEMM 相关），需更强确定性配置或复现同 codes 机制。

## 2026-01-29 23:00 UTC
步骤（确定性开关验证）
- 新增 `TTS_DEEP_STREAM_DETERMINISTIC_STRICT=1`：启用 `torch.use_deterministic_algorithms` + 禁 TF32。
- 结合 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 测试：
  - stream 连续两次 codes hash 完全一致；
  - stream 与 offline codes hash 一致。
- 但在流式回归中触发 CUDA device-side assert（indexSelect），导致 /tts/stream 断流、/synthesize 500。

结论
- 严格确定性可消除 codes 漂移，但当前实现会触发 CUDA assert，不可用于回归/线上。

## 2026-01-29 23:03 UTC
步骤（严格确定性只用于 codegen）
- 增加 `TTS_DEEP_STREAM_DETERMINISTIC_STRICT_DECODER`，尝试仅在 codegen 进程启用 strict。
- 仍出现 /tts/stream 断流（CUDA device-side assert），说明 strict 触发问题并未消除。

结论
- strict 模式目前不可用（无论作用于 codegen 还是 decoder），需另寻确定性方案。

## 2026-01-30 01:28 UTC
步骤（soft 确定性 & code dump 增强）
- 增加 soft 级确定性开关（不启用 `use_deterministic_algorithms`）：
  - `TTS_DEEP_STREAM_DETERMINISTIC_SOFT`
  - `TTS_DEEP_STREAM_DETERMINISTIC_SOFT_DECODER`
- codes dump 增加 `min_code/max_code/out_of_range_count/codebook_size` 统计。
- 增加 `TTS_DEEP_STREAM_CODEGEN_DEVICE`（可让 codegen 走 CPU，仅用于回归/排查）。

## 2026-01-30 03:06 UTC
步骤（确定性实测 & 纠错）
- 修正 `tts_server.py` 末尾多余 `port)` 语法错误。
- worker 改为每次请求前重新设定随机种子；模型走 eval（`model.model.eval()`）。
- 新增 `TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD` 选项（控制 codegen 单线程）。
- 新增 `TTS_DEEP_STREAM_VALIDATE_CODES` / `TTS_DEEP_STREAM_CLAMP_CODES`（越界检测/保护）。

结果（P0 验证）
- CPU codegen + single-thread：stream 两次 hash 一致（deterministic OK），但 offline hash 仍与 stream 不一致。
- strict(GPU) + clamp：stream 两次 hash 一致（deterministic OK），offline hash 仍与 stream 不一致。

结论
- 目前 stream 内部可稳定复现，但 offline 与 stream 仍存在“同文不同码”的差异，需继续定位 codegen 路径差异或改用同一份 codes 对齐回归。

## 2026-02-01 06:10 UTC
Q1–Q30 关键结论（自解释版）
说明：以下结论来自技术总监 Q1–Q30 的“只问问题”清单，目标是确定 drift（抖动）来源与并行 overlap 的影响链条。每条都给出“结论 + 证据位置/文件”。

### Q1–Q3（进程/并行形态）
- Q1：`TTS_DEEP_STREAM_PROCESS=1` 是**常驻 worker 进程**（不是每请求新进程）。
  - 证据：`clients/tts_server.py` 中 worker 初始化逻辑（spawn + 常驻队列）。
- Q2：multiprocessing start method 为 **spawn**。
  - 证据：`clients/tts_server.py` 内 `mp.get_context("spawn")` 使用。
- Q3：codegen worker 与 decoder 默认在**同一块 GPU**，且并行 overlap。
  - 证据：server 启动 env（`TTS_DEEP_STREAM_CODEGEN_DEVICE` 默认同 `TTS_DEEP_STREAM_DEVICE`）；
    `TTS_DEEP_STREAM_TRACE_TIMING` 日志显示 overlap（见 2026-01-31 03:45 UTC 记录）。

### Q4–Q12（漂移形态与确定性）
- Q4：`deterministic=1` 并不等于 greedy；策略由 `TTS_DEEP_STREAM_DETERMINISTIC_POLICY` 决定（seeded 仍采样）。
  - 证据：`clients/tts_server.py` 中 deterministic policy 分支 + DEV_LOG 2026-01-30 06:10。
- Q5：多进程漂移是**同文本两次请求 hash 不同**，且首个 diff 很早（常见 frame=4）。
  - 证据：code dump meta + 后续 diff 统计（见 Q33/Q34 与 2026-01-31 03:45）。
- Q6：漂移时**未出现 out_of_range/clamp**；min/max/out_of_range_count 一致。
  - 证据：`output/code_dumps/meta_*.json` 的 `min_code/max_code/out_of_range_count`。
- Q7/Q8：strict determinism 会触发 CUDA assert（indexSelectSmallIndex），**长文本更易触发**。
  - 证据：server Traceback + `TTS_DEEP_STREAM_TRACE_POS` 日志（2026-01-31 03:45）。
- Q9：process=0 无 overlap 变慢，TTFA 明显上升（见 Q31/Q32）。
  - 证据：`output/debug/q31_process0_packet4.json`。
- Q10：process=0 pop_click_score 变高并非 incremental decoder 本体退化。
  - 证据：同 codes 的增量解码 PoC（`clients/tts_cached_decode_poc.py` 结果）。
- Q11：流式路径实际使用的 decode 模式可从 `X-Deep-Decode-Mode` 头确认（windowed / incremental:cache|window|full）。
  - 证据：`clients/tts_server.py` 输出 header 逻辑。
- Q12：漂移“从很早开始”的结论复现（frame≈4 对应 ~0.32s）。
  - 证据：code diff 统计与 Q33/Q34 结果。

## 2026-01-31 00:22 UTC
步骤（定位“多进程 vs 单进程”）
- 关闭 codegen worker：`TTS_DEEP_STREAM_PROCESS=0`（同进程线程生成）
- 保持 `deterministic=1` + `seed_mode=content`，跑 long_03 两次对比 hash

### Q28–Q30（漂移方向）
- Q28：overlap 条件下**codes hash 会变化**，音频变化与 codes 漂移高度相关。
  - 证据：Q33/Q34 的 `hash_unique>1` 与 `first_diff_frame`。
- Q29：logit 层面已观察到 tie/极小差异，提示数值扰动可翻转。
  - 证据：Q20/Q23 记录（top1/top2 gap=0 或 Δlogit=0）。
- Q30：漂移常在**极早帧出现**（frame≈4 或更早）。
  - 证据：Q33/Q34 `first_diff_frame` 统计。

## 2026-02-01 06:30 UTC
背景（Q31–Q36 裁决实验）
- 目标：用一组“强控制变量”的实验，回答以下关键问题：
  - Q31：无 overlap（process=0）下 TTFA 的真实下限是多少？
  - Q32：packet_tokens=2/1 是否能在无 overlap 下接近 350ms？
  - Q33：overlap 是否在“首包之后再开启”就能稳定（hybrid overlap）？
  - Q34：overlap 漂移到底由 decoder 的哪一段触发（pre_transformer vs conv/upsample）？
  - Q35：漂移是否对数值精度敏感（bf16 vs fp32）？
  - Q36：logit 层面最早差异出现在 step 0–100 的哪个位置？（未完成）
- 关键判据：
  - `hash_unique > 1` 代表同文多次请求 codes 不一致（抖动风险）。
  - `first_packet_hash_unique` 代表首包是否稳定（用 `tmp_codes_analysis.py` 对比）。

步骤（工具与 SOP）
- 启动服务：`clients/tts_server.py` + 指定环境变量（见 `SKILL.md` SOP）。
- 预热：`python3 clients/tts_codes_dump.py --text-id long_03`
- 统计：`python3 tmp_ttfa_runs.py --text-id long_03 --count N --out ...`
- 首包/首 diff 分析：`python3 tmp_codes_analysis.py --packet-tokens 4 --compare --tags <tag1> <tag2>`
- 新增脚本：
  - `tmp_codes_analysis.py`（full hash + first packet hash + first_diff_frame）
- 修复脚本：
  - `tmp_ttfa_runs.py` 增加 `meta_missing`/`fail_count`，避免 code dump 缺失导致 hang

结果（Q31：无 overlap 基线）
- 环境：`process=0` + `packet_tokens=4` + `left_context=72` + `incremental` + deterministic
- 输出：`output/debug/q31_process0_packet4.json`
- TTFA P50/P95：660.46 / 695.84 ms
- code_ms P50/P95：315.62 / 338.31 ms
- decode_first_ms P50/P95：341.58 / 380.70 ms
- RTF P50/P95：1.567 / 1.597
- hash_unique=1（稳定）
- 结论：无 overlap + packet=4 远高于 350ms

结果（Q32：packet_tokens=2/1）
- packet=2
  - 输出：`output/debug/q32_process0_packet2_v2.json`
  - TTFA P50/P95：349.82 / 401.00 ms
  - code_ms P50/P95：161.02 / 173.86 ms
  - decode_first_ms P50/P95：188.90 / 227.91 ms
  - RTF P50/P95：1.556 / 1.647
  - hash_unique=1（稳定）
  - 结论：P50 接近 350ms，但 P95 仍明显超标
- packet=1（异常）
  - 输出：`output/debug/q32_process0_packet1.json`
  - TTFA P50/P95：40033.99 / 92013.27 ms（40–90s）
  - code_ms P50/P95：39276.54 / 91828.40 ms
  - decode_first_ms P50/P95：187.53 / 271.79 ms
  - RTF P50/P95：7.11 / 12.88
  - hash_unique=23（明显漂移）
  - 伴随现象：frames≈135、音频仅 10–12s（显著短于正常 22s）
  - 结论：packet=1 路径当前不可用（需要单独排查）

结果（Q33：overlap 形态 A/B）
- 方案 A（立即 overlap）：`TTS_DEEP_STREAM_PREFILL_PACKETS=0`
  - 输出：`output/debug/q33A_process1_packet4.json`
  - TTFA P50/P95：651.79 / 1138.02 ms
  - RTF P50/P95：3.24 / 4.14
  - hash_unique=25（漂移）
  - 首包分析：`first_packet_hash_unique=2`，`first_diff_frame=0`
  - 结论：首包即漂移
- 方案 B（首包后 overlap）：`TTS_DEEP_STREAM_PREFILL_PACKETS=1`
  - 输出：`output/debug/q33B_process1_prefill1_packet4.json`
  - TTFA P50/P95：459.82 / 476.02 ms
  - RTF P50/P95：1.055 / 1.079
  - hash_unique=30（漂移仍在）
  - 首包分析：`first_packet_hash_unique=1`，`first_diff_frame=8`
  - 结论：prefill 能“推迟”漂移，但不能消除漂移

结果（Q34：触发源定位，dummy decoder）
- B0 codegen-only：稳定（已有结论，hash 不漂移）
- B1 pre_transformer only：稳定
  - `full_hash_unique=1`，`first_diff_frame=-1`
- B2 conv/upsample only：漂移
  - `full_hash_unique=2`，`first_diff_frame=4`
- B3 full decoder：漂移
  - `full_hash_unique=2`，`first_diff_frame=16`
- 结论：漂移触发更偏向 conv/upsample 路径，不是 pre_transformer

结果（Q35：精度敏感性，overlap）
- D0（bf16）：`output/debug/q35_d0_bf16.json`
  - TTFA P50/P95：463.54 / 490.97 ms
  - hash_unique=20（全部漂移）
- D1（codegen fp32）：`output/debug/q35_d1_codegen_fp32.json`
  - TTFA P50/P95：468.22 / 489.83 ms
  - hash_unique=20（漂移）
- D2（decoder fp32）：`output/debug/q35_d2_decoder_fp32.json`
  - TTFA P50/P95：437.87 / 452.87 ms
  - hash_unique=20（漂移）
- D3（codegen+decoder fp32）：`output/debug/q35_d3_both_fp32.json`
  - TTFA P50/P95：440.73 / 459.36 ms
  - hash_unique=20（漂移）
- 结论：漂移对 FP32 不敏感，更像并行/调度级非确定性

状态（Q36）
- 已完成（step 0–100）：
  - 使用 `TTS_CODEGEN_DEBUG_TOPK=1`，step 0–100 记录 top1/top2
  - `tmp_topk_diff.py` 结果：两次 run **step 0–100 无任何 top1/top2 差异**
    - req_id=6 vs 7（seed 相同、text_hash 相同）
  - 但同一对请求的 codes 仍漂移：
    - tags: `1769969064722_5f6ce504` vs `1769969088218_768ef956`
    - `first_diff_frame=4`（`tmp_codes_analysis.py`）
  - 结论：logit topK（0–100）层面未见差异，漂移可能发生在更深的采样/后处理环节，
    或差异小于当前 topK 打印精度（需后续扩大日志精度或记录完整 logits）。

## 2026-02-01 07:30 UTC
步骤（为 Q37–Q43 增加最小实验开关）
- 新增 cuDNN/TF32 精细开关（用于 Q37）：
  - `TTS_CUDNN_BENCHMARK`
  - `TTS_CUDNN_DETERMINISTIC`
  - `TTS_CUDNN_ALLOW_TF32`
  - `TTS_CUDA_MATMUL_ALLOW_TF32`
  - `TTS_CUDNN_TRACE=1` 时打印 cudnn enabled/available/version
- 新增 dummy 解码模式（用于 Q38b）：
  - `TTS_DEEP_STREAM_DUMMY_DECODER=noop`（保留 pre_transformer，跳过 conv/upsample）
  - 位置：`clients/tts_incremental_decoder.py`

结论
- 以上改动仅用于裁决实验，不改变默认主路径行为。

## 2026-02-01 08:40 UTC
结果（Q37：conv/upsample-only + overlap=true，10 runs each）
固定条件：
- 文本：short_01 + long_03
- greedy：`TTS_DEEP_STREAM_DETERMINISTIC_POLICY=greedy`
- 固定 seed：`TTS_DEEP_STREAM_SEED_MODE=fixed` + `TTS_DEEP_STREAM_SEED=42`
- decoder：`TTS_DEEP_STREAM_DUMMY_DECODER=conv_only`

### Q37 baseline（默认 flags）
- long：`output/debug/q37_baseline_long.json`
  - TTFA P50/P95：435.06 / 464.38 ms
  - hash_unique=10，first_diff_frame=12
- short：`output/debug/q37_baseline_short.json`
  - TTFA P50/P95：430.97 / 443.45 ms
  - hash_unique=5，first_diff_frame=4

### Q37.benchmark=False
- long：`output/debug/q37_benchmark0_long.json`
  - TTFA P50/P95：435.36 / 441.37 ms
  - hash_unique=10，first_diff_frame=4
- short：`output/debug/q37_benchmark0_short.json`
  - TTFA P50/P95：426.16 / 445.62 ms
  - hash_unique=4，first_diff_frame=-1（首两次未发现差异，但整体仍漂移）

### Q37.cudnn_deterministic=True（benchmark=False）
- long：`output/debug/q37_cudnn_deterministic_long.json`
  - TTFA P50/P95：440.42 / 457.80 ms
  - hash_unique=10，first_diff_frame=4
- short：`output/debug/q37_cudnn_deterministic_short.json`
  - TTFA P50/P95：434.14 / 455.40 ms
  - hash_unique=7，first_diff_frame=4

### Q37.TF32 off（cudnn det + benchmark=0 + allow_tf32=0）
- long：`output/debug/q37_tf32off_long.json`
  - TTFA P50/P95：23572 / 24079 ms
  - hash_unique=10，first_diff_frame=8
- short：`output/debug/q37_tf32off_short.json`
  - TTFA P50/P95：1434.93 / 1639.13 ms
  - hash_unique=7，first_diff_frame=4

结论（Q37）
- cudnn benchmark/deterministic 的切换**不消除漂移**（hash_unique 仍 >1）
- 关闭 TF32 会显著拉长 TTFA（到秒/十秒级），但**漂移仍在**

## 2026-02-01 09:40 UTC
结果（Q38：decoder 路径归因）
Q38a：decoder 强制 CPU（codegen 仍在 GPU）
- 启动参数：`TTS_DEEP_STREAM_DEVICE=cpu`, `TTS_DEEP_STREAM_CODEGEN_DEVICE=cuda:0`
- 两次请求 tags：
  - `1769980005765_48b08abd`, `1769980131382_af8a1584`
- `tmp_codes_analysis.py`：
  - full_hash_unique=2（整体 codes 不同）
  - first_packet_hash_unique=1
  - first_diff_frame=-1（前缀一致，差异可能在尾部长度）
- 结论：decoder CPU 并未阻止 codes 漂移（codes 仍变化）

Q38b：decoder 在 GPU，但 conv/upsample 置为 noop
- 启动参数：`TTS_DEEP_STREAM_DUMMY_DECODER=noop`
- 两次请求 tags：
  - `1769981265768_e9b633b0`, `1769981290898_6b79f962`
- `tmp_codes_analysis.py`：
  - full_hash_unique=1（codes 完全一致）
  - first_diff_frame=-1
- 结论：**conv/upsample 路径参与时才触发漂移**；noop 时漂移消失


## 2026-02-02 00:05 UTC
结果（Q39：显式同步）
- 配置：`TTS_DEEP_STREAM_SYNC_MODE=sync`（process=0，同进程同步）
- long_03：`output/debug/q39_sync_long.json`
  - TTFA P50/P95：608.998 / 665.234 ms
  - hash_unique=1（稳定）
- short_01：`output/debug/q39_sync_short.json`
  - TTFA P50/P95：633.263 / 649.975 ms
  - hash_unique=1（稳定）
- 备注：目前仅验证显式同步（sync），尚未实现“独立 stream + event wait”的 Q39b

## 2026-02-02 00:52 UTC
结果（Q37.1：cudnn 可用性）
- 环境输出：
  - cudnn_enabled=True, cudnn_available=True, cudnn_version=91002
  - cudnn_benchmark=False, cudnn_deterministic=False, cudnn_allow_tf32=True
  - conv1d_out=(1, 32, 98), convtranspose1d_out=(1, 8, 202)
- 结论：cudnn 可用且 conv1d/convtranspose 能在 GPU 上正常执行

## 2026-02-02 00:10 UTC
结果（Q40：降低 decoder 触发频率）
- 配置：`TTS_DEEP_STREAM_DECODE_EVERY=2`，`TTS_DEEP_STREAM_DUMMY_DECODER=conv_only`
- long_03：`output/debug/q40_decode_every2_long.json`
  - TTFA P50/P95：849.11 / 1244.12 ms
  - hash_unique=10，first_diff_frame=8
- short_01：`output/debug/q40_decode_every2_short.json`
  - TTFA P50/P95：2062.65 / 2261.32 ms
  - hash_unique=3，first_diff_frame=8
- 结论：降低解码频率会推迟漂移（diff_frame=8），但仍不能消除漂移且 TTFA 明显变差

## 2026-02-02 00:25 UTC
结果（Q41：packet=1 异常定位，packet_trace）
- 配置：`process=0`, `packet_tokens=1`, `TTS_DEEP_STREAM_PACKET_TRACE=1`
- 输出：`output/debug/q41_packet1_long.json`
- meta（tag: `1769991705617_00de68a5`）：
  - code_ms=20464.94 ms, decode_first_ms=152.29 ms, ttfa_ms=20887.52 ms
  - queue_wait_ms=11680.29, decode_calls=141
  - codes_frames_max=1, pcm_samples_total=270165, pcm_samples_max=1920
- meta（tag: `1769991760070_2c691ff4`）：
  - code_ms=43753.83 ms, decode_first_ms=204.50 ms, ttfa_ms=44407.66 ms
  - queue_wait_ms=13692.64, decode_calls=141
  - codes_frames_max=1, pcm_samples_total=270165, pcm_samples_max=1920
- 结论：packet=1 时主要卡在 codegen（code_ms 20–44s），decode 本身很快

## 2026-02-02 00:38 UTC
结果（Q42：packet=1 codegen-only）
- 使用脚本：`tmp_codegen_only_stream.py`（不走 decoder）
- 输出：`output/debug/q42_codegen_only_packet1.json`
- long_03：
  - run0：first_packet_ms=652.57, total_ms=23851.11, frames=305
  - run1：first_packet_ms=83.96, total_ms=23233.24, frames=305
  - hash_unique=1（codes 稳定）
- 结论：codegen-only 没有出现 40–90s 卡死，说明 packet=1 异常更可能来自解码/拼接链路

## 2026-02-02 00:44 UTC
结果（Q43：decode-only，packet=1 喂固定 codes）
- 固定 codes：tag `1769915670616_7828eb21`（来自 packet=2 稳定 run）
- 输出（内联脚本）：
  - frames=278, upsample=1920
  - samples_total=533205, samples_expected=533760（短 555）
  - samples_max=1920, samples_min=1365
  - bad_len_calls=1, zero_calls=0
  - decode_ms_sum=12732.40, decode_ms_p50=44.13
- 结论：packet=1 decode 基本线性，但存在尾部长度不足的单次输出（需继续确认拼接逻辑）

## 2026-02-02 01:45 UTC
决策表
- 输出：`output/debug/decision_table.md`

## 2026-02-02 06:05 UTC
结果（Q39b：CUDA event 最小同步）
- 代码改动（可回滚，默认关闭）：
  - `TTS_DEEP_STREAM_SYNC_MODE=event` 时在 codegen 侧记录 CUDA event，并在 decoder 侧单独 stream 等待
  - 仅对 `process=0`（同进程）有效；跨进程无法共享 CUDA event
- 配置：
  - `process=0`, `packet_tokens=4`, `left_context=72`
  - `TTS_DEEP_STREAM_INCREMENTAL=1`
  - `TTS_DEEP_STREAM_DETERMINISTIC=1`, `policy=greedy`, `seed=42`
  - `TTS_DEEP_STREAM_SYNC_MODE=event`
- long_03：`output/debug/q39b_event_long.json`
  - TTFA P50/P95：620.908 / 628.707 ms
  - code_ms P50/P95：290.315 / 296.978 ms
  - decode_first_ms P50/P95：330.315 / 334.981 ms
  - RTF P50/P95：1.425 / 1.437
  - hash_unique=1
  - `tmp_codes_analysis.py`：first_diff_frame=-1（稳定）
- short_01：`output/debug/q39b_event_short.json`
  - TTFA P50/P95：618.962 / 686.446 ms
  - code_ms P50/P95：289.860 / 310.106 ms
  - decode_first_ms P50/P95：327.882 / 376.195 ms
  - RTF P50/P95：1.557 / 13.794（存在 1 次异常长尾）
  - hash_unique=1
  - `tmp_codes_analysis.py`：first_diff_frame=-1（稳定）
- 异常细节（需留意）：
  - short_01 的 `idx=1`（tag `1770011831984_8c8bf955`）`total_s=15.13s` 导致 RTF P95 异常偏高，疑似队列等待或短暂 stall
- 结论：event 同步在同进程下可消除漂移；TTFA 与 sync 模式接近，但不适用于 `process=1` 跨进程并发

## 2026-02-02 06:12 UTC
结果（Q37.1：cudnn profiler 佐证）
- 输出：`output/debug/q37_1_cudnn_profile.json`
- profiler 关键项：
  - `aten::cudnn_convolution`
  - `aten::cudnn_convolution_transpose`
- 结论：conv1d/convtranspose1d 在当前环境走 cudnn

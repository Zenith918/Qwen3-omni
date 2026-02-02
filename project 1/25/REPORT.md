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
- Starter 缓存：启动时预生成 5 个 starter PCM，命中时首包 < 0.1s（`X-Cache-Starter=true`）
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
  - cold TTFA P50/P95: **1.125s / 1.293s**（n=5, fail=0）
  - cold 总耗时 P50/P95: **3.573s / 3.790s**
  - cold RTF P50/P95: **0.945 / 0.959**
  - warm TTFA P50/P95: **1.129s / 1.348s**（n=20, fail=0）
  - warm 总耗时 P50/P95: **3.269s / 3.914s**
  - warm RTF P50/P95: **0.931 / 0.970**
  - 头信息：`x-sample-rate=24000`, `x-chunk-ms=30`, `x-warm-request`, `x-segments=2`
  - 备注：已避开 LLM 加载并发
- Bridge（`bridge_demo.py`）
  - 首包可播放音频 P50/P95: **0.006s / 0.009s**（n=30, fail=0，starter 缓存命中）
  - 总时长 P50/P95: **3.257s / 5.529s**
  - 音频总时长: **3.246s**（avg）
  - chunk 数 & flush 触发统计: **chunks=16.00, flush_punct=6.00, flush_len=9.00, starter=1.00**
  - 反压：`queue_len_peak=1`，`bp_overwrite>0`（丢弃中间段，只保留最新 main）
  - 备注：当前为“分段生成 + 分块回传”，非增量 codes/解码

## F) Deep Streaming PoC（12Hz / 4 tokens=320ms）
- PoC 脚本：`/workspace/project 1/25/clients/tts_stream_poc.py`
- /tts/stream 深度模式：`TTS_DEEP_STREAM_ENABLE=1`（增量 codes → 增量 PCM）
- Packet size：`4 tokens`（≈333ms @12Hz）
- /tts/stream 深度模式 TTFA（n=5）：
  - P50: **0.377s**
  - P95: **0.380s**
- t_req_in → t_first_pcm_out（n=5）：
  - P50: **0.370s**
  - P95: **0.935s**
- 一致性：stream vs offline 拼接 MAE ~`3e-4`（无明显错音）
- 备注：已切换为“滑窗重解 + overlap 拼接”（window packets 可配）

## G) Deep-stream 工程化(B1)
- CI 回归脚本：`/workspace/project 1/25/scripts/run_ci_regression.sh`
- 固定测试集：`/workspace/project 1/25/clients/texts.json`
- 回归产物：`/workspace/project 1/25/output/regression/latest/`（offline/stream wav + metrics）
- 指标口径：TTFA P50/P95、RTF、MAE/SNR、pop_click_score、duration_diff_ms
- 取消回收：深度流式使用进程化 worker，断连时终止并重启
- 回归结果：由 `tts_regression_suite.py` 自动追加一行

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
- 深度流式：/tts/stream 深度模式已接入（增量 codes + 4-token packet）
  - 已切换“滑窗重解 + overlap”，`TTS_DEEP_STREAM_WINDOW_PACKETS` 可配
  - 断连时使用进程化 worker 强制终止并重启（`TTS_DEEP_STREAM_PROCESS=1`）
- 回归 20260127_213736: FAIL TTFA_P50=360.83ms TTFA_P95=369.30ms RTF_P50=1.493 MAE_P50=0.071137 SNR_P50=-1.00dB
- 回归 20260127_224522: FAIL TTFA_P50=355.08ms TTFA_P95=368.29ms RTF_P50=0.143 MAE_P50=0.058855 SNR_P50=-1.00dB
- 回归 20260127_230555: FAIL TTFA_P50=-1.00ms TTFA_P95=-1.00ms RTF_P50=-1.000 MAE_P50=-1.000000 SNR_P50=-1.00dB
- 回归 20260127_230555: FAIL TTFA_P50=-1.00ms TTFA_P95=-1.00ms RTF_P50=-1.000 MAE_P50=-1.000000 SNR_P50=-1.00dB
- 回归 20260127_225655: FAIL TTFA_P50=108718.52ms TTFA_P95=108718.52ms RTF_P50=46.710 MAE_P50=0.041300 SNR_P50=-1.00dB
- 回归 20260128_064849: FAIL TTFA_P50=361.70ms TTFA_P95=370.07ms RTF_P50=1.876 MAE_P50=0.050472 SNR_P50=-1.63dB
- 回归 20260128_075323: FAIL TTFA_P50=361.79ms TTFA_P95=374.43ms RTF_P50=1.013 MAE_P50=0.057515 SNR_P50=-2.39dB
- 回归 20260128_153853: FAIL TTFA_P50=362.15ms TTFA_P95=385.84ms RTF_P50=1.108 MAE_P50=0.058665 SNR_P50=-2.51dB
- 回归 20260128_154912: FAIL TTFA_P50=-1.00ms TTFA_P95=-1.00ms RTF_P50=-1.000 MAE_P50=-1.000000 SNR_P50=-1.00dB
- 回归 20260128_155356: FAIL TTFA_P50=354.75ms TTFA_P95=375.68ms RTF_P50=1.097 MAE_P50=0.050448 SNR_P50=-3.05dB
- 回归 20260128_160543: FAIL TTFA_P50=968.71ms TTFA_P95=1519.88ms RTF_P50=1.768 MAE_P50=0.052750 SNR_P50=-2.57dB
- 回归 20260128_165114: FAIL TTFA_P50=356.89ms TTFA_P95=366.79ms RTF_P50=1.100 MAE_P50=0.059059 SNR_P50=-1.66dB
- 回归 20260128_171910: FAIL TTFA_P50=-1.00ms TTFA_P95=-1.00ms RTF_P50=-1.000 MAE_P50=-1.000000 SNR_P50=-1.00dB
- 回归 20260128_202111: FAIL TTFA_P50=1460.17ms TTFA_P95=1873.55ms RTF_P50=2.851 MAE_P50=0.033031 SNR_P50=-1.12dB
- 回归 20260128_213018: FAIL TTFA_P50=6062.04ms TTFA_P95=8419.06ms RTF_P50=18.779 MAE_P50=0.000021 SNR_P50=64.93dB
- 回归 20260128_222542: FAIL TTFA_P50=676.82ms TTFA_P95=1037.93ms RTF_P50=2.816 MAE_P50=0.000025 SNR_P50=65.22dB
- 回归 20260128_234927: FAIL TTFA_P50=845.19ms TTFA_P95=2512.20ms RTF_P50=2.732 MAE_P50=0.000025 SNR_P50=64.51dB
- 回归 20260129_032028: FAIL E2E_TTFA_P50=-1.00ms E2E_TTFA_P95=-1.00ms MODEL_TTF_P50=-1.00ms RTF_P50=-1.000 MAE_P50=-1.000000 SNR_P50=-1.00dB
- 回归 20260129_032345: FAIL E2E_TTFA_P50=364.28ms E2E_TTFA_P95=367.68ms MODEL_TTF_P50=359.03ms RTF_P50=1.087 MAE_P50=0.044830 SNR_P50=-2.43dB
- 回归 20260129_034340: PASS E2E_TTFA_P50=344.78ms E2E_TTFA_P95=357.30ms MODEL_TTF_P50=339.71ms RTF_P50=1.040 MAE_P50=0.000027 SNR_P50=64.35dB
- 回归 20260129_063825: PASS E2E_TTFA_P50=347.32ms E2E_TTFA_P95=359.22ms MODEL_TTF_P50=342.09ms RTF_P50=1.029 MAE_P50=0.000027 SNR_P50=64.35dB
- 回归 20260129_064415: PASS E2E_TTFA_P50=356.42ms E2E_TTFA_P95=361.61ms MODEL_TTF_P50=350.94ms RTF_P50=1.048 MAE_P50=0.000027 SNR_P50=64.30dB
- 回归 20260129_070412: FAIL E2E_TTFA_P50=644.99ms E2E_TTFA_P95=652.29ms MODEL_TTF_P50=638.84ms RTF_P50=0.994 MAE_P50=0.000027 SNR_P50=64.26dB
- 回归 20260129_072241: PASS E2E_TTFA_P50=352.60ms E2E_TTFA_P95=360.30ms MODEL_TTF_P50=347.35ms RTF_P50=1.047 MAE_P50=0.000027 SNR_P50=64.30dB
- 回归 20260129_222236: FAIL E2E_TTFA_P50=463.12ms E2E_TTFA_P95=467.61ms MODEL_TTF_P50=456.03ms RTF_P50=1.005 MAE_P50=0.040130 SNR_P50=-2.84dB
- 回归 20260129_222742: FAIL E2E_TTFA_P50=455.81ms E2E_TTFA_P95=462.95ms MODEL_TTF_P50=450.64ms RTF_P50=0.999 MAE_P50=0.011663 SNR_P50=3.27dB
- 回归 20260129_223257: FAIL E2E_TTFA_P50=453.05ms E2E_TTFA_P95=458.86ms MODEL_TTF_P50=445.34ms RTF_P50=1.008 MAE_P50=0.009286 SNR_P50=4.59dB
- 回归 20260129_225717: FAIL E2E_TTFA_P50=-1.00ms E2E_TTFA_P95=-1.00ms MODEL_TTF_P50=-1.00ms RTF_P50=-1.000 MAE_P50=-1.000000 SNR_P50=-1.00dB
- 回归 20260130_044339: FAIL E2E_TTFA_P50=726.82ms E2E_TTFA_P95=4622.95ms MODEL_TTF_P50=654.94ms RTF_P50=2.090 MAE_P50=0.040323 SNR_P50=-2.67dB
- 回归 20260130_044339: FAIL E2E_TTFA_P50=411.50ms E2E_TTFA_P95=443.75ms MODEL_TTF_P50=405.42ms RTF_P50=1.062 MAE_P50=0.000027 SNR_P50=62.29dB
- 回归 20260131_002331: FAIL E2E_TTFA_P50=648.10ms E2E_TTFA_P95=657.93ms MODEL_TTF_P50=641.00ms RTF_P50=1.528 MAE_P50=0.000026 SNR_P50=66.63dB

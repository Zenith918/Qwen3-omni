Qwen3-TTS references (saved notes)

Official repo:
- https://github.com/QwenLM/Qwen3-TTS
- README notes: vLLM-Omni supports offline inference only for Qwen3-TTS; streaming/online support is future work.

DashScope real-time API docs:
- CN: https://help.aliyun.com/zh/model-studio/qwen-tts-realtime
- EN: https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime

Key API behaviors (paraphrased):
- WebSocket streaming events include response.audio.delta for incremental PCM bytes.
- server_commit mode: server decides segmentation and synthesis timing for low latency.
- commit mode: client controls segmentation by committing buffered text.

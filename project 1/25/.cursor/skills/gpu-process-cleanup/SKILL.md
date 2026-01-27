---
name: gpu-process-cleanup
description: Prevents GPU memory leaks from orphaned processes when running vLLM and TTS services. Use when starting/stopping GPU servers, troubleshooting GPU memory that remains allocated after a process exits, or when changing GPU memory utilization settings.
---

# GPU Process Cleanup

## Quick Start

- Always start servers via the project scripts in `scripts/`.
- Stop servers with SIGTERM or Ctrl+C; avoid `kill -9` unless hung.
- If GPU memory stays allocated, clean up orphaned processes before restarting.

## Standard Workflow

1. Start servers using the scripts:
   - `bash "scripts/run_llm_server.sh"`
   - `bash "scripts/run_tts_server.sh"`
2. Stop servers cleanly:
   - Ctrl+C in the terminal, or
   - `kill <script_pid>` to trigger the process-group trap.
3. Verify GPU is clean before restart:
   - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`

## If GPU Memory Remains Allocated

1. Identify orphaned GPU processes:
   - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`
2. Gracefully stop them:
   - `kill <pid>`
3. If still present after 5–10s:
   - `kill -9 <pid>` 
4. if still present, ask user to restart pod(last resort)

## Notes

- vLLM and vLLM-Omni spawn worker processes; killing only the parent leaves GPU allocations behind.
- Use the built-in process-group trap in `scripts/run_llm_server.sh` and `scripts/run_tts_server.sh` to avoid orphan workers.
- Avoid running multiple GPU servers concurrently unless needed; memory fragmentation can prevent startup.

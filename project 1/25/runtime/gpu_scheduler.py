#!/usr/bin/env python3
"""
Day 4 P0-1: GPU 硬优先级调度器

设计原则：
  - Fast lane 永远不被 slow lane 阻塞（SLO 保证）
  - Slow lane 永远不排队等待（non-blocking try_acquire）
  - 任何回合 fast TTFT 增幅 > 30%（vs 基线）→ 算作"干扰"，必须为 0

实现：
  - Semaphore(1) 控制 GPU 排他访问
  - fast_acquire(): 立即获取（仅等其他 fast 完成，不等 slow）
  - slow_try_acquire(): 非阻塞，拿不到就跳过
  - Barge-in 后冷却时间：slow lane 至少延迟 N 秒
  - Slow lane 执行预算：超时自动标记（无法中断 CUDA，但记录）
"""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class GPUSchedulerStats:
    """调度器统计"""
    fast_calls: int = 0
    fast_total_ms: float = 0.0
    fast_max_ms: float = 0.0
    fast_blocked_count: int = 0          # fast 等了 slow 的次数
    fast_blocked_total_ms: float = 0.0   # fast 等 slow 的总时间

    slow_calls: int = 0
    slow_completed: int = 0
    slow_skipped: int = 0                # slow 因 fast 活跃被跳过
    slow_skipped_cooldown: int = 0       # slow 因冷却期被跳过
    slow_total_ms: float = 0.0
    slow_max_ms: float = 0.0
    slow_over_budget: int = 0            # slow 超预算次数

    interference_count: int = 0          # fast TTFT 被 slow 干扰（>30% 增幅）的次数

    def to_dict(self) -> dict:
        return {
            "fast_calls": self.fast_calls,
            "fast_avg_ms": round(self.fast_total_ms / max(1, self.fast_calls), 1),
            "fast_max_ms": round(self.fast_max_ms, 1),
            "fast_blocked_count": self.fast_blocked_count,
            "fast_blocked_total_ms": round(self.fast_blocked_total_ms, 1),
            "slow_calls": self.slow_calls,
            "slow_completed": self.slow_completed,
            "slow_skipped": self.slow_skipped,
            "slow_skipped_cooldown": self.slow_skipped_cooldown,
            "slow_total_ms": round(self.slow_total_ms, 1),
            "slow_max_ms": round(self.slow_max_ms, 1),
            "slow_over_budget": self.slow_over_budget,
            "interference_count": self.interference_count,
        }


class GPUScheduler:
    """
    硬优先级 GPU 调度器。

    - fast lane: 调用 with fast_lane() 上下文管理器
    - slow lane: 调用 with slow_lane() 上下文管理器（不阻塞，yield success/skip）
    """

    def __init__(self,
                 slow_budget_ms: float = 800,
                 bargein_cooldown_ms: float = 3000,
                 fast_baseline_ms: float = 100,
                 min_idle_before_slow_ms: float = 2000):
        self._gpu_sem = threading.Semaphore(1)
        self._state_lock = threading.Lock()

        self._fast_active = False           # fast lane 是否正在运行
        self._fast_waiting = False          # fast lane 是否在等待
        self._slow_active = False           # slow lane 是否正在运行
        self._last_bargein_time: float = 0  # 上次 barge-in 时间
        self._last_fast_end_time: float = 0 # 上次 fast lane 结束时间

        self.slow_budget_ms = slow_budget_ms
        self.bargein_cooldown_ms = bargein_cooldown_ms
        self.fast_baseline_ms = fast_baseline_ms  # fast lane 基线延迟
        self.min_idle_before_slow_ms = min_idle_before_slow_ms  # slow lane 最小空闲窗口

        self.stats = GPUSchedulerStats()

    def notify_bargein(self):
        """通知调度器发生了 barge-in，启动冷却计时"""
        self._last_bargein_time = time.time()

    # ── Fast Lane ────────────────────────────────────────────
    class _FastContext:
        def __init__(self, scheduler: 'GPUScheduler'):
            self.sched = scheduler
            self._t_start = 0
            self._t_acquired = 0

        def __enter__(self):
            self._t_start = time.time()

            with self.sched._state_lock:
                self.sched._fast_waiting = True

            # Acquire GPU — 如果 slow lane 持有，会等待（但会记录）
            blocked = not self.sched._gpu_sem.acquire(blocking=False)
            if blocked:
                # Slow lane 持有 GPU，记录阻塞
                with self.sched._state_lock:
                    self.sched.stats.fast_blocked_count += 1
                self.sched._gpu_sem.acquire()  # 阻塞等待
                block_ms = (time.time() - self._t_start) * 1000
                with self.sched._state_lock:
                    self.sched.stats.fast_blocked_total_ms += block_ms

            self._t_acquired = time.time()
            with self.sched._state_lock:
                self.sched._fast_active = True
                self.sched._fast_waiting = False

            return self

        def __exit__(self, *args):
            elapsed_ms = (time.time() - self._t_acquired) * 1000
            with self.sched._state_lock:
                self.sched._fast_active = False
                self.sched._last_fast_end_time = time.time()
                self.sched.stats.fast_calls += 1
                self.sched.stats.fast_total_ms += elapsed_ms
                self.sched.stats.fast_max_ms = max(self.sched.stats.fast_max_ms, elapsed_ms)

                # 检查是否被 slow lane 干扰（TTFT > baseline * 1.3）
                total_ms = (time.time() - self._t_start) * 1000
                if total_ms > self.sched.fast_baseline_ms * 1.3:
                    self.sched.stats.interference_count += 1

            self.sched._gpu_sem.release()

    def fast_lane(self):
        """获取 fast lane GPU 访问（上下文管理器）"""
        return self._FastContext(self)

    # ── Slow Lane ────────────────────────────────────────────
    class _SlowResult:
        def __init__(self, acquired: bool, reason: str = ""):
            self.acquired = acquired
            self.reason = reason

    class _SlowContext:
        def __init__(self, scheduler: 'GPUScheduler'):
            self.sched = scheduler
            self._acquired = False
            self._t_start = 0
            self.result = GPUScheduler._SlowResult(False)

        def __enter__(self):
            self._t_start = time.time()

            with self.sched._state_lock:
                self.sched.stats.slow_calls += 1

                # 检查 barge-in 冷却期
                time_since_bargein = (time.time() - self.sched._last_bargein_time) * 1000
                if time_since_bargein < self.sched.bargein_cooldown_ms:
                    self.sched.stats.slow_skipped_cooldown += 1
                    self.result = GPUScheduler._SlowResult(
                        False, f"cooldown ({time_since_bargein:.0f}ms < {self.sched.bargein_cooldown_ms}ms)")
                    return self

                # 检查 fast lane 是否活跃或等待
                if self.sched._fast_active or self.sched._fast_waiting:
                    self.sched.stats.slow_skipped += 1
                    self.result = GPUScheduler._SlowResult(False, "fast_lane_active")
                    return self

                # D4: 检查最小空闲窗口（fast lane 结束后至少 N ms 才允许 slow）
                time_since_fast = (time.time() - self.sched._last_fast_end_time) * 1000
                if time_since_fast < self.sched.min_idle_before_slow_ms:
                    self.sched.stats.slow_skipped += 1
                    self.result = GPUScheduler._SlowResult(
                        False, f"idle_window_too_short ({time_since_fast:.0f}ms < {self.sched.min_idle_before_slow_ms}ms)")
                    return self

            # 非阻塞尝试获取 GPU
            if self.sched._gpu_sem.acquire(blocking=False):
                # 二次检查（获取后 fast 可能刚到）
                with self.sched._state_lock:
                    if self.sched._fast_active or self.sched._fast_waiting:
                        self.sched._gpu_sem.release()
                        self.sched.stats.slow_skipped += 1
                        self.result = GPUScheduler._SlowResult(False, "fast_arrived_after_acquire")
                        return self
                    self.sched._slow_active = True

                self._acquired = True
                self.result = GPUScheduler._SlowResult(True)
            else:
                with self.sched._state_lock:
                    self.sched.stats.slow_skipped += 1
                self.result = GPUScheduler._SlowResult(False, "gpu_busy")

            return self

        def __exit__(self, *args):
            if self._acquired:
                elapsed_ms = (time.time() - self._t_start) * 1000
                with self.sched._state_lock:
                    self.sched._slow_active = False
                    self.sched.stats.slow_completed += 1
                    self.sched.stats.slow_total_ms += elapsed_ms
                    self.sched.stats.slow_max_ms = max(self.sched.stats.slow_max_ms, elapsed_ms)
                    if elapsed_ms > self.sched.slow_budget_ms:
                        self.sched.stats.slow_over_budget += 1

                self.sched._gpu_sem.release()

    def slow_lane(self):
        """尝试获取 slow lane GPU 访问（上下文管理器，不阻塞）"""
        return self._SlowContext(self)


# ── 测试 ────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    sched = GPUScheduler(slow_budget_ms=500, bargein_cooldown_ms=2000, fast_baseline_ms=80)

    # 模拟场景
    print("=== Fast lane test ===")
    with sched.fast_lane():
        time.sleep(0.05)  # 50ms
        print(f"  Fast done")

    print("=== Slow lane (should succeed) ===")
    with sched.slow_lane() as ctx:
        if ctx.result.acquired:
            time.sleep(0.1)  # 100ms
            print(f"  Slow done")
        else:
            print(f"  Slow skipped: {ctx.result.reason}")

    print("=== Fast + Slow concurrent ===")
    def slow_job():
        time.sleep(0.1)
        with sched.slow_lane() as ctx:
            if ctx.result.acquired:
                time.sleep(0.3)
                print(f"  Slow done (during fast)")
            else:
                print(f"  Slow skipped: {ctx.result.reason}")

    t = threading.Thread(target=slow_job, daemon=True)
    t.start()
    time.sleep(0.05)

    with sched.fast_lane():
        time.sleep(0.05)
        print(f"  Fast done (concurrent)")
    t.join()

    print(f"\nStats: {json.dumps(sched.stats.to_dict(), indent=2)}")


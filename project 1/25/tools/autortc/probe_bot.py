#!/usr/bin/env python3
"""
D9 probe_bot: subscribe agent audio, record full + reply segment,
send probe_ready barrier, receive reply_start/reply_end events.
"""
import argparse
import asyncio
import contextlib
import json
import time

import numpy as np
from livekit import rtc

from common import now_epoch_s, write_json, write_wav_int16


DEFAULT_SR = 48000


def _pick_remote_participant(
    room: rtc.Room,
    local_identity: str,
    target_identity: str,
    target_prefix: str,
    exclude_prefix: str,
):
    candidates = list(room.remote_participants.values())
    if not candidates:
        return None
    if target_identity:
        for p in candidates:
            if p.identity == target_identity and p.identity != local_identity:
                return p
    if target_prefix:
        for p in candidates:
            if p.identity != local_identity and p.identity.startswith(target_prefix):
                return p
    for p in candidates:
        if p.identity == local_identity:
            continue
        if exclude_prefix and p.identity.startswith(exclude_prefix):
            continue
        return p
    for p in candidates:
        if p.identity != local_identity:
            return p
    return None


async def _wait_target_participant(
    room: rtc.Room,
    local_identity: str,
    target_identity: str,
    target_prefix: str,
    exclude_prefix: str,
    timeout_s: float,
):
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        selected = _pick_remote_participant(
            room=room,
            local_identity=local_identity,
            target_identity=target_identity,
            target_prefix=target_prefix,
            exclude_prefix=exclude_prefix,
        )
        if selected is not None:
            return selected
        await asyncio.sleep(0.1)
    return None


async def run_probe_bot(args: argparse.Namespace) -> None:
    room = rtc.Room()

    # ── D9: DataChannel 事件收集 ──
    reply_events: list[dict] = []  # {type, trace_id, t_ms, reply_seq}

    def _on_data_received(data_packet):
        try:
            topic = getattr(data_packet, "topic", "")
            if topic not in ("autortc.reply", "autortc.trace"):
                return
            payload = json.loads(data_packet.data.decode("utf-8"))
            evt_type = payload.get("event", "")
            if evt_type in ("reply_start", "reply_end"):
                reply_events.append({
                    "event": evt_type,
                    "trace_id": payload.get("trace_id", ""),
                    "case_id": payload.get("case_id", ""),
                    "reply_seq": payload.get("reply_seq", 0),
                    "t_ms": now_epoch_s(),
                    "agent_t_ms": payload.get("t_ms", 0),
                })
        except Exception:
            pass

    room.on("data_received", _on_data_received)
    await room.connect(args.url, args.token)

    local_identity = room.local_participant.identity
    target = await _wait_target_participant(
        room,
        local_identity=local_identity,
        target_identity=args.target_identity,
        target_prefix=args.target_identity_prefix,
        exclude_prefix=args.exclude_identity_prefix,
        timeout_s=args.wait_participant_s,
    )
    if target is None:
        if args.result_json:
            write_json(args.result_json, {
                "ok": False, "error": "target participant not found",
                "room": args.room,
            })
        await room.disconnect()
        raise RuntimeError("target participant not found")

    stream = rtc.AudioStream.from_participant(
        participant=target,
        track_source=rtc.TrackSource.SOURCE_MICROPHONE,
        sample_rate=DEFAULT_SR,
        num_channels=1,
        frame_size_ms=args.frame_ms,
    )

    # ── D9 P0-2: 订阅成功后发 probe_ready barrier ──
    await asyncio.sleep(0.3)  # 确保 stream 订阅就绪
    try:
        ready_payload = json.dumps({
            "type": "autortc_probe",
            "event": "probe_ready",
            "probe_identity": local_identity,
            "t_ms": now_epoch_s(),
        }, ensure_ascii=False)
        await room.local_participant.publish_data(
            ready_payload, reliable=True, topic="autortc.probe",
        )
    except Exception:
        pass  # non-fatal

    chunks: list[bytes] = []
    t_record_start = now_epoch_s()
    first_audio_ts = None
    prev_frame_ts = None
    inter_arrival_ms = []
    frame_timestamps = []

    async def consume():
        nonlocal first_audio_ts, prev_frame_ts
        async for event in stream:
            frame = event.frame
            now_ts = now_epoch_s()
            if first_audio_ts is None:
                first_audio_ts = now_ts
            if prev_frame_ts is not None:
                inter_arrival_ms.append((now_ts - prev_frame_ts) * 1000.0)
            prev_frame_ts = now_ts
            frame_timestamps.append(now_ts)
            chunks.append(bytes(frame.data))

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.sleep(args.record_seconds)
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        await stream.aclose()

    pcm_full = b"".join(chunks)

    # ── 保存 full 录音 ──
    if args.output_wav:
        write_wav_int16(args.output_wav, pcm=pcm_full, sample_rate=DEFAULT_SR, num_channels=1)

    # ── D9 P0-1: 提取 reply 段 ──
    reply_wav_path = ""
    reply_pcm = b""
    if reply_events and frame_timestamps and chunks:
        # D9 fix: 过滤掉 trace_id 为空/None 的陈旧事件（来自上一个 case 的 Agent 进程）
        valid_events = [e for e in reply_events if e.get("trace_id")]
        starts = [e for e in valid_events if e["event"] == "reply_start"]
        ends = [e for e in valid_events if e["event"] == "reply_end"]

        # 取所有 reply 段的并集
        reply_frame_indices = set()
        frame_bytes = int(DEFAULT_SR * args.frame_ms / 1000) * 2  # bytes per frame

        for rs in starts:
            t_start = rs["t_ms"]
            # 找对应的 reply_end（同 reply_seq 且同 trace_id）
            t_end = None
            for re in ends:
                if (re.get("reply_seq") == rs.get("reply_seq")
                        and re.get("trace_id") == rs.get("trace_id")
                        and re["t_ms"] > t_start):  # end must be after start
                    t_end = re["t_ms"]
                    break
            if t_end is None:
                # 没有显式 end，取到录音结束
                t_end = frame_timestamps[-1] + 0.1

            # 映射 timestamp → frame index
            for i, ft in enumerate(frame_timestamps):
                if t_start - 0.05 <= ft <= t_end + 0.05:  # 50ms tolerance
                    reply_frame_indices.add(i)

        if reply_frame_indices:
            sorted_idx = sorted(reply_frame_indices)
            reply_chunks = [chunks[i] for i in sorted_idx if i < len(chunks)]
            reply_pcm = b"".join(reply_chunks)

            # 保存 reply wav
            if args.output_wav:
                reply_wav_path = args.output_wav.replace("_agent.wav", "_reply.wav")
                if reply_wav_path == args.output_wav:
                    reply_wav_path = args.output_wav.rsplit(".", 1)[0] + "_reply.wav"
                write_wav_int16(reply_wav_path, pcm=reply_pcm, sample_rate=DEFAULT_SR, num_channels=1)

    if args.result_json:
        p95_inter_arrival = (
            float(np.percentile(np.asarray(inter_arrival_ms, dtype=np.float64), 95))
            if inter_arrival_ms else None
        )
        write_json(
            args.result_json,
            {
                "ok": True,
                "room": args.room,
                "local_identity": local_identity,
                "target_identity": args.target_identity,
                "actual_target_identity": target.identity,
                "t_probe_record_start": t_record_start,
                "t_probe_first_audio_recv": first_audio_ts,
                "record_seconds": args.record_seconds,
                "frame_ms": args.frame_ms,
                "output_wav": args.output_wav,
                "frames_received": len(chunks),
                "inter_arrival_p95_ms": p95_inter_arrival,
                "frame_timestamps": frame_timestamps,
                "pcm_bytes": len(pcm_full),
                "audio_duration_s": float(len(pcm_full)) / 2.0 / float(DEFAULT_SR),
                # D9 reply segment
                "reply_wav": reply_wav_path,
                "reply_pcm_bytes": len(reply_pcm),
                "reply_events": reply_events,
                "reply_duration_s": float(len(reply_pcm)) / 2.0 / float(DEFAULT_SR) if reply_pcm else 0.0,
            },
        )

    await room.disconnect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoRTC probe_bot: subscribe target audio and record wav.")
    p.add_argument("--room", required=True, help="LiveKit room name")
    p.add_argument("--url", required=True, help="LiveKit ws url")
    p.add_argument("--token", required=True, help="JWT token for probe bot")
    p.add_argument("--target_identity", default="", help="participant identity exact match")
    p.add_argument("--target_identity_prefix", default="agent-", help="preferred identity prefix")
    p.add_argument("--exclude_identity_prefix", default="autortc-", help="ignore identity prefix")
    p.add_argument("--frame_ms", type=int, default=20, help="audio stream frame size")
    p.add_argument("--wait_participant_s", type=float, default=15.0, help="wait time for target participant")
    p.add_argument("--record_seconds", type=float, default=20.0, help="fixed recording window")
    p.add_argument("--output_wav", required=True, help="recorded wav output path")
    p.add_argument("--result_json", default="", help="optional path to dump probe timestamps")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_probe_bot(parse_args()))

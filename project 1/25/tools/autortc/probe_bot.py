#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
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

    # 1) 显式 identity 精确匹配
    if target_identity:
        for p in candidates:
            if p.identity == target_identity and p.identity != local_identity:
                return p

    # 2) identity 前缀匹配（如 agent-）
    if target_prefix:
        for p in candidates:
            if p.identity != local_identity and p.identity.startswith(target_prefix):
                return p

    # 3) 非自己且不在排除前缀（默认排除 autortc-）
    for p in candidates:
        if p.identity == local_identity:
            continue
        if exclude_prefix and p.identity.startswith(exclude_prefix):
            continue
        return p

    # 4) 最后兜底：任意非自己
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
            write_json(
                args.result_json,
                {
                    "ok": False,
                    "error": "target participant not found",
                    "room": args.room,
                    "target_identity": args.target_identity,
                    "target_identity_prefix": args.target_identity_prefix,
                    "exclude_identity_prefix": args.exclude_identity_prefix,
                },
            )
        await room.disconnect()
        raise RuntimeError("target participant not found")

    stream = rtc.AudioStream.from_participant(
        participant=target,
        track_source=rtc.TrackSource.SOURCE_MICROPHONE,
        sample_rate=DEFAULT_SR,
        num_channels=1,
        frame_size_ms=args.frame_ms,
    )

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

    pcm = b"".join(chunks)
    if args.output_wav:
        write_wav_int16(args.output_wav, pcm=pcm, sample_rate=DEFAULT_SR, num_channels=1)

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
                "pcm_bytes": len(pcm),
                "audio_duration_s": float(len(pcm)) / 2.0 / float(DEFAULT_SR),
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


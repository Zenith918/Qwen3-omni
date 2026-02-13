#!/usr/bin/env python3
import argparse
import asyncio
import json
import time

import numpy as np
from livekit import rtc

from common import (
    load_wav_mono_int16,
    now_epoch_s,
    resample_linear_int16,
    write_json,
)


DEFAULT_SR = 48000


async def run_user_bot(args: argparse.Namespace) -> None:
    room = rtc.Room()
    await room.connect(args.url, args.token)

    source = rtc.AudioSource(sample_rate=DEFAULT_SR, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("autortc-user", source)
    pub_opts = rtc.TrackPublishOptions()
    pub_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, pub_opts)

    audio, src_sr = load_wav_mono_int16(args.wav)
    audio = resample_linear_int16(audio, src_sr, DEFAULT_SR)

    frame_samples = int(DEFAULT_SR * args.frame_ms / 1000)
    frame_samples = max(frame_samples, 1)

    if args.start_delay_s > 0:
        await asyncio.sleep(args.start_delay_s)

    t_send_start = now_epoch_s()
    if args.trace_id:
        payload = {
            "type": "autortc_trace",
            "event": "user_send_start",
            "trace_id": args.trace_id,
            "case_id": args.case_id,
            "turn_id": args.turn_id,
            "t_user_send_start": t_send_start,
        }
        await room.local_participant.publish_data(
            json.dumps(payload, ensure_ascii=False),
            reliable=True,
            topic="autortc.trace",
        )
    next_tick = time.monotonic()
    for idx in range(0, len(audio), frame_samples):
        chunk = audio[idx:idx + frame_samples]
        if len(chunk) < frame_samples:
            pad = frame_samples - len(chunk)
            chunk = chunk if pad <= 0 else np.pad(chunk, (0, pad), mode="constant")

        frame = rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=DEFAULT_SR,
            num_channels=1,
            samples_per_channel=frame_samples,
        )
        await source.capture_frame(frame)

        if args.realtime == 1:
            next_tick += args.frame_ms / 1000.0
            delay = next_tick - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    await source.wait_for_playout()
    if args.post_silence_ms > 0:
        await asyncio.sleep(args.post_silence_ms / 1000.0)
    t_send_end = now_epoch_s()

    if args.trace_id:
        payload = {
            "type": "autortc_trace",
            "event": "user_send_end",
            "trace_id": args.trace_id,
            "case_id": args.case_id,
            "turn_id": args.turn_id,
            "t_user_send_end": t_send_end,
        }
        await room.local_participant.publish_data(
            json.dumps(payload, ensure_ascii=False),
            reliable=True,
            topic="autortc.trace",
        )

    if args.result_json:
        write_json(
            args.result_json,
            {
                "ok": True,
                "room": args.room,
                "wav": args.wav,
                "frame_ms": args.frame_ms,
                "realtime": args.realtime,
                "t_user_send_start": t_send_start,
                "t_user_send_end": t_send_end,
                "audio_samples_48k": int(len(audio)),
                "audio_duration_s": float(len(audio)) / float(DEFAULT_SR),
            },
        )

    await room.disconnect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoRTC user_bot: publish wav as realtime mic audio.")
    p.add_argument("--room", required=True, help="LiveKit room name")
    p.add_argument("--url", required=True, help="LiveKit ws url, e.g. wss://...livekit.cloud")
    p.add_argument("--token", required=True, help="JWT token for user bot")
    p.add_argument("--wav", required=True, help="input wav path")
    p.add_argument("--realtime", type=int, default=1, help="1=realtime pace, 0=as fast as possible")
    p.add_argument("--frame_ms", type=int, default=20, help="frame size in ms")
    p.add_argument("--start_delay_s", type=float, default=2.5, help="delay before sending first frame")
    p.add_argument("--post_silence_ms", type=int, default=300, help="tail wait before exit")
    p.add_argument("--trace_id", default="", help="optional trace id sent via data channel")
    p.add_argument("--case_id", default="", help="optional case id for trace metadata")
    p.add_argument("--turn_id", default="", help="optional turn id for trace metadata")
    p.add_argument("--result_json", default="", help="optional path to dump send timestamps")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_user_bot(parse_args()))


# Network Impairment Testing Roadmap

## Current Status

### toxiproxy (v2.9.0) — Application-Layer Only

toxiproxy operates at the **TCP socket level**. It proxies TCP connections and can inject:
- Latency (delay + jitter)
- Bandwidth limiting
- Connection reset / timeout
- Slow close

**Limitation**: toxiproxy does NOT affect the WebRTC media path.
- WebRTC audio/video uses **UDP** (SRTP/SRTCP), not TCP.
- toxiproxy can only impair the **signaling** (WebSocket to LiveKit) and **token API** (HTTP).
- Testing with toxiproxy alone will NOT simulate real network degradation for voice quality.

**What toxiproxy CAN test**:
- Token fetch latency → join delay
- WebSocket signaling delay → room join/leave timing
- ICE candidate exchange delay → connection setup time

**What toxiproxy CANNOT test**:
- Audio packet loss / jitter / delay (the actual user experience)
- Codec switching under poor conditions
- Jitter buffer behavior

### netem — Kernel-Level (Recommended)

`tc netem` operates at the **network interface level** (Layer 3), affecting ALL traffic including UDP.

**Requirements**:
- Linux kernel with `sch_netem` module
- `cap_net_admin` capability (missing in current RunPod container)

**How to enable on RunPod**:
1. Create pod with "Enable Privileged Container" or add `NET_ADMIN` capability
2. Or use a custom Docker image with `NET_ADMIN` in the Dockerfile

**Usage**:
```bash
# Add 50ms delay with 10ms jitter to all traffic
tc qdisc add dev eth0 root netem delay 50ms 10ms

# Add 5% packet loss
tc qdisc change dev eth0 root netem loss 5%

# Remove
tc qdisc del dev eth0 root netem
```

### Recommended Testing Strategy

| Layer | Tool | Tests | Priority |
|-------|------|-------|----------|
| Signaling (TCP) | toxiproxy | Join delay, token latency | P1 |
| Media (UDP) | netem | Audio quality under loss/jitter | P0 (needs NET_ADMIN) |
| Application | Python proxy | Custom impairment scenarios | P2 |

### Next Steps

1. Request RunPod pod with `NET_ADMIN` capability
2. Create network profile presets in `run_suite.py` (already has `NET_PROFILES` dict)
3. Run full 16 cases under each profile: `wifi_good`, `wifi_bad`, `4g`, `3g`
4. Compare USER_KPI distributions across profiles

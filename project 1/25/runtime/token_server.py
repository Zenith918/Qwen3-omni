#!/usr/bin/env python3
"""
Token Server — 为 WebRTC 前端生成 LiveKit JWT Token

持久化 API：
  GET /api/token          → 生成新 token（默认 room + identity）
  GET /api/token?room=xxx&identity=yyy  → 自定义 room/identity
  GET /api/health         → 健康检查

前端只需 fetch('/api/token') 就能拿到新鲜 token，永不过期。
"""

import datetime
import json
import os
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# LiveKit SDK
from livekit import api as livekit_api

# ── 配置 ─────────────────────────────────────────────────────
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "API7fj35wGLumtc")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "WK8k8fUhhsHoa2R2qfO076lyuDHgJubwemQuY4nk398B")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://renshenghehuoren-mpdsjfwe.livekit.cloud")

DEFAULT_ROOM = "voice-agent-test"
DEFAULT_IDENTITY = "voice-user-1"
TOKEN_TTL = 86400  # 24 小时

# 前端静态文件目录
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_token(room: str, identity: str) -> str:
    """生成 LiveKit JWT Token"""
    token = livekit_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity)
    token.with_name("User")
    token.with_grants(livekit_api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    ))
    token.with_ttl(datetime.timedelta(seconds=TOKEN_TTL))
    return token.to_jwt()


class VoiceAgentHandler(SimpleHTTPRequestHandler):
    """统一 HTTP Handler：静态文件 + Token API"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/token":
            self._handle_token(parsed)
        elif path == "/api/health":
            self._handle_health()
        elif path == "/api/config":
            self._handle_config()
        elif path == "/" or path == "":
            # 默认返回 webrtc 前端
            self.path = "/webrtc_test.html"
            super().do_GET()
        else:
            super().do_GET()

    def _handle_token(self, parsed):
        """生成并返回 Token"""
        params = parse_qs(parsed.query)
        room = params.get("room", [DEFAULT_ROOM])[0]
        identity = params.get("identity", [DEFAULT_IDENTITY])[0]

        try:
            jwt_token = generate_token(room, identity)
            response = {
                "token": jwt_token,
                "url": LIVEKIT_URL,
                "room": room,
                "identity": identity,
                "ttl": TOKEN_TTL,
            }
            self._json_response(200, response)
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _handle_health(self):
        """健康检查"""
        self._json_response(200, {
            "status": "ok",
            "services": {
                "token_server": True,
                "livekit_url": LIVEKIT_URL,
            },
            "timestamp": time.time(),
        })

    def _handle_config(self):
        """返回前端所需配置"""
        self._json_response(200, {
            "livekit_url": LIVEKIT_URL,
            "default_room": DEFAULT_ROOM,
        })

    def _json_response(self, code, data):
        """返回 JSON"""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        """减少日志噪音"""
        if "/api/" in str(args[0]) if args else False:
            super().log_message(format, *args)


if __name__ == "__main__":
    port = int(os.environ.get("TOKEN_SERVER_PORT", "9090"))
    server = HTTPServer(("0.0.0.0", port), VoiceAgentHandler)
    print(f"🚀 Voice Agent Server running on port {port}")
    print(f"   Frontend: http://localhost:{port}/")
    print(f"   Token API: http://localhost:{port}/api/token")
    print(f"   Health: http://localhost:{port}/api/health")
    print(f"   LiveKit: {LIVEKIT_URL}")
    server.serve_forever()


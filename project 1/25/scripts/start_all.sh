#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Qwen3 Voice Agent — 一键启动所有服务
#
# 用法:
#   bash scripts/start_all.sh         # 启动所有
#   bash scripts/start_all.sh restart # 重启所有
#   bash scripts/start_all.sh status  # 查看状态
#   bash scripts/start_all.sh stop    # 停止所有
# ═══════════════════════════════════════════════════════════════

set -e
PROJECT_DIR="/workspace/project 1/25"
cd "$PROJECT_DIR"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 环境变量
export LIVEKIT_URL="wss://renshenghehuoren-mpdsjfwe.livekit.cloud"
export LIVEKIT_API_KEY="API7fj35wGLumtc"
export LIVEKIT_API_SECRET="WK8k8fUhhsHoa2R2qfO076lyuDHgJubwemQuY4nk398B"
export PYTHONPATH="/workspace/vllm-omni"

ACTION="${1:-start}"

status() {
    echo -e "${GREEN}═══ 服务状态 ═══${NC}"
    
    # LLM
    if pgrep -f "vllm.entrypoints" > /dev/null; then
        echo -e "  LLM (vLLM):      ${GREEN}✅ 运行中${NC} (port 8000)"
    else
        echo -e "  LLM (vLLM):      ${RED}❌ 未运行${NC}"
    fi
    
    # TTS
    if pgrep -f "tts_server.py" > /dev/null; then
        echo -e "  TTS Server:      ${GREEN}✅ 运行中${NC} (port 9000)"
    else
        echo -e "  TTS Server:      ${RED}❌ 未运行${NC}"
    fi
    
    # Agent
    if pgrep -f "livekit_agent.py" > /dev/null; then
        echo -e "  LiveKit Agent:   ${GREEN}✅ 运行中${NC} (port 8089)"
    else
        echo -e "  LiveKit Agent:   ${RED}❌ 未运行${NC}"
    fi
    
    # Token Server
    if pgrep -f "token_server.py" > /dev/null; then
        echo -e "  Token Server:    ${GREEN}✅ 运行中${NC} (port 9090)"
    else
        echo -e "  Token Server:    ${RED}❌ 未运行${NC}"
    fi
    
    # Nginx
    if pgrep nginx > /dev/null; then
        echo -e "  Nginx:           ${GREEN}✅ 运行中${NC} (port 9091)"
    else
        echo -e "  Nginx:           ${RED}❌ 未运行${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}═══ 访问地址 ═══${NC}"
    POD_ID="${RUNPOD_POD_ID:-bw99o2iauzf2hb}"
    echo -e "  🌐 前端:  https://${POD_ID}-9091.proxy.runpod.net/"
    echo -e "  🔑 Token: https://${POD_ID}-9091.proxy.runpod.net/api/token"
    echo -e "  ❤️  Health: https://${POD_ID}-9091.proxy.runpod.net/api/health"
}

stop_service() {
    local name=$1
    local pattern=$2
    local pid=$(pgrep -f "$pattern" 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        kill $pid 2>/dev/null
        sleep 2
        kill -9 $pid 2>/dev/null
        echo -e "  ${YELLOW}停止 $name (PID $pid)${NC}"
    fi
}

stop_all() {
    echo -e "${YELLOW}═══ 停止所有服务 ═══${NC}"
    stop_service "LiveKit Agent" "livekit_agent.py"
    stop_service "Token Server" "token_server.py"
    # 不停 LLM 和 TTS（启动太慢）
    echo -e "  ${YELLOW}（LLM 和 TTS 保持运行）${NC}"
}

start_token_server() {
    if pgrep -f "token_server.py" > /dev/null; then
        echo -e "  Token Server: ${GREEN}已在运行${NC}"
        return
    fi
    cd "$PROJECT_DIR/runtime"
    python3 token_server.py > /tmp/token_server.log 2>&1 &
    disown
    echo -e "  Token Server: ${GREEN}已启动${NC} (PID $!)"
}

start_agent() {
    if pgrep -f "livekit_agent.py" > /dev/null; then
        echo -e "  LiveKit Agent: ${GREEN}已在运行${NC}"
        return
    fi
    cd "$PROJECT_DIR"
    python3 runtime/livekit_agent.py start > /tmp/livekit_agent.log 2>&1 &
    disown
    echo -e "  LiveKit Agent: ${GREEN}已启动${NC} (PID $!)"
}

start_nginx() {
    # 确保配置已链接
    ln -sf /etc/nginx/sites-available/voice-agent /etc/nginx/sites-enabled/voice-agent 2>/dev/null
    
    if pgrep nginx > /dev/null; then
        nginx -s reload 2>/dev/null
        echo -e "  Nginx: ${GREEN}已重载${NC}"
    else
        nginx 2>/dev/null
        echo -e "  Nginx: ${GREEN}已启动${NC}"
    fi
}

start_all() {
    echo -e "${GREEN}═══ 启动服务 ═══${NC}"
    start_token_server
    sleep 2
    start_agent
    sleep 3
    start_nginx
    echo ""
    
    # 等 Agent 注册
    echo -e "${YELLOW}等待 Agent 注册到 LiveKit Cloud...${NC}"
    for i in $(seq 1 10); do
        if grep -q "registered" /tmp/livekit_agent.log 2>/dev/null; then
            echo -e "${GREEN}✅ Agent 已注册！${NC}"
            break
        fi
        sleep 2
    done
    
    echo ""
    status
}

case "$ACTION" in
    start)
        start_all
        ;;
    restart)
        stop_all
        sleep 3
        start_all
        ;;
    stop)
        stop_all
        ;;
    status)
        status
        ;;
    *)
        echo "用法: $0 {start|restart|stop|status}"
        exit 1
        ;;
esac


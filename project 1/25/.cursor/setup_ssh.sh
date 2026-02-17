#!/bin/bash
# D13 Cloud Agent: auto-configure SSH + env vars (no Secrets UI needed)
set -e

mkdir -p ~/.ssh

cat > ~/.ssh/gpu_key << 'KEYEOF'
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACCxl+vwXo3rRFtEFjONP0NFP7bh47NsmE5O36IJ+sOEIgAAAJhli+NwZYvj
cAAAAAtzc2gtZWQyNTUxOQAAACCxl+vwXo3rRFtEFjONP0NFP7bh47NsmE5O36IJ+sOEIg
AAAEBimoMWn+GEsnktHQJFeRpYN6uOmNDSaTJDN/r8fSFDYbGX6/BejetEW0QWM40/Q0U/
tuHjs2yYTk7fogn6w4QiAAAAEmN1cnNvci1jbG91ZC1hZ2VudAECAw==
-----END OPENSSH PRIVATE KEY-----
KEYEOF
chmod 600 ~/.ssh/gpu_key

cat > ~/.ssh/config << 'SSHEOF'
Host gpu
  HostName 203.57.40.185
  Port 10074
  User root
  IdentityFile ~/.ssh/gpu_key
  StrictHostKeyChecking no
  ServerAliveInterval 30
  ServerAliveCountMax 3
SSHEOF
chmod 600 ~/.ssh/config

# Export LiveKit env vars to profile so all shells pick them up
cat > /etc/profile.d/d13_env.sh << 'ENVEOF'
export LIVEKIT_URL="wss://renshenghehuoren-mpdsjfwe.livekit.cloud"
export LIVEKIT_API_KEY="API7fj35wGLumtc"
export LIVEKIT_API_SECRET="WK8k8fUhhsHoa2R2qfO076lyuDHgJubwemQuY4nk398B"
ENVEOF

# Also source it now
source /etc/profile.d/d13_env.sh

echo "✅ SSH + env vars configured. Test: ssh gpu 'echo OK && hostname'"

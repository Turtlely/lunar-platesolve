#!/usr/bin/env bash
# Start the API (127.0.0.1:8000 only) and the Cloudflare Tunnel.  Stop both with Ctrl+C.
#
#   bash deploy/server/run.sh           named tunnel (TUNNEL_TOKEN in deploy/server/.env)
#   bash deploy/server/run.sh --quick   temporary https://....trycloudflare.com address, no DNS
#
# Logs: deploy/server/api.log
set -euo pipefail
cd "$(dirname "$0")/../.."
APP="$(pwd)"
export PATH="$HOME/.local/bin:$PATH"
PORT="${PORT:-8000}"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

TUNNEL_TOKEN=""
THREADS=""
if [ -f deploy/server/.env ]; then
    TUNNEL_TOKEN=$(sed -n 's/^[[:space:]]*TUNNEL_TOKEN[[:space:]]*=[[:space:]]*//p' deploy/server/.env | tail -1 | tr -d '\r[:space:]')
    THREADS=$(sed -n 's/^[[:space:]]*THREADS[[:space:]]*=[[:space:]]*//p' deploy/server/.env | tail -1 | tr -d '\r[:space:]')
fi
if [ "$QUICK" = 0 ] && [ -z "$TUNNEL_TOKEN" ]; then
    echo "Set TUNNEL_TOKEN in deploy/server/.env, or use --quick."
    exit 1
fi
export OMP_NUM_THREADS="${THREADS:-$(nproc)}"
export LUNARSOLVE_DEVICE="${LUNARSOLVE_DEVICE:-cpu}"

echo "starting the API on 127.0.0.1:$PORT (log: deploy/server/api.log) ..."
"$APP/.venv/bin/python" server/api.py --host 127.0.0.1 --port "$PORT" >deploy/server/api.log 2>&1 &
API_PID=$!
trap 'echo "stopping the API"; kill $API_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do
    sleep 2
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "The API stopped. Last lines of deploy/server/api.log:"
        tail -20 deploy/server/api.log
        exit 1
    fi
    if curl -fs "http://127.0.0.1:$PORT/api/health" | grep -q '"loaded":true'; then
        break
    fi
done
curl -fs "http://127.0.0.1:$PORT/api/health" >/dev/null || { echo "The API did not start in 3 minutes."; exit 1; }
echo "API ready."

if [ "$QUICK" = 1 ]; then
    echo "Look for the https://....trycloudflare.com line below."
    cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT"
else
    cloudflared tunnel --no-autoupdate run --token "$TUNNEL_TOKEN"
fi

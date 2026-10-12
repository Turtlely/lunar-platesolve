#!/usr/bin/env bash
# Install the lunar plate solver API on a Linux server without root access.
#
#   bash deploy/server/setup_user.sh          (run from the unpacked code folder)
#
# Installs into your home directory only:
#   ~/.local/bin/uv           Python package manager (also installs Python 3.12)
#   ./.venv                   Python 3.12 with CPU-only torch and the server packages
#   ~/.cache/lunarsolve       solver data (about 1 GB) from the Hugging Face dataset
#   ~/.local/bin/cloudflared  Cloudflare Tunnel client
# Needs: outgoing HTTPS access, about 4 GB of free disk space.
set -euo pipefail
cd "$(dirname "$0")/../.."
APP="$(pwd)"
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

echo "== checks"
mem_gb=$(awk '/MemTotal/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)
echo "memory: ${mem_gb} GB (the API needs about 3 GB)"
echo "cpu cores: $(nproc)   architecture: $(uname -m)"
df -h "$HOME" | tail -1 | awk '{print "free disk in home: " $4}'

echo "== uv and Python 3.12"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv venv "$APP/.venv" --python 3.12 --allow-existing

echo "== Python packages (CPU torch, about 1 GB download)"
uv pip install --python "$APP/.venv/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$APP/.venv/bin/python" -r "$APP/deploy/server/requirements.txt"

echo "== solver data"
"$APP/.venv/bin/python" "$APP/server/fetch_data.py"

echo "== cloudflared"
if ! command -v cloudflared >/dev/null 2>&1; then
    case "$(uname -m)" in
        x86_64|amd64) arch=amd64 ;;
        aarch64|arm64) arch=arm64 ;;
        armv7l) arch=arm ;;
        *) echo "unknown architecture $(uname -m)"; exit 1 ;;
    esac
    curl -fL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${arch}" \
        -o "$HOME/.local/bin/cloudflared"
    chmod +x "$HOME/.local/bin/cloudflared"
fi
cloudflared --version

if [ ! -f "$APP/deploy/server/.env" ]; then
    cp "$APP/deploy/server/.env.example" "$APP/deploy/server/.env"
    echo "created deploy/server/.env (put the tunnel token in it)"
fi
echo
echo "Done. Test:   bash deploy/server/run.sh --quick"
echo "Then:         bash deploy/server/run.sh        (named tunnel; token in deploy/server/.env)"

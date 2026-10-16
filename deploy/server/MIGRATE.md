# Move the API to a new server

This procedure moves the API from the old server (no Docker, `run.sh`) to a new server with
Docker. It also turns on the protections. Do the steps in this order: the website must have the
human-check key before the server asks for tokens.

Downtime: about 1 minute (step 6).

## 1. Upload the data (on the PC)

The new server downloads the data from the Hugging Face dataset. The dataset must contain the
CraterNet catalog files.

```bash
python scripts/deploy_hf.py --user turtlely --data-only
```

## 2. Make a Turnstile widget (Cloudflare dashboard)

1. Go to **Turnstile → Add widget**.
2. Name: `lunar`. Hostname: `lunar.rtang.dev`. Widget mode: **Managed**.
3. Copy the **Site Key** (public) and the **Secret Key** (private).

## 3. Update the website (on the PC)

1. In `web/config.js`, put the site key in `turnstileSiteKey` (the second value, for the public
   site).
2. Commit and push. GitHub Pages publishes the site.

The old server has no secret, so the website sends no token yet. Solves continue to work.

## 4. Make a new tunnel (Cloudflare dashboard)

Make a new tunnel. Do not reuse the old one: its token was on the insecure server.

1. Go to **Zero Trust → Networks → Tunnels → Create a tunnel → Cloudflared**.
2. Name: `lunar-api-2`.
3. Copy the token: the long string after `--token` in the install command. Do not run the
   install command.
4. Do not add a public hostname yet. Click past that step.

## 5. Set up the new server

The server needs at least 6 GB of RAM (8 GB is better), 2 CPU cores and 15 GB of free disk.

1. Install Docker (needs `sudo`), then log out and log in again:

   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker $USER
   ```

2. Copy the code. On the PC, in the repository folder:

   ```bash
   tar -czf lunar-deploy.tgz --exclude=__pycache__ --exclude=deploy/server/.env --exclude="deploy/server/api.log*" .dockerignore deploy/server lunarsolve server weights
   ```

   ```bash
   scp lunar-deploy.tgz USER@NEW_SERVER:~
   ```

   On the new server:

   ```bash
   mkdir -p ~/lunar-platesolve && tar -xzf ~/lunar-deploy.tgz -C ~/lunar-platesolve && rm ~/lunar-deploy.tgz
   ```

   (After you push all changes, `git clone https://github.com/Turtlely/lunar-platesolve` also
   works. Then copy `weights/craternet_v3.pt` with `scp`, because it is not in git.)

3. Make the settings file:

   ```bash
   cd ~/lunar-platesolve/deploy/server
   cp .env.example .env
   chmod 600 .env
   nano .env
   ```

   Set `TUNNEL_TOKEN` (step 4) and `TURNSTILE_SECRET` (step 2). If the server has less than 8 GB
   of RAM, set `MEM_LIMIT` to about 1 GB less than the RAM.

4. Build and start. The first build downloads about 1.5 GB and takes 10–20 minutes:

   ```bash
   docker compose up -d --build
   ```

5. Check. The API loads in 1–3 minutes. Then the `api` service is `healthy`:

   ```bash
   docker compose ps
   docker compose logs --tail 20 api
   ```

   The log must show `human check on`. In the Cloudflare dashboard, the new tunnel must show
   **Healthy**.

## 6. Move the hostname to the new tunnel (Cloudflare dashboard)

1. Open the **old** tunnel → **Public Hostname**. Delete `lunar-api.rtang.dev`. The API is now
   offline.
2. Open the **new** tunnel → **Public Hostname → Add**. Subdomain `lunar-api`, domain
   `rtang.dev`, type `HTTP`, URL `localhost:8000`. Save.
3. Test from any computer:

   ```bash
   curl https://lunar-api.rtang.dev/api/health
   ```

   ```bash
   curl -F image=@photo.png https://lunar-api.rtang.dev/api/solve
   ```

   The first command must show `"human_check":true`. The second must give "Human check
   missing" (HTTP 403). Then solve an example on https://lunar.rtang.dev.

## 7. Add a Cloudflare rate limit (Cloudflare dashboard)

This limit stops floods before they reach the server.

1. Go to **rtang.dev → Security → WAF → Rate limiting rules → Create rule**.
2. Name: `lunar solve`. Condition: **Hostname** equals `lunar-api.rtang.dev` **and** **URI Path**
   equals `/api/solve`.
3. Rate: 3 requests per 10 seconds, per IP. Action: **Block** for 10 seconds. (The free plan
   allows only the 10-second period.)
4. Deploy.

## 8. Remove the old server

1. Delete the old tunnel: **Zero Trust → Networks → Tunnels → old tunnel → Delete**. This makes
   the old token useless.
2. On the old server, stop the API and the tunnel:

   ```bash
   pkill -f "server/api.py"; pkill -f cloudflared
   ```

   ```bash
   ps aux | grep -E "api.py|cloudflared" | grep -v grep
   ```

   The second command must show nothing.
3. Remove the automatic start. Look for lines with `lunar` or `run.sh`:

   ```bash
   crontab -l
   ```

   If there are such lines, run `crontab -e` and delete them. If you started the server in
   `tmux` or `screen`, close that session (`tmux ls`, then `tmux kill-session -t NAME`).
4. Delete the files. This cannot be undone:

   ```bash
   rm -rf ~/lunar-platesolve ~/.cache/lunarsolve ~/.cache/huggingface ~/.local/bin/cloudflared ~/.cloudflared
   ```

   If no other program on the old server uses `uv`, also delete it:

   ```bash
   rm -rf ~/.local/bin/uv ~/.local/bin/uvx ~/.local/share/uv ~/.cache/uv
   ```

## Later updates on the new server

Copy the new code (step 5.2), then:

```bash
cd ~/lunar-platesolve/deploy/server && docker compose up -d --build
```

To stop: `docker compose down`. Logs: `docker compose logs -f api`.

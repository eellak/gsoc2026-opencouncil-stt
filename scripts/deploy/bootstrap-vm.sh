#!/usr/bin/env bash
# One-shot VM bootstrap for the OpenCouncil review UI.
#
# Run ONCE on a fresh Oracle Cloud Always Free Ubuntu 24.04 box:
#   ssh oracle-vm 'bash -s' < scripts/deploy/bootstrap-vm.sh
#
# Idempotent — re-runs are safe; each step skips when the desired state is
# already in place. After this finishes you still need to:
#   1) Add the GitHub Actions deploy key to ~/.ssh/authorized_keys with a
#      command="..." restriction (see scripts/deploy/README.md).
#   2) rsync the initial groups.v1.sqlite from your laptop.
#   3) Open ports 80 + 443 in the Oracle Console VCN Security List ingress
#      rules — that lives in the cloud control plane and cannot be set from
#      inside the VM.

set -euo pipefail

# --- Visible logging ------------------------------------------------------
step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

REPO_URL="https://github.com/eellak/gsoc2026-opencouncil-stt.git"
REPO_DIR="$HOME/opencouncil-fine-tuning"
BRANCH="codex/file-backed-review-ui"
SERVICE_USER="ubuntu"
APP_PORT=3000
PUBLIC_HOST="79-76-114-184.sslip.io"

step "apt: refresh + base packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
	build-essential git curl ca-certificates gnupg lsb-release \
	debian-keyring debian-archive-keyring apt-transport-https \
	iptables-persistent fail2ban unzip

step "swap: 1 GB swapfile (one-shot)"
if ! sudo swapon --show | grep -q /swapfile; then
	sudo fallocate -l 1G /swapfile
	sudo chmod 600 /swapfile
	sudo mkswap /swapfile
	sudo swapon /swapfile
	# Persist across reboots.
	if ! grep -q '/swapfile' /etc/fstab; then
		echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
	fi
	echo "swap enabled."
else
	echo "swap already active — skipped."
fi

step "node: install Node 22 LTS (NodeSource)"
if ! command -v node >/dev/null || ! node --version | grep -qE '^v(22|24|25)\.'; then
	curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
	sudo apt-get install -y -qq nodejs
fi
node --version

step "bun: install (used for install/build, not runtime)"
if ! command -v bun >/dev/null && [ ! -x "$HOME/.bun/bin/bun" ]; then
	curl -fsSL https://bun.sh/install | bash
fi
"$HOME/.bun/bin/bun" --version

step "caddy: install via Cloudsmith repo"
if ! command -v caddy >/dev/null; then
	curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
		| sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
	curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
		| sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
	sudo apt-get update -qq
	sudo apt-get install -y -qq caddy
fi
caddy version

step "iptables: open 80/443 (Oracle Ubuntu blocks them by default)"
# Oracle's stock INPUT chain ends with a catch-all
# `REJECT all -- 0.0.0.0/0 0.0.0.0/0 reject-with icmp-host-prohibited`
# rule. New ACCEPT rules MUST land before that REJECT or they're dead. We
# detect its position dynamically rather than hardcode a slot, because the
# chain length can shift between Ubuntu cloud-image revisions.
REJECT_POS=$(sudo iptables -L INPUT -n --line-numbers | awk '$2 == "REJECT" {print $1; exit}')
if [ -z "${REJECT_POS:-}" ]; then
	REJECT_POS=1
fi
for PORT in 80 443; do
	if ! sudo iptables -C INPUT -p tcp -m tcp --dport "$PORT" -j ACCEPT 2>/dev/null; then
		sudo iptables -I INPUT "$REJECT_POS" -p tcp -m tcp --dport "$PORT" -j ACCEPT
		echo "iptables ACCEPT :$PORT inserted at position $REJECT_POS."
	else
		echo "iptables ACCEPT :$PORT already present."
	fi
done
sudo netfilter-persistent save

step "caddy: Caddyfile reverse-proxy to localhost:${APP_PORT}"
# A bare `:80 { reverse_proxy ... }` would shadow Caddy's automatic ACME
# listener and intercept /.well-known/acme-challenge/... requests, breaking
# cert renewal. So we leave port 80 to Caddy's auto-https machinery (which
# also handles the HTTP→HTTPS redirect for the named host).
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
${PUBLIC_HOST} {
	encode gzip
	request_body {
		max_size 5MB
	}
	reverse_proxy localhost:${APP_PORT}
}
EOF
sudo systemctl enable caddy >/dev/null
sudo systemctl restart caddy

step "clone repo + branch (if missing)"
if [ ! -d "$REPO_DIR/.git" ]; then
	git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --quiet origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"

step "systemd: opencouncil-ui service unit"
sudo tee /etc/systemd/system/opencouncil-ui.service >/dev/null <<EOF
[Unit]
Description=OpenCouncil Review UI (SvelteKit + SQLite)
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${REPO_DIR}/ui
Environment=NODE_ENV=production
Environment=PORT=${APP_PORT}
Environment=HOST=127.0.0.1
Environment=REVIEW_REPO=sqlite
# Runs the adapter-node build on stock Node. We deliberately do not use Bun
# at runtime because better-sqlite3's native binding doesn't load under Bun
# (see https://github.com/oven-sh/bun/issues/4290).
ExecStart=/usr/bin/node build/index.js
Restart=on-failure
RestartSec=5
# Keep the unit's allocations within the box's headroom — leaves plenty for
# Caddy + OS page cache. If we ever exceed this we want to see it, not OOM
# the whole VM.
MemoryMax=720M

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload

step "sudoers: allow ${SERVICE_USER} to restart the service without password"
sudo tee /etc/sudoers.d/opencouncil-ui >/dev/null <<EOF
${SERVICE_USER} ALL=(root) NOPASSWD: /bin/systemctl restart opencouncil-ui, /bin/systemctl status opencouncil-ui, /bin/systemctl reload caddy
EOF
sudo chmod 0440 /etc/sudoers.d/opencouncil-ui

step "done — next steps"
cat <<EOF

Bootstrap complete on \$(hostname).

Not done by this script (because they require off-VM context):

  1) From your laptop, rsync the initial SQLite cache:
       cd <repo-root>
       bun --cwd ui run build:cache -- --format sqlite
       rsync -avz --progress ui/.cache/groups.v1.sqlite \\
         oracle-vm:${REPO_DIR}/ui/.cache/groups.v1.sqlite.incoming
       ssh oracle-vm 'mv ${REPO_DIR}/ui/.cache/groups.v1.sqlite.incoming \\
                        ${REPO_DIR}/ui/.cache/groups.v1.sqlite'

  2) Install deps + build + start:
       ssh oracle-vm 'cd ${REPO_DIR}/ui && ~/.bun/bin/bun install --frozen-lockfile && \\
                      ~/.bun/bin/bun run build && \\
                      sudo systemctl enable --now opencouncil-ui'

  3) Add the GitHub Actions deploy public key to ~/.ssh/authorized_keys with:
       command="${REPO_DIR}/scripts/deploy/deploy.sh",no-port-forwarding,no-agent-forwarding,no-X11-forwarding \\
       ssh-ed25519 AAAA... github-actions

  4) In the Oracle Console, open VCN -> Security Lists -> Default Security List
     -> Ingress Rules. Add:
        Source 0.0.0.0/0 / TCP / Destination port 80
        Source 0.0.0.0/0 / TCP / Destination port 443
     iptables on the VM was already opened by this script, but the VCN
     security list is a separate, mandatory layer.

  5) Smoke test from anywhere:
       curl -fsS https://${PUBLIC_HOST}/api/stats | head -c 200

EOF

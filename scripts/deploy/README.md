# Oracle VM deploy

Self-hosted SvelteKit (adapter-node) deploy of the file-backed review UI on
the Oracle Cloud Always Free VM (`oracle-vm` in `~/.ssh/config`, AMD shape,
1 vCPU / 1 GB RAM, Ubuntu 24.04).

## Topology

```
Mac (laptop)                 GitHub                    Oracle VM
─────────────                ──────                    ─────────
git push   ──────────►  Actions runner  ─SSH─►  deploy.sh
                                                   │
                                                   ├─ git fetch + reset
                                                   ├─ bun install + build
                                                   └─ systemctl restart

build:cache  ─rsync .sqlite─────────────────────►  ui/.cache/groups.v1.sqlite

                                       Caddy ◄── public 80/443 ◄── client
                                         │
                                         └── reverse_proxy → 127.0.0.1:3000 (Node)
```

Code lives in git; data (`groups.v1.sqlite`) lives outside git on the VM and
is refreshed manually via `rsync` when the source CSV changes.

## First-time setup

1. **Bootstrap the box** (one shot):
   ```bash
   ssh oracle-vm 'bash -s' < scripts/deploy/bootstrap-vm.sh
   ```
   That installs Node 22, Bun, Caddy; sets up swap, iptables, a systemd
   service, and a `sudoers.d` entry so the deploy script can restart the
   service without a password.

2. **Open VCN ingress 80/443** in the Oracle Console — this is separate
   from iptables and the script can't do it from inside the VM.

3. **Build the SQLite cache locally and push it**:
   ```bash
   bun --cwd ui run build:cache -- --format sqlite
   rsync -avz --progress ui/.cache/groups.v1.sqlite \
     oracle-vm:opencouncil-fine-tuning/ui/.cache/groups.v1.sqlite.incoming
   ssh oracle-vm '
     mv opencouncil-fine-tuning/ui/.cache/groups.v1.sqlite.incoming \
        opencouncil-fine-tuning/ui/.cache/groups.v1.sqlite &&
     sudo /bin/systemctl restart opencouncil-ui
   '
   ```
   The `.incoming` → rename dance keeps the running service from reading a
   half-written file.

4. **First deploy** (afterwards GitHub Actions does this on every push):
   ```bash
   ssh oracle-vm 'bash /home/ubuntu/opencouncil-fine-tuning/scripts/deploy/deploy.sh'
   ```

5. **Wire up GitHub Actions** — see the next section.

## GitHub Actions wiring

Generate a deploy-only keypair (do NOT reuse your personal `oracle-vm.key`):

```bash
ssh-keygen -t ed25519 -f /tmp/oc-deploy -N '' -C 'github-actions deploy'
```

Append the **public** key to the VM with a hard restriction that forces every
authenticated session to run the deploy script:

```bash
ssh oracle-vm 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
{
  echo -n 'command="/home/ubuntu/opencouncil-fine-tuning/scripts/deploy/deploy.sh",no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-user-rc '
  cat /tmp/oc-deploy.pub
} | ssh oracle-vm 'cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

Then, in **GitHub → Settings → Secrets and variables → Actions**:

| Secret name        | Value                                                  |
|--------------------|--------------------------------------------------------|
| `DEPLOY_SSH_KEY`   | The full contents of `/tmp/oc-deploy` (the **private** key). |
| `DEPLOY_HOST`      | `79.76.114.184`                                        |

Finally delete the local copies you no longer need:

```bash
shred -u /tmp/oc-deploy /tmp/oc-deploy.pub
```

## Updating

| Change            | What you do                                        |
|-------------------|----------------------------------------------------|
| UI code           | `git push origin codex/file-backed-review-ui` — Actions does the rest. |
| Source CSV / data | Rebuild SQLite locally + `rsync` (see step 3).     |
| Caddy / systemd   | Edit `bootstrap-vm.sh`, rerun it (idempotent).     |

## Monitoring

```bash
ssh oracle-vm 'sudo journalctl -u opencouncil-ui -n 100 --no-pager'
ssh oracle-vm 'systemctl status opencouncil-ui'
ssh oracle-vm 'sudo journalctl -u caddy -n 50 --no-pager'
```

## Why the choices

- **Node, not Bun, at runtime.** `better-sqlite3` doesn't load under Bun yet
  (oven-sh/bun#4290). Bun is still used for install + build because it's
  faster than npm and the lockfile is already a `bun.lock`.
- **Caddy not nginx.** Auto Let's Encrypt against `79-76-114-184.sslip.io`
  means TLS works without owning a domain.
- **systemd `MemoryMax=720M`.** Caps the process so a runaway iteration
  can't take down the whole VM (and Caddy with it). Crash → systemd
  restart → systemd log entry, which is observable.
- **Mutable state outside git.** `ui/.state/` (sidecar) and `ui/.cache/`
  (the SQLite + JSON caches) are gitignored, so `deploy.sh`'s
  `git reset --hard` doesn't touch them.

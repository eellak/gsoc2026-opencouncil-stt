# Disaster recovery — review UI

How to bring the OpenCouncil review UI back up on a fresh host if the Oracle VM
is lost (reclaimed, disk failure, etc.), and what must be preserved so nothing is
lost. Written 2026-06-03 from the live VM; update when the deploy changes.

## Mental model: critical vs derivable

Only three things are genuinely critical — everything else is regenerated from
them on first boot.

| Artifact | Where it lives | Backed up? | If lost… |
|---|---|---|---|
| **Code** | GitHub `eellak/gsoc2026-opencouncil-stt`, branch `main` | ✅ git | re-clone |
| **Flags / labels** (`ui/.state/review-events.jsonl`, append-only, source of truth) | VM only at runtime | ✅ daily → private repo `angelospk/oc_review_log` (gzip) + ad-hoc copy on the Mac (`~/opencouncil-flag-backups/`) | restore from `oc_review_log` |
| **Corpus source** (`data-1779206108158.csv`, ~235 MB) | Mac (`/Users/harold/projects/opencouncil-fine-tuning/`) — gitignored, NOT on the VM | ✅ gzip (~34 MB) in `oc_review_log` under `corpus/` | restore from `oc_review_log/corpus/` |
| Built SQLite (`ui/.cache/groups.v1.sqlite`, ~580 MB) | VM only | ➖ derivable from the CSV via `build-cache` | rebuild from CSV |
| `.state/*.snapshot.json` (labels, category-index ~138 MB, stats, meeting-eligibility) | VM | ➖ all derived from `review-events.jsonl` + the SQLite | regenerate on boot (~30 s of scans) |
| Audio | remote CDN (`data.opencouncil.gr`, `audio_url` per group) | n/a — never stored locally | nothing to restore |

**Copies of the corpus today:** Mac CSV + VM SQLite (derived) + gzipped CSV in
`oc_review_log/corpus/` (durable, off-box). All three would have to be lost before
a re-export from OpenCouncil is needed.

## Current production setup (snapshot 2026-06-03)

- **Host:** Oracle free VM, Ubuntu, `ssh oracle-vm`, public IP `79.76.114.184`.
- **Runtime:** Node **v22** (`node build/index.js`, adapter-node). **No Bun at
  runtime** — `better-sqlite3`'s native binding doesn't load under Bun.
- **Service:** systemd `opencouncil-ui.service`, `User=ubuntu`,
  `WorkingDirectory=/home/ubuntu/opencouncil-fine-tuning/ui`. Env:
  `PORT=3000`, `HOST=127.0.0.1`, `REVIEW_REPO=sqlite`, `NODE_ENV=production`,
  `NODE_OPTIONS=--max-old-space-size=768`, `MemoryMax=880M`. (Optional:
  `MEETING_MIN_HUMAN_UTTERANCES=10` is the default; set `0` to disable the
  eligibility filter.)
- **Reverse proxy:** **Caddy** (auto-TLS), `/etc/caddy/Caddyfile`:
  domain `79-76-114-184.sslip.io` → `reverse_proxy localhost:3000`, `encode gzip`,
  `request_body max_size 5MB`. The domain is derived from the IP via sslip.io, so
  a new host gets a new `<new-ip>.sslip.io` automatically.
- **Flags backup:** `~/flags-backup/backup.sh` + systemd `flags-backup.timer`
  (daily 04:00 UTC, `Persistent=true`), pushing to `oc_review_log` via a deploy
  key at `~/.ssh/flags_backup_ed25519` (write-scoped to that one repo).

## Recovery runbook (new host)

1. **Provision** an Ubuntu host. Install Node 22 (e.g. nodesource) and Caddy.
   Add an SSH alias `oracle-vm` (or similar) for convenience.

2. **Code:**
   ```bash
   cd ~ && git clone https://github.com/eellak/gsoc2026-opencouncil-stt.git opencouncil-fine-tuning
   cd opencouncil-fine-tuning && git checkout main
   ```

3. **Corpus → SQLite.** Copy the source CSV up (from the Mac), then build:
   ```bash
   scp data-1779206108158.csv NEWHOST:~/opencouncil-fine-tuning/
   cd ~/opencouncil-fine-tuning/ui && npm ci
   npx tsx scripts/build-cache.ts ../data-1779206108158.csv .cache --format sqlite
   # → ui/.cache/groups.v1.sqlite  (verify cache_version matches CACHE_VERSION in src)
   ```
   If the Mac CSV is gone, restore it from the backup repo first:
   `gzip -dc oc_review_log/corpus/data-1779206108158.csv.gz > data-1779206108158.csv`.
   Or, if the VM's SQLite survived, just copy `ui/.cache/groups.v1.sqlite` over and
   skip the rebuild entirely.

4. **Flags.** Restore the append-only log from the backup repo:
   ```bash
   git clone git@github.com:angelospk/oc_review_log.git /tmp/flags
   mkdir -p ~/opencouncil-fine-tuning/ui/.state
   gzip -dc /tmp/flags/review-events.jsonl.gz \
     > ~/opencouncil-fine-tuning/ui/.state/review-events.jsonl
   # Do NOT copy old *.snapshot.json — the server rebuilds them by replaying the log.
   ```

5. **Build the app:**
   ```bash
   cd ~/opencouncil-fine-tuning/ui && npm run build   # → build/
   ```

6. **systemd service.** Recreate `/etc/systemd/system/opencouncil-ui.service` with
   the env block above (`node build/index.js`), then
   `sudo systemctl enable --now opencouncil-ui.service`.

7. **Caddy.** Put the Caddyfile in place with the new host's
   `<new-ip>.sslip.io` domain → `reverse_proxy localhost:3000`, reload Caddy.

8. **Re-arm flag backups on the new host** (the old deploy key stays on the dead
   VM): regenerate a key, add it to `oc_review_log` as a write deploy key, and
   recreate `~/flags-backup/` + the `flags-backup.timer` (see "Current setup").
   ```bash
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/flags_backup_ed25519 -C flags-backup@newvm
   gh api repos/angelospk/oc_review_log/keys -f title=newvm \
     -f key="$(cat ~/.ssh/flags_backup_ed25519.pub)" -F read_only=false
   ```

9. **First boot is slow once:** the first request triggers the meeting-eligibility
   scan (~17 s) and the first `/category` visit builds the category index (~17 s).
   Both persist to `.state/` and are instant thereafter. Confirm the queue total
   (~269 k eligible utterances) and that a few utterances load.

## Two writers to `oc_review_log`

Both the VM (daily flags) and the Mac (corpus, ad-hoc) push to this repo. To avoid
non-fast-forward rejects, the VM's `backup.sh` does `git fetch && git reset --hard
origin/main` before regenerating its gz — so it always fast-forwards. Push corpus
or other large artifacts from the Mac freely; the next VM run just syncs them down.

## Open improvements (not done yet)

- **Backup verification.** The flags backup is verified at write time (event count
  in the commit message) but there's no periodic restore test.
- **`utterance-edits-may12-26.csv`** (~208 MB, an earlier export) is also Mac-only;
  not backed up since `data-1779206108158.csv` is the corpus the SQLite is built
  from. Add it to `corpus/` too if it turns out to matter.
- **Infra-as-text.** The systemd units and Caddyfile are reproduced in this doc;
  if they change on the VM, update here in the same edit.

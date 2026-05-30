# Oracle Free VM

Personal Oracle Cloud Always Free instance used for ad-hoc remote work for this project (e.g. long-running scripts, throwaway services). Not yet load-bearing — no production data lives here.

## Box

- **Public IP:** `79.76.114.184`
- **Shape:** AMD, 1 OCPU / 1 GB RAM (Always Free)
- **OS:** Ubuntu 24.04 LTS (`instance-20260522-1240`)
- **Provisioned:** 2026-05-22
- **User:** `ubuntu`

## SSH

The private key lives at `~/.ssh/oracle-vm.key` (mode 600). It is **never** committed — `.gitignore` blocks `*.key` / `*.pem` repo-wide, and the file is outside the repo anyway.

Original download: `~/Downloads/chuck/ssh-key-2026-05-22 (4).key` (kept as backup; Oracle re-issues fresh keys on every provision, so this is the most recent one).

`~/.ssh/config` entry:

```
Host oracle-vm
    HostName 79.76.114.184
    User ubuntu
    IdentityFile ~/.ssh/oracle-vm.key
    IdentitiesOnly yes
```

### Connect

```bash
ssh oracle-vm
```

That is the only command you need day-to-day. Equivalent long form, if the config is missing:

```bash
ssh -i ~/.ssh/oracle-vm.key ubuntu@79.76.114.184
```

### File transfer

```bash
scp ./local-file oracle-vm:/home/ubuntu/
rsync -avz ./dir/ oracle-vm:/home/ubuntu/dir/
```

## Re-provisioning notes

If Oracle reclaims the instance (Always Free idle reclaim is a real thing) or you rebuild it:

1. Download the new private key into `~/Downloads/chuck/`.
2. Replace `~/.ssh/oracle-vm.key` with the new key (`chmod 600`).
3. Remove the stale host key: `ssh-keygen -R 79.76.114.184`.
4. Update the IP in `~/.ssh/config` and in this doc if it changed.

## What it is used for

- _(nothing yet — update this section as services land on the box)_

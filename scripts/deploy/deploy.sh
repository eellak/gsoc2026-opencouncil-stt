#!/usr/bin/env bash
# Re-runnable deploy executed on the Oracle VM. Triggered by GitHub Actions
# via a restricted SSH key (see .github/workflows/deploy.yml +
# ~/.ssh/authorized_keys command="..."), or by you manually:
#   ssh oracle-vm bash /home/ubuntu/opencouncil-fine-tuning/scripts/deploy/deploy.sh
#
# Mutable state (`ui/.state/`, `ui/.cache/`) is in .gitignore so `git reset
# --hard` leaves it alone. If that ever changes, this script will start
# wiping user labels — so test gitignore coverage when refactoring.

set -euo pipefail

REPO_DIR="$HOME/opencouncil-fine-tuning"
BRANCH="main"
BUN="$HOME/.bun/bin/bun"

cd "$REPO_DIR"
git fetch --quiet origin "$BRANCH"
git reset --hard "origin/$BRANCH"

cd ui
"$BUN" install --frozen-lockfile
"$BUN" run build

# Restart through sudo NOPASSWD whitelist set up by bootstrap-vm.sh.
sudo /bin/systemctl restart opencouncil-ui

echo "deployed $(git -C "$REPO_DIR" rev-parse --short HEAD)"

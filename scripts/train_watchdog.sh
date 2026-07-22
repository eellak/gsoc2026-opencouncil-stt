#!/bin/bash
# Watchdog for an unattended RunPod training run on a COMMUNITY pod: every 25 min
# it backs up the pod's checkpoints to the local machine (durable — survives a pod
# reclaim) and detects termination. Exits (-> agent notification) on:
#   TRAINING DONE   - the adapter was saved
#   TRAINING CRASHED - a fatal error appeared in the log
#   POD DOWN        - ssh failed repeatedly (pod likely reclaimed) -> needs restart
# Usage: train_watchdog.sh <ip> <port>
IP="$1"; PORT="$2"
KEY=/home/harold/.ssh/id_ed25519
SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
DEST=/home/harold/oc-train-checkpoints
mkdir -p "$DEST"
fails=0
while true; do
  if scp $SSHOPT -P "$PORT" "root@$IP:/workspace/train.log" "$DEST/train.log" 2>/dev/null; then
    fails=0
    rsync -az -e "ssh $SSHOPT -p $PORT" \
      "root@$IP:/workspace/whisper-run/adapter/" "$DEST/adapter/" 2>/dev/null || true
    nck=$(ls -d "$DEST"/adapter/checkpoint-* 2>/dev/null | wc -l)
    last=$(grep -E "AFTER|BASELINE|built |datasets:|epoch" "$DEST/train.log" 2>/dev/null | tail -1)
    echo "$(date -u +%H:%M) sync ok | local checkpoints=$nck | $last"
    grep -qE "adapter saved ->" "$DEST/train.log" && { echo "TRAINING DONE"; break; }
    grep -qE "train FATAL|Traceback|CUDA out of memory" "$DEST/train.log" && { echo "TRAINING CRASHED"; break; }
    # FREEZE detection: process alive + ssh ok but log not written in >35min =
    # a frozen community pod (throttle). ssh-liveness alone misses this.
    age=$(ssh $SSHOPT -p "$PORT" "root@$IP" 'echo $(( $(date +%s) - $(stat -c %Y /workspace/train.log) ))' 2>/dev/null)
    if [ -n "$age" ] && [ "$age" -gt 2100 ]; then
      echo "RUN FROZEN — no log write in ${age}s (pod throttled) — needs restart"; break
    fi
  else
    fails=$((fails + 1))
    echo "$(date -u +%H:%M) ssh fail $fails/3"
    [ "$fails" -ge 3 ] && { echo "POD DOWN — needs restart from $DEST/adapter"; break; }
  fi
  sleep 1500
done
echo "checkpoints backed up locally at $DEST/adapter :"; ls "$DEST/adapter/" 2>/dev/null
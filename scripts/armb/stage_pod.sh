set -e
export AWS_ACCESS_KEY_ID="$1" AWS_SECRET_ACCESS_KEY="$2" AWS_DEFAULT_REGION=EUR-IS-1
E=https://s3api-eur-is-1.runpod.io/
B=s3://qzw88vdwv2
mkdir -p /workspace/oc
cd /workspace/oc
pip install --break-system-packages -q awscli 2>&1 | tail -2
echo "== syncing bundle =="
aws s3 sync $B/oc-bundles/clean-pack-screen-736cedc61ce3a5fe/ /workspace/oc/bundle/ --endpoint-url $E --region EUR-IS-1 --only-show-errors
du -sh /workspace/oc/bundle
echo "== installing deps =="
pip install --break-system-packages -q -r /workspace/oc/bundle/code/notebooks/requirements-runpod.txt 2>&1 | tail -3
echo "== syncing spliced packs =="
aws s3 sync $B/packs/spliced-jitter/ /workspace/oc/spliced/ --endpoint-url $E --region EUR-IS-1 --only-show-errors
du -sh /workspace/oc/spliced
echo "STAGE_OK"

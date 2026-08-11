# OpenCouncil benchmark HTTP API

Status: available · Last verified: 2026-08-10 · Ledger capability: `cap-bench-http-api`

Everything the benchmark UI does is available over HTTP. **An agent can evaluate a
model end to end without a human touching a browser.** If you are about to tell the
user "someone has to start this run", you are wrong — read this first.

The full vendor manual is cached at
[`.claude/skills/transcription-benchmark/SKILL.md`](../../.claude/skills/transcription-benchmark/SKILL.md)
(gitignored) and served at `https://bench.opencouncil.gr/api/skill`. This runbook is
the project-specific part: what we actually run, and what has bitten us.

## Auth

```bash
source .env            # BENCH_API_KEY=tbk_...
AUTH="Authorization: Bearer $BENCH_API_KEY"
BASE=https://bench.opencouncil.gr
```

The key lives in `.env` (gitignored) under `BENCH_API_KEY`. Never commit it, never
put it in a provider `baseURL`, never paste it into a report — provider configs are
published inside every public `report.json`.

**Report endpoints are public.** `GET /api/runs/<id>/report.json` and
`/report` need no key at all. Reading past results costs nothing and requires
nothing.

## Cheapest smoke check

```bash
curl -s -H "$AUTH" $BASE/api/config/providers | head -c 300
```

A provider list back means the key works. Do this before concluding the API is
unavailable.

## The run we clone

`2026-07-23-june-benchmark-runpod-gpu-finetune-corre` — 260 windows, 10 cities,
79,114 reference words, 8 providers already scored. Cloning copies **every existing
provider's results for free**; you pay only to transcribe with the new provider.

Metrics: `wer`, `wer-nodiacritics`, `wer-nofillers`, `cer`. Lower is better.

## The loop

```bash
# 1. read the provider list, append yours, PUT the whole list back
curl -s -H "$AUTH" $BASE/api/config/providers > /tmp/prov.json
# ... edit ... then:
curl -s -X PUT -H "$AUTH" -H 'Content-Type: application/json' \
  $BASE/api/config/providers -d @/tmp/prov_new.json

# 2. give the provider its own key, if the endpoint needs one
curl -s -X PUT -H "$AUTH" -H 'Content-Type: application/json' \
  $BASE/api/config/keys -d '{"instanceId":"<id>","apiKey":"<secret>"}'

# 3. one real ~3 min clip through every enabled provider, before paying for a run
curl -s -X POST -H "$AUTH" $BASE/api/config/providers/test
curl -s -H "$AUTH" $BASE/api/config/providers/test/status

# 4. clone with ONLY the new provider
curl -s -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  $BASE/api/runs/2026-07-23-june-benchmark-runpod-gpu-finetune-corre/clone \
  -d '{"name":"...","providerInstanceIds":["<id>"]}'

# 5. start, then poll every ~30s until manifest.status == "completed"
curl -s -X POST -H "$AUTH" $BASE/api/runs/<cloneId>/start
curl -s -H "$AUTH" $BASE/api/runs/<cloneId>

# 6. read (public)
curl -s $BASE/api/runs/<cloneId>/report.json
```

Re-POSTing `/start` after errors re-runs **only the failed items**.

## What has bitten us

**Serving on CPU does not work.** The mini-PC provider scored 238/260 with 22
timeouts at 128.1s mean latency. The same model on a RunPod GPU scored 260/260 with
0 errors at 11.3s. The ~100s proxy/edge limit is real but a GPU sits an order of
magnitude under it. Serve on GPU, concurrency 1.

**Expose `8000/http` when you create the pod.** A pod created with only `22/tcp` is
unreachable by the benchmark and cannot be fixed without recreating it. The provider
URL is then `https://<podId>-8000.proxy.runpod.net/v1`.

**Disable your old provider instances.** Stale entries pointing at dead pod URLs will be
re-run and fail if left enabled. Removing them from `/api/config/providers` is safe —
completed runs keep their own copy of the results and the `configSnapshot`, so history
survives.

**Deleting a provider does not clean up past leaderboards.** For the same reason: a
completed run's report still lists every provider that ever ran in it. There is no
documented endpoint that removes one — `POST /api/runs/:id/providers` only adds. Decide
which providers belong in a run *before* you start it.

**Rename the `label`, never the `id`.** The `instanceId` is the key a completed run uses
to attach its results; changing it orphans the measurement you paid for.

## Reading the result honestly

Two rules, both learned the expensive way:

- **This benchmark's reference is OpenCouncil's own published transcript.** It
  measures *agreement with the product*, not accuracy. Record it, do not let it
  decide. See [the reference problem](../reports/2026-08-03-the-reference-problem.md).
- **105 of the 203 benchmark meetings are in our training set.** Report the
  `city:argos` and `city:orestiada` strata — the only training-disjoint cities —
  separately from the pooled number, and never quote the pool as model quality.

Before quoting any delta, check whether one window carries it. In
`exp-2026-08-10-packed-training` a single window supplied 67% of a headline effect.

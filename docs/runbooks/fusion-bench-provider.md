# Running the fusion route as a benchmark provider

Status: available · Last verified: 2026-09-08 · Ledger capability: `cap-fusion-bench-route`

The fusion route (Scribe v2 + Soniox `stt-async-v5` + `artifact-ct2-cleanpack-cont-s47`,
fused by the rules engine) speaks the OpenAI transcription protocol, so the benchmark can
call it like any other provider. This is the procedure that worked on 2026-09-08, including
the four things that had to be fixed first.

For the benchmark API itself, read [`benchmark-api.md`](benchmark-api.md) — this runbook
only covers what is specific to the fusion route.

## What the route is

- Code: `opencouncil-tasks`, branch `feat/fusion-provider`, entrypoint
  `fusion_bench_server.mjs` (untracked; it wires `mountOpenAiCompatRoute` onto an express
  app with bearer auth). It runs the compiled `dist/`, so **`npm run build` after any
  source change** or the route serves the old code.
- Surface: `POST /v1/audio/transcriptions`, multipart field `file`, body field `model`.
  Accepts `model` ∈ {`scribe`, `fusion-rules`, `fusion-policy-opus`, `fusion-policy-sonnet`},
  `language` absent or `el`, `response_format` absent or `json`. Returns `{"text": …}` only.
- Public at `https://ocfusion.haroldpoi.dev` through the `oc-asr` cloudflared tunnel
  (`~/.cloudflared/config.yml`), and at `https://minipc.tail11f3d4.ts.net` through
  Tailscale Funnel. Cloudflare cuts an origin request at ~100 s (524); Funnel does not.

## Supervise it before a run

A 391-window run is three to four hours. Do not run the route from a shell.

```bash
systemctl --user status fusion-bench-route.service
journalctl --user -u fusion-bench-route -n 50    # or tail the log below
tail -f ~/.cache/oc-public/fusion-bench-route.log
```

The unit is `~/.config/systemd/user/fusion-bench-route.service` (`Restart=always`,
`WorkingDirectory` at the tasks repo) and its secrets live in
`~/.config/fusion-bench/route.env` (mode 600). That file must carry:

| variable | why |
|---|---|
| `API_TOKENS` | **`dist/lib/auth.js` reads it at import time and exits if it is missing**, even though the server passes its tokens explicitly. Write it as `API_TOKENS='["tbp_…"]'` — the single quotes keep systemd from eating the inner ones. |
| `BENCH_PROVIDER_TOKEN` | the bearer token the benchmark sends |
| `SCRIBE_API_KEY`, `ELEVENLABS_API_KEY` | Scribe |
| `SONIOX_API_KEY` | Soniox |
| `RUNPOD_API_KEY`, `OC_ASR_ENDPOINT_ID` | our adapter on the serverless endpoint |
| `PORT` | 8791, matching the tunnel ingress |

`loginctl show-user harold -p Linger` must say `Linger=yes` or the unit dies at logout.

Hand the port over in this order, or the new process fails to bind while the old one keeps
answering health checks: confirm who owns the port (`ss -ltnp | grep 8791`), stop that
process, start the unit, then confirm the listener's pid equals the unit's `MainPID`.

## Register and run

1. **Disable any provider that points at a dead pod.** `POST /api/config/providers/test`
   sends its clip through *every enabled* provider, and a provider can never be removed
   from a run once added. `PUT` the whole list back, then re-`GET` and diff: exactly the
   fields you meant to change, and no provider lost.
2. **Register**, `type: openai-compatible`, `params: {baseURL: "https://ocfusion.haroldpoi.dev/v1",
   model: "fusion-rules", language: "el"}`, `concurrency: 1` (the serverless endpoint has
   `workersMax 1`). Put the token in `PUT /api/config/keys`, **never** in `baseURL` —
   provider params are published inside every public `report.json`.
3. **Check the contract before paying.** `POST /api/config/providers/test`, then read the
   *row for your instance*, not the aggregate: other providers fail for their own reasons.
   Then confirm the trace it produced says `outcome: fused` (below).
4. **Clone, inspect, then start.** Clone with `providerInstanceIds` set to your instance
   alone. Before `/start`, check `progress.perProvider`: every historical provider at
   `done == total` and yours at `0/391`. `POST /api/runs/<id>/pause` works on a running run;
   there is no `/stop` or `/cancel`.

## The gate that matters: every window must actually fuse

`FusionTranscriber` refuses two-system fusion — *any* aux failure returns **exact Scribe**
(`outcome: "scribe-fallback"`), and silence returns Scribe too (`"scribe-only"`). The
benchmark sees only `{"text": …}`, so a run can silently score a mixture of fusion and
Scribe windows and still report zero errors.

Traces are the only place this is visible. They are written per request to
`~/.cache/oc-public/fusion-bench-traces/<audioSha>.<arm>.<requestId>.json`
(never git — they carry word counts and provenance, and the audio they describe is PII).
Audit them against the run:

```bash
python3 eval/bench_report_analysis.py <cached report.json> <your instance> scribe soniox
```

and count outcomes for traces written since the run started. Require:

- every window `outcome: fused`,
- one `configSha` across the whole run (a mid-run restart with different config would show two),
- three components with non-zero `wordCount` on each window.

Report any non-fused window as a Scribe row inside a fusion score, not as an error.

## Known rough edges

- The trace records `model` for the `ours` component as the endpoint id
  (`o1jda6sxo85dnk`) rather than the model.bin hash: the handler returns
  `ct2_model_bin_sha256` while `ocasr.ts` looks for `model_bin_sha256`. Cosmetic for the
  cache — `paramsSha` hashes the whole provenance blob, so the decode config *is* in the
  key — but it makes a trace harder to attribute at a glance.
- The component cache (`~/.cache/oc-public/fusion-bench-cache`) keys on audio sha, provider
  id and that params sha. It survives across runs on purpose, so a retried window is a cache
  hit rather than a second vendor bill. Purge it when the *fusion* configuration changes,
  because the fused output is what stops being comparable.
- `base64` transport tops out near 6.7 MB of audio (~140 s window fits, a 15-minute segment
  does not). Benchmark windows fit; production segments need `audioUrl`.

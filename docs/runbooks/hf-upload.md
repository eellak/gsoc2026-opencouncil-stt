# Publishing the dataset to Hugging Face

How to upload `data/hf-dataset/public/` to the team's Hugging Face **organization**
(or a personal account). Anyone with **write** access to the org can run this.

> **Staging (2026-07-15):** the current build is already uploaded **private** at
> <https://huggingface.co/datasets/haroldpoi/opencouncil-greek-asr> (round-trip
> verified: 28,967 train / 7,879 validation rows). The local `haroldpoi` token
> has no org membership, so the org publish below still needs org write access.
> Once granted, run step 3 against the org repo id, then delete or keep the
> staging copy.

## What gets uploaded

**Only** `data/hf-dataset/public/`:

- `train.parquet` / `validation.parquet` (+ `.jsonl` mirrors)
- `README.md`: the dataset card (renders the HF viewer; license `CC-BY-SA-4.0`)
- `split_assignments.json`: seed + speaker→split map (reproducibility)

Nothing else from `data/hf-dataset/` is uploaded. The internal artifacts
(`raw-export-*.jsonl`, `boundary*.jsonl`, `*-report.md`, `*-audit.csv`,
`align-failed-dropped.csv`) contain reviewer notes / signed URLs and must **never**
be pushed. The command below targets `public/` explicitly, so they stay local.

## Prerequisites

1. A Hugging Face account.
2. **Write access to the org.** You are a *member*, but uploading a dataset needs
   the **write** role. If step 3 fails with a 403, ask an org admin to grant you
   write access (or to create the empty dataset repo once; then you can push).
3. `hf` CLI, already installed here (`.venv-eval/bin/hf`). Elsewhere:
   `pip install -U huggingface_hub`.

## Steps

**1. Create a write token** → <https://huggingface.co/settings/tokens>
→ *New token* → type **Write** → copy it.
(If your org enforces it, make the token org-scoped.)

**2. Log in** (once per machine):

```bash
.venv-eval/bin/hf auth login      # paste the write token when prompted
```

**3. Upload to the org** (start **private**, flip to public later):

```bash
.venv-eval/bin/hf upload <ORG>/opencouncil-greek-asr data/hf-dataset/public . \
  --repo-type=dataset --private
```

- `<ORG>` = the org's HF namespace. Find it at huggingface.co → your avatar →
  *Organizations* (it's the name in the URL, e.g. `huggingface.co/<ORG>`).
- Argument order is `REPO_ID  LOCAL_FOLDER  PATH_IN_REPO`:
  - `<ORG>/opencouncil-greek-asr`: the repo id (namespace/name **only**),
  - `data/hf-dataset/public`: the folder to upload,
  - `.`: put it at the repo root.
- The repo is created automatically on first upload.
- `--private` = visible only to org members. Drop it (or use `--no-private`) to
  make it public.

**4. Check it** → `https://huggingface.co/datasets/<ORG>/opencouncil-greek-asr`
The card + dataset viewer should render.

## Updating it later (same command)

Re-running the exact same `hf upload` overwrites/commits the changed files, so
after regenerating the dataset (`.venv-eval/bin/python -m eval.hf_export.build
finalize`), just run step 3 again. Each upload is a new commit; users can pin a
revision.

## Making it public

When license (confirm with Schema Labs) + the human-gate reports are cleared:

```bash
.venv-eval/bin/hf repo settings <ORG>/opencouncil-greek-asr --repo-type=dataset --no-private
# or flip it in the repo's Settings page on huggingface.co
```

## Notes

- Non-interactive alternative (CI): `HF_TOKEN=hf_xxx .venv-eval/bin/hf upload ...`
  (the token appears in shell history, so prefer `hf auth login`).
- Audio is **not** in the dataset (metadata-only). Users fetch each clip's
  segment on demand. See the card's "Getting the audio" section and
  `scripts/fetch_clip.py`.

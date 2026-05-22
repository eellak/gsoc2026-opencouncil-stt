# Latest-per-utterance reduction

Companion report to [decisions/data.md → Keep only the latest edit per utterance](../../docs/decisions/data.md#2026-05-19---keep-only-the-latest-edit-per-utterance).

Source CSV: `data-1779206108158.csv` (~246 MB, 393 970 valid rows after CSV-level filtering, 397 556 lines including header).

## Why

The raw export records every reviewer edit. For training/evaluation we only need the final corrected text per utterance; the intermediate steps are noise (transient typos, accidental whitespace, partial replacements). Keeping them in the live DB would waste reviewer time and ~60 % more storage on data the reviewer themselves discarded one click later.

The selection rule:

```sql
PARTITION BY utterance_id
ORDER BY COALESCE(edit_updated_at, edit_timestamp) DESC, edit_id DESC
LIMIT 1
```

The CSV stays as source of truth — if a chain-history table is needed for audit, we re-ingest it into a separate table.

## Numbers

- 393 970 edits across 287 605 unique utterances
- **106 365 edits dropped** (27 % of rows)
- **287 605 latest edits kept** (100 % of utterances)

### Distribution of edits per utterance

| Edits | Utterances | % of utterances |
| ----- | ---------- | --------------- |
| 1     | 201 748    | 70.15 %         |
| 2     | 70 270     | 24.43 %         |
| 3     | 12 292     | 4.27 %          |
| 4     | 2 369      | 0.82 %          |
| 5     | 588        | 0.20 %          |
| 6     | 201        | 0.07 %          |
| 7     | 70         | 0.02 %          |
| 8     | 31         | 0.01 %          |
| 9–12  | 26         | <0.01 %         |
| 14–17 | 8          | <0.01 %         |
| 23    | 1          | <0.01 %         |
| 27    | 1          | <0.01 %         |

70 % of utterances have a single edit and lose nothing in the reduction. The reduction effect comes almost entirely from the long tail of multi-edit chains.

### DB impact

After deleting the 106 365 superseded rows and running `VACUUM (FULL) corrections`:

| Stage             | DB size | corrections table |
| ----------------- | ------- | ----------------- |
| Before delete     | 568 MB  | 403 MB            |
| After delete only | 568 MB  | 403 MB            |
| After VACUUM      | 215 MB  | 106 MB            |
| After VACUUM FULL | 214 MB  | 106 MB            |

`VACUUM` reclaims dead tuples; `VACUUM FULL` rewrites the table physically. Both fit the freed disk window.

## Examples

Three real chains from `Δημοτικό Συμβούλιο 18/05/26`, `edited_by: task` (automated reviewer doing iterative cleanup).

### Example 1 — `utterance_id: cmpbpcjns06d4bn0fg71jk94u`

Punctuation → speaker tag inserted → final orthography fix (`παρόν` → `παρών`).

| # | Updated at              |       | Text                                                            |
| - | ----------------------- | ----- | --------------------------------------------------------------- |
| 1 | 2026-05-19 06:53:57.169 | before | ένα λεπτάκι Τουλάχιστον όταν είμαι παρόν,                       |
|   |                         | after  | ένα λεπτάκι. Τουλάχιστον όταν είμαι παρόν,                      |
| 2 | 2026-05-19 15:52:22.395 | before | ένα λεπτάκι. Τουλάχιστον όταν είμαι παρόν,                      |
|   |                         | after  | Κύριε Δήμαρχε, ένα λεπτάκι. Τουλάχιστον όταν είμαι παρόν,       |
| 3 | 2026-05-19 15:52:26.203 | before | Κύριε Δήμαρχε, ένα λεπτάκι. Τουλάχιστον όταν είμαι παρόν,       |
|   | **[LATEST]**            | after  | Κύριε Δήμαρχε, ένα λεπτάκι. Τουλάχιστον όταν είμαι **παρών**,   |

What survives in the DB: only edit #3. The chain steps #1 and #2 are intermediate states already corrected by step #3.

### Example 2 — `utterance_id: cmpbpcjnq06awbn0fraqpz3mn`

Reviewer fixes one homophone, second-guesses, then settles on a different word entirely.

| # | Updated at              |       | Text                                  |
| - | ----------------------- | ----- | ------------------------------------- |
| 1 | 2026-05-19 06:53:56.985 | before | το θεάτους κανονικά πρέπει να πω,     |
|   |                         | after  | θέατρο κανονικά πρέπει να πω,         |
| 2 | 2026-05-19 15:46:44.474 | before | θέατρο κανονικά πρέπει να πω,         |
|   |                         | after  | […] κανονικά πρέπει να πω,            |
| 3 | 2026-05-19 15:46:56.76  | before | […] κανονικά πρέπει να πω,            |
|   | **[LATEST]**            | after  | **το θράσος** κανονικά πρέπει να πω,  |

Edit #1 would have actively misled training (wrong word). Keeping only edit #3 avoids that.

### Example 3 — `utterance_id: cmpbpcm4r090vbn0fp9ql810g`

Spelling cluster + prefix insertion + punctuation fix + final word correction (`σφικτικά` → `σφιχτά` → `ασφυκτικά`).

| # | Updated at              |       | Text (truncated to 120 chars)                                                                                                          |
| - | ----------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | 2026-05-19 06:54:02.988 | before | να κάνουμε μια εξατλητική διαβούλευση χωρίς αυτά τα σφικτικά χρονοδιαγράμματα …                                                        |
|   |                         | after  | να κάνουμε μια εξαντλητική διαβούλευση χωρίς αυτά τα **σφιχτά** χρονοδιαγράμματα …                                                     |
| 2 | 2026-05-19 15:00:22.628 | before | να κάνουμε μια εξαντλητική διαβούλευση χωρίς αυτά τα σφιχτά χρονοδιαγράμματα …                                                         |
|   |                         | after  | Νούμερο τρία,να κάνουμε μια εξαντλητική διαβούλευση χωρίς αυτά τα σφιχτά χρονοδιαγράμματα …                                            |
| 3 | 2026-05-19 15:00:25.636 | before | Νούμερο τρία,να κάνουμε μια εξαντλητική διαβούλευση …                                                                                  |
|   |                         | after  | Νούμερο τρία, να κάνουμε μια εξαντλητική διαβούλευση …                                                                                 |
| 4 | 2026-05-19 15:00:32.347 | before | Νούμερο τρία, να κάνουμε μια εξαντλητική διαβούλευση χωρίς αυτά τα σφιχτά χρονοδιαγράμματα …                                           |
|   | **[LATEST]**            | after  | Νούμερο τρία, να κάνουμε μια εξαντλητική διαβούλευση χωρίς αυτά τα **ασφυκτικά** χρονοδιαγράμματα …                                    |

The reviewer corrects `σφικτικά` to `σφιχτά` first, then later decides `ασφυκτικά` is the right word. Only the final state ends up in the DB; the earlier orthography is correctly discarded as noise.

## Reproducing

Examples were extracted from the CSV with `/tmp/find-chains.ts` (one-shot script, not committed) — it groups rows by `utterance_id`, sorts by `edit_updated_at`, and filters for chains of length 3–6 where the first `before_text` differs from the last `after_text`. Distribution counts come from the same pipeline, reading `data-1779206108158.csv` directly.

DB-side reduction was performed by:

1. `scripts/ingest-csv-v2.ts` — upserts all 393 970 rows + populates the `meetings` table.
2. `/tmp/latest-flag2.ts` (one-shot) — sets `latest_per_utterance` via `ROW_NUMBER()` window function partitioned by `utterance_id`.
3. Batched `DELETE FROM corrections WHERE latest_per_utterance = false` (5 000 rows / batch, 22 batches).
4. `VACUUM (FULL, ANALYZE) corrections`.

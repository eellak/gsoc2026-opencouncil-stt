# Ενσωμάτωση του fusion στο pipeline του OpenCouncil

Προδιαγραφή και οδηγός εκτέλεσης. Γράφτηκε 2026-08-26 μετά από grill τριών
γύρων (18 αποφάσεις, §1). Απευθύνεται σε agent που θα την εκτελέσει σε άλλη
συνεδρία: κάθε στάδιο έχει είσοδο, έξοδο, εντολές και πύλη.

Μία κριτική Codex (plan review, job `bac0a868`, effort high) διόρθωσε το
πρώτο κείμενο σε **εννέα** σημεία πριν πάρει αυτή τη μορφή· σημειώνονται
όπου άλλαξαν απόφαση: κοινά component results ανάμεσα στα arms (§5.3),
instance `scribe` για τη σύγκριση (§5.3), guard-on/guard-off ως δύο fixtures
(§4.4), ένα Scribe call στο shadow (§6), κανονικό αντικείμενο ήχου (§4.5),
πίνακας αποτυχιών (§4.5), προϋπολογισμός DP για 15 λεπτά (§7), κλειδί
συνένωσης ίχνους↔report (§4.2), και η απόσυρση της εξαίρεσης «rules-only
χωρίς listening test» (§6). Τα tests του §4.6 είναι δικά της πρότασης.

Ledger record: `exp-2026-08-26-fusion-integration` (OPEN, `PRODUCT`).
Επεκτείνει τα [`2026-08-21-fusion-production.md`](2026-08-21-fusion-production.md)
(σχήμα `oc-fusion/1`, κανόνες κατοχής) και
[`2026-08-21-postasr-architecture.md`](2026-08-21-postasr-architecture.md)
(«fusion ως provider»). Όπου εκείνα λένε N=2 χωρίς LLM, εδώ ισχύει **N=3** με
το μετρημένο arm της [`2026-08-26-category-chooser.md`](../reports/2026-08-26-category-chooser.md).

## 0. Τι στέλνουμε και γιατί

Το καλύτερο μετρημένο: τρισυστημική σύνθεση **Scribe v2 + Soniox `stt-async-v5`
+ `artifact-ct2-cleanpack-cont-s47`** πάνω σε τριπλή ευθυγράμμιση (`msa.align3`).

| arm | WER 391 | del | held-out 169 | καθεστώς |
|---|---:|---:|---:|---|
| Scribe v2 μόνο (ό,τι τρέχει σήμερα) | 0.13771 | 0.00956 | 0.15210 | incumbent |
| ψήφος W | 0.12020 | 0.01203 | 0.13798 | |
| **R1+R2** (guard off — έτσι μετρήθηκε) | **0.11154** | 0.01685 | 0.12825 | **στέλνεται πρώτο**, με guard on (§4.4) |
| policy (R1+R2 + LLM σε 2 κατηγορίες) | 0.10581 | 0.02517 | 0.12431 | flag, μόνο bench μέχρι να κλείσει `exp-2026-08-26-chooser-deletion-safety` |

Το δισυστημικό **δεν** στέλνεται (`exp-2026-08-21-fusion-production` CLOSED:
κερδίζει WER, αποτυγχάνει στο insertion guard). Όλα τα νούμερα είναι
agreement-with-OpenCouncil· κανένα δεν αποφασίζει fidelity-to-audio. Το
0.11154 είναι **guard off**· ο guard ψήφου/άρνησης κοστολογήθηκε ≈ +0.0002 WER
και δεν έχει δικό του end-to-end νούμερο — το παίρνει στο §4.4.

**Η ιδέα «Scribe βάση, παρέμβαση μόνο όπου το Scribe δεν είναι σίγουρο»
κρατιέται ως αρχή και αλλάζει σκανδάλη.** Σκανδάλη είναι η **διαφωνία** των
τριών πλήρων transcripts (το μετρημένο· R2 = «πάρε Scribe στην τριπλή
διαφωνία» είναι ήδη «Scribe βάση + μπάλωμα»), όχι η βεβαιότητα: η βεβαιότητα
Soniox δεν κέρδισε την ψήφο (`exp-2026-08-16-w-rt-confidence`), η δική μας
βρίσκει ύποπτη λέξη αλλά όχι σωστό υποψήφιο (`exp-2026-08-16-adapter-confidence`),
και η βεβαιότητα Scribe δεν έχει μετρηθεί ποτέ. Το `exp(logprob)` του Scribe
καταγράφεται ανά λέξη στο ίχνος και μετριέται ως arm (§5.4). Και τα τρία
συστήματα τρέχουν **πάντα σε όλο τον ήχο**: κόστος ≈ $0.32/ώρα ήχου + GPU.

## 1. Οι αποφάσεις (grill 2026-08-26)

| # | απόφαση |
|---|---|
| 1 | Πρώτο arm: **R1+R2 + guard** (β). LLM chooser πίσω από flag, default off. |
| 2 | Ο κώδικας ζει **στο `opencouncil-tasks`**, όχι εδώ. |
| 3 | Ο provider παίρνει **ήχο** και ενορχηστρώνει μόνος του τις τρεις κλήσεις (απαίτηση του bench). |
| 4 | Rollout: bench → shadow (Scribe παραμένει έξοδος, το fusion τρέχει δίπλα) → flip. |
| 5 | Τρίτη ψήφος: `artifact-ct2-cleanpack-cont-s47` μέσω `cap-serverless-asr`. |
| 6 | Αυτό το αρχείο + ledger record + ticket στο Wayfinder issue #50. |
| 7 | Χρόνοι λέξεων: **πάντα Scribe** (γ)· φραγμένη interpolation όπου το Scribe δεν έχει τη λέξη (§4.3). |
| 8 | `confidence` = συμφωνία στηλών, `providerConfidence` ξεχωριστό πεδίο, `alternatives[]` όπου η συμφωνία < 1 (γ). |
| 9 | Σκανδάλη παρέμβασης: νησιά διαφωνίας (α)· Scribe logprob ως μετρούμενο arm. |
| 10 | Python πακέτο μέσα στο tasks τώρα (stdlib-only), TS port ως ξεχωριστό ticket με το ίδιο conformance test ως πύλη (γ). |
| 11 | Επιφάνεια: νέο route `POST /v1/audio/transcriptions` στο tasks (openai-compatible)· το ίδιο module καλείται από το `transcribe.ts` (α). |
| 12 | Ίχνος: `oc-fusion/1` ανά segment με `stage` ανά νησί (α). Ποτέ git. |
| 13 | LLM chooser: **on** στο bench, **off** σε production. |
| 14 | Artifact για τη Νένα: HTML explorable σελίδα, ελληνικά (§8). |
| 15 | Host bench σταδίου: RunPod CPU pod με το image του tasks (α)· fallback DO droplet· το PR preview `pr-N.tasks.opencouncil.gr` είναι ο host του shadow. |
| 16 | Κλειδιά ElevenLabs/Anthropic: αγορά δικών μας τώρα **και** αίτημα scoped κλειδιών από OpenCouncil (γ). |
| 17 | Provider instances στο bench: `fusion-rules`, `fusion-policy-opus`, `fusion-policy-sonnet` (α) — **συν `scribe`** από την ίδια cache (διόρθωση Codex, §5.3). |
| 18 | Πύλη bench: WER με CI· το deletion rate **αναφέρεται**, γίνεται blocking πριν το flip (α). |

## 2. Αρχιτεκτονική

```
 ήχος (segment ≤15 min ή bench window ~143 s)  →  audio object {bytes, sha256, url?}
   │
   │   component cache: key = (audio sha256, provider id, provider params sha)
   ├─▶ Scribe v2 (υπάρχων ScribeTranscribe.ts)      words{text,start,end,logprob,type}
   ├─▶ Soniox stt-async-v5 (νέο SonioxTranscribe.ts) tokens{text,start_ms,end_ms,confidence}
   └─▶ serverless s47 (νέο OcAsrTranscribe.ts)       words{text,start,end,prob}
                       │  (τα τρία παράλληλα, κοινό deadline)
                       ▼
   fusion/ (Python, stdlib)  stdin JSON ─▶ stdout JSON
     wtoks() κανονικοποίηση, raw↔norm map ανά σύστημα
     align3(scribe, soniox, ours)  → cols
     islands(cols) → classify() → 21 κατηγορίες
     ανά νησί: rule (ψήφος | R1+R2 | Scribe | σβήσε | guard) ή LLM (flag, urllib → Messages API)
     έξοδος: fused tokens {raw, norm, src, col, island, stage, agreement}
                       │
                       ▼
   TS: χρόνοι από Scribe (§4.3) → Utterance[] (υπάρχον scribeWordsToUtterances)
       + ίχνος oc-fusion/1 → Spaces ($SC στο bench)
                       │
        ┌──────────────┴──────────────┐
   route /v1/audio/transcriptions   transcribe.ts (FUSION_MODE)
   → {"text": …}  (bench)           → Transcript (production)
```

Το `applyDiarization` και το `/fixTranscript` **δεν αλλάζουν**: παίρνουν
`Transcript` με `words[].{word,start,end,confidence}` όπως σήμερα.

## 3. Βήμα 0 — διάσωση του chooser (έγινε 2026-08-26)

Ο κώδικας του chooser ζούσε μόνο σε ephemeral scratchpad κάτω από `/tmp`.
Αντιγράφηκε:

- κώδικας + `freeze.json` (`6537f8e2451e7dbd`) + `confirm_log.json` +
  `wer_result*.json` + `autoprompt_state.json` →
  `eval/controlled_eval/chooser/` (31 αρχεία, χωρίς κείμενο μεταγραφών —
  ελέγχθηκε με grep για `referenceText|hypothesisText`).
- δεδομένα με κείμενο (`cats.json`, `page*.json`, `disagreements.jsonl`,
  `deletions.json`, `r1.json`, listening samples) →
  `~/.cache/oc-public/chooser-2026-08-25/`, μαζί με πλήρες tarball του
  scratchpad (`scratchpad-full-2026-08-26.tgz`, 411 MB).

Τα scripts είναι env-var driven (`REPORT=… OUT=… python3 cats.py`,
`FREEZE=… REPORT=… CITIES=ALL python3 reconstruct.py`) και θέλουν
`sys.path` προς `eval/controlled_eval/` (`atlas_substrate`, `msa`, `scoring`,
`error_atlas`, `column_classes`). Πρώτη ενέργεια του επόμενου agent: τρέξε
`reconstruct.py` με `CITIES=ALL` από τη νέα θέση και επιβεβαίωσε ότι βγάζει
τα νούμερα του `wer_result_ALL.json` (policy 0.10581, R1+R2 0.11154, W
0.12020). Αν όχι, σταμάτα: το όργανο δεν έχει διασωθεί.

## 4. Στάδιο 1 — το πακέτο `fusion/` μέσα στο opencouncil-tasks

Fork: `angelospk/opencouncil-tasks` (τοπικό clone
`~/projects/opencouncil/opencouncil-tasks`, στάσιμο από 2026-06-24 σε feature
branch — `git fetch upstream && git checkout -b feat/fusion-provider upstream/main`).

### 4.1 Διάταξη

```
fusion/
  __init__.py
  normalize.py      wtoks() ΑΥΤΟΥΣΙΟ από eval/controlled_eval/scoring.py:22
  msa.py            align3, compose, vote_column ΑΥΤΟΥΣΙΑ από eval/controlled_eval/msa.py
  islands.py        islands() από error_atlas.py:133, column_class από column_classes.py
  categories.py     cats.py: routes NUM/FILL/DUP/LOW + 13 signatures, ATOMIC/COMPOSITE
  rules.py          ΜΙΑ συνάρτηση για W / R1 / R2 / R1+R2 / Scribe / σβήσε / guard
  policy.json       το freeze.json, αυτούσιο· το sha256[:16] του ελέγχεται στο import
  chooser.py        render()/parse() από run_chooser.py· κλήση Messages API με urllib (stdlib)
  fuse.py           entrypoint: stdin JSON → stdout JSON
tests/fusion/
  test_conformance.py
  fixtures/MANIFEST.json   sha256 του report.json και κάθε fixture (§4.4) — ΜΟΝΟ hashes
```

Κανόνας: **καμία αλλαγή λογικής** κατά τη μεταφορά. Το R1+R2 υπάρχει σήμερα
τρεις φορές inline (`rules.py:job`, `cats.py:job`, `reconstruct.py:window_parts`)·
εδώ γίνεται μία συνάρτηση και το conformance test αποδεικνύει ότι είναι η ίδια.
Ο guard ψήφου/άρνησης (ποτέ μη σβήνεις νησί του οποίου η μόνη διαφορά είναι
«ναι»/«δεν») μπαίνει ως ρητός κανόνας με δικό του flag.

**Το LLM μένει μέσα στην Python** (διόρθωση Codex: «stdlib-only» και
«Anthropic SDK» δεν συμβιβάζονται). Το `chooser.py` καλεί το Messages API με
`urllib.request`. Ο **πλήρης φάκελος αιτήματος** παγώνει στο `policy.json`:
`model`, `system`, `thinking.budget_tokens: 4000`, `max_tokens`, `temperature`
(ό,τι ισοδυναμεί με το `claude -p --append-system-prompt … MAX_THINKING_TOKENS=4000`
του πειράματος), batch 60, και ο sha256 του φακέλου γράφεται στο ίχνος.

### 4.2 Συμβόλαιο `fuse.py`

Είσοδος (stdin):

```json
{
  "schema": "oc-fusion-in/1",
  "audio_sha256": "…",
  "systems": [
    {"id": "scribe", "params_sha": "…", "words": [{"raw": "Επιτροπή,", "start": 820.11, "end": 820.63, "conf": 0.91}]},
    {"id": "soniox", "params_sha": "…", "words": [{"raw": "επιτροπής", "start": 820.13, "end": 820.66, "conf": 0.44}]},
    {"id": "ours",   "params_sha": "…", "words": [{"raw": "επιτροπή",  "start": 820.10, "end": 820.62, "conf": 0.88}]}
  ],
  "config": {"arm": "rules", "guard": true, "llm": null}
}
```

`arm ∈ {W, rules, policy}`· `llm = {"model": "claude-opus-4-1"|"claude-sonnet-4-6"}`
μόνο όταν `arm = policy` (τα υπόλοιπα από το `policy.json`). Ένα raw word
μπορεί να δώσει 0..n normalized tokens (`wtoks` σπάει σε `\w+`)· το
`raw↔norm` map κρατιέται ώστε κάθε fused token να ξέρει από ποιο raw word
ποιου συστήματος προήλθε. Τα Scribe `type: spacing` tokens δεν είναι words.

Έξοδος (stdout):

```json
{
  "schema": "oc-fusion/1",
  "audio_sha256": "…",
  "config": {"arm": "rules", "guard": true, "policy_sha": "6537f8e2451e7dbd",
             "llm_envelope_sha": null, "code_rev": "…", "normalizer_sha": "…",
             "components": {"scribe": "<params_sha>", "soniox": "…", "ours": "…"}},
  "tokens": [
    {"i": 0, "text": "Επιτροπή,", "norm": "επιτροπη", "src": "scribe", "col": 3140,
     "island": null, "stage": "agree", "agreement": 1.0, "alternatives": null}
  ],
  "islands": [
    {"id": "isl_207", "cols": [3141, 3141], "category": "PAIR_soniox+ours|ATOMIC",
     "candidates": {"scribe": ["επιτροπης"], "soniox+ours": ["επιτροπη"], "r12": ["επιτροπης"], "empty": []},
     "stage": "rule", "rule": "ψήφος", "chosen": "soniox+ours",
     "scribe_conf_min": 0.44, "llm": null}
  ],
  "dropped": [{"island": "isl_300", "src": "soniox", "text": "να πούμε κι εμείς", "rule": "R1"}],
  "stats": {"n_tokens": 0, "n_islands": 0, "by_category": {}, "by_stage": {"agree": 0, "rule": 0, "llm": 0}}
}
```

`stage ∈ {agree, rule, llm}` είναι το «τι κράτησε από το Scribe, πού κάλεσε
κανόνα, πού κάλεσε LLM» για την οπτική επιθεώρηση. Τα tokens που **δεν**
είναι `agree` κουβαλούν `alternatives[]` = οι υποψήφιοι του νησιού ανά
σύστημα με το `conf` τους (απόφαση 8).

**Κλειδί συνένωσης (διόρθωση Codex):** το openai-compatible αίτημα δεν
φέρει αναγνωριστικό παραθύρου. Το ίχνος αποθηκεύεται ως
`<audio_sha256>.<arm>.json` και η απάντηση φέρει header `x-oc-request-id` =
`audio_sha256:arm`. Το report του bench ενώνεται με τα ίχνη μέσω του sha256
του ήχου κάθε παραθύρου (κατεβαίνει από το bench, hash-άρεται τοπικά).

### 4.3 Χρόνοι και βεβαιότητα (TS πλευρά, `src/lib/fusion.ts`)

- Token με `src = scribe` → `start/end` του raw word του Scribe,
  `timing_source: "scribe-native"`.
- Token σε στήλη όπου το Scribe **έχει** token (διαφωνία ταυτότητας) → οι
  χρόνοι του Scribe token της ίδιας στήλης, `timing_source: "scribe-column"`.
- Token σε στήλη όπου το Scribe **σιωπά** (μπάλωμα κατοχής) → γραμμική
  interpolation ανάμεσα στο `end` του προηγούμενου και το `start` του
  επόμενου Scribe token, αναλογικά με το μήκος χαρακτήρων,
  `timing_source: "interpolated"`. Αν το κενό υπερβαίνει **2.0 s**, είναι
  μηδενικό, ή τα tokens βρίσκονται πριν από το πρώτο / μετά το τελευταίο
  Scribe token, κολλάνε στην άκρη του πλησιέστερου γείτονα, το καθένα
  0.05 s διαδοχικά, `timing_source: "unaligned"`.
- Ένα raw word που έδωσε πολλά norm tokens μοιράζει το διάστημά του
  αναλογικά· ένα raw word που έδωσε μηδέν tokens (μόνο στίξη) κολλάει στο
  προηγούμενο.
- **Invariant, ελεγμένο σε κάθε έξοδο:** `start ≤ end`, μονότονοι μη
  επικαλυπτόμενοι χρόνοι, όλα μέσα στα όρια του segment. Παραβίαση = σφάλμα
  του segment, όχι σιωπηλή διόρθωση.
- Οι χρόνοι Soniox/δικοί μας **δεν** χρησιμοποιούνται (απόφαση 7). Μένουν
  στο ίχνος.
- `confidence` = `agreement` (1.0 / 0.67 / 0.33 για 3/3, 2/3, νησί χωρίς
  πλειοψηφία)· `providerConfidence` = το `conf` του παραγωγού· τα
  `Utterance.confidence/minWordConfidence/totalConfidence` υπολογίζονται από το
  `confidence` όπως σήμερα (`scribeWordsToUtterances`). Έλεγχος πριν το
  κλείδωμα: ότι το `Word` type δέχεται πρόσθετο προαιρετικό πεδίο χωρίς να
  σπάει το callback `version: 4` στο κύριο app (αν σπάει → `version: 5`).

### 4.4 Conformance test (η πύλη του σταδίου)

Το bench report είναι **δημόσιο** (`GET https://bench.opencouncil.gr/api/runs/2026-08-22-post-june-held-out-test-clean-pack-cont-/report.json`)
και περιέχει `hypothesisText` και `referenceText` ανά παράθυρο για
`scribe`, `soniox`, `oc-cleanpack-cont-s47-b`. Άρα κάθε fixture
**αναπαράγεται από δημόσια δεδομένα + κώδικα**: στο git μπαίνει μόνο το
`MANIFEST.json` με sha256 του report και κάθε fixture· το CI κατεβάζει το
report, ελέγχει το hash, ξαναπαράγει και συγκρίνει (διόρθωση Codex: fixture
μόνο στο `$SC` είναι machine-local).

Τα fixtures παράγονται **μία φορά** από το διασωθέν `reconstruct.py` (§3),
τέσσερα αρχεία στο `$SC/chooser-2026-08-25/`:

| fixture | περιεχόμενο | αναμενόμενο |
|---|---|---|
| `fixture_W_391.json` | ακολουθία norm tokens ανά παράθυρο + S/I/D/N | WER 0.12020 |
| `fixture_rules_off_391.json` | R1+R2, guard off | WER 0.11154 |
| `fixture_rules_on_391.json` | R1+R2, guard on — **νέο νούμερο**, μετριέται εδώ | ≈ 0.1117 |
| `fixture_policy_islands_391.json` | ανά νησί: κατηγορία, candidates, route, rule/llm, rendered prompt sha, επιλογή (από το `page_islands.json`/`llm_out.json` της confirmation) | 13,414 νησιά |

Το test απαιτεί **byte-ισότητα** της ακολουθίας tokens και **ακριβή**
S/I/D/N ανά παράθυρο — όχι στρογγυλεμένο WER — για W, rules-off, rules-on,
και **ανά νησί** ισότητα για το policy (διόρθωση Codex: τα συνολικά πλήθη ανά
κατηγορία μπορούν να συμφωνούν ενώ κάθε νησί δρομολογείται λάθος). Η LLM
κλήση τεστάρεται με mock που επιστρέφει τις παγωμένες απαντήσεις, και μία
φορά ζωντανά (§5.4) με replay των ίδιων frozen transcripts.

Χωρίς πράσινο εδώ, τίποτα από το §5 δεν ξεκινά — αλλιώς είναι «δεύτερο stack».

### 4.5 Adapters, route, αποτυχίες

**Κανονικό αντικείμενο ήχου (διόρθωση Codex).** Κάθε είσοδος γίνεται
`{bytes, sha256, mime, durationSec}`. Το Scribe δέχεται bytes· Soniox και
serverless θέλουν URL → τα bytes ανεβαίνουν **μία φορά** σε προσωρινό
Spaces key `fusion-tmp/<sha256>.<ext>` (υπάρχων `uploadToSpaces`), με
διαγραφή στο τέλος του αιτήματος και όριο μεγέθους 200 MB. Το `transcribe.ts`
ήδη έχει URL, οπότε κατεβάζει bytes μόνο για το hash.

- `src/lib/SonioxTranscribe.ts`: async REST (`POST /v1/files` → `POST /v1/transcriptions`
  → poll → `GET /v1/transcriptions/{id}/transcript`), `model: "stt-async-v5"`,
  `language_hints: ["el"]`, επιστρέφει `tokens[{text,start_ms,end_ms,confidence}]`.
  Αναφορά: `~/projects/soniox-tools/async_transcribe.py`. Env `SONIOX_API_KEY`,
  `SONIOX_MAX_CONCURRENT`.
- `src/lib/OcAsrTranscribe.ts`: `POST https://api.runpod.ai/v2/<endpoint>/runsync`,
  `{"input":{"op":"transcribe_url","url":…}}`, word timestamps on. Runbook
  `docs/runbooks/serverless-asr-endpoint.md`. Env `RUNPOD_API_KEY`,
  `OC_ASR_ENDPOINT_ID`. **Provenance (διόρθωση Codex):** πριν από κάθε run
  `op: provenance` και το ίχνος καταγράφει container digest, ct2 `model.bin`
  sha256 (`3aa6368fb1b59a13…`), decode options, VAD, language, beam — όλα
  συγκρίνονται με το `manifest.providers` του μετρημένου bench run, όχι μόνο
  το prefix του model.bin.
- **Component cache:** `key = (audio sha256, provider id, sha256(params))` →
  raw απάντηση, στο `FUSION_CACHE_DIR` (pod) ή Spaces `fusion-cache/`. Κάθε
  arm ξαναχρησιμοποιεί το ίδιο component result· το ίχνος γράφει τα
  `components.*` hashes. Έτσι τα arms του bench είναι **paired** και οι
  recognisers τρέχουν μία φορά ανά παράθυρο.
- `src/lib/fusion.ts`: `fuseSegment({audio, arm, scribeResult?})` → τα τρία
  `Promise.allSettled` με **κοινό deadline** (`FUSION_DEADLINE_MS`, default
  240 000) και ακύρωση· subprocess `python3 fusion/fuse.py`· §4.3· ίχνος.
  Αν δοθεί `scribeResult`, το Scribe **δεν** ξανακαλείται (shadow, §6).
- Route `POST /v1/audio/transcriptions` (`src/routes/openaiCompat.ts`):
  multipart `file`, `model`, `language`· Bearer κατά `API_TOKENS`· σύγχρονο·
  επιστρέφει `{"text": …}` + `x-oc-request-id`. Το `model` επιλέγει arm:
  `scribe` (η cached Scribe υπόθεση, για τη σύγκριση του §5) |
  `fusion-rules` | `fusion-policy-opus` | `fusion-policy-sonnet`.
  `language ≠ el` → 400.
- `transcribe.ts`: το ένα call site `scribeTranscriber.transcribe(...)` γίνεται
  `FUSION_MODE === "off" ? scribe : fuseSegment`· σε `shadow` καλείται το
  Scribe **μία φορά**, το αποτέλεσμά του επιστρέφεται ως έξοδος **και**
  δίνεται στο `fuseSegment` ως `scribeResult`· και τα δύο ίχνη αποθηκεύονται.
- Env: `FUSION_MODE=off|shadow|on` (default off), `FUSION_LLM=off|on`,
  `FUSION_LLM_MODEL`, `FUSION_TRACE_DIR` / Spaces prefix, `FUSION_CACHE_DIR`,
  `FUSION_DEADLINE_MS`. `on`/`shadow` μόνο για `cityLanguage = "el"`.
- Dockerfile: `apt-get install -y python3` στο runner stage (stdlib αρκεί).

**Πίνακας αποτυχιών (διόρθωση Codex — το «fallback Scribe» δεν ορίζεται όταν
αποτυγχάνει το Scribe):**

| συμβάν | bench route | production `on` | production `shadow` |
|---|---|---|---|
| Scribe αποτυγχάνει | 502, χωρίς fallback | σφάλμα segment → υπάρχον retry του `transcribe.ts` | ίδιο με σήμερα |
| Soniox ή serverless αποτυγχάνει / deadline | 502 (το bench ξαναστέλνει failed items) | έξοδος Scribe, `metadata.provider = "scribe-fallback"`, λόγος στο ίχνος | Scribe έξοδος, ίχνος fusion `status: failed` |
| `fuse.py` non-zero / άκυρο JSON / παραβίαση invariant | 500 | όπως πάνω, fallback Scribe | όπως πάνω |
| LLM σφάλμα ή parse failure σε νησί | το νησί πέφτει στον κανόνα της κατηγορίας (`r12`), `llm.error` στο ίχνος | (LLM off) | (LLM off) |

### 4.6 Tests πριν από τον κώδικα (TDD, από την κριτική Codex)

Vitest για τα TS, pytest για το `fusion/`. Κόκκινα πριν γραφτεί οτιδήποτε:

1. `rules_guard_on_matches_frozen_391`, `rules_guard_off_matches_frozen_391`,
   `W_matches_frozen_391` — ακολουθία tokens + S/I/D/N ανά παράθυρο.
2. `policy_matches_every_frozen_island` — ανά νησί, mock LLM.
3. `production_words_equal_benchmark_normalization` — ≥20 πραγματικές
   απαντήσεις Scribe (spacing, στίξη, τόνοι, one-to-many): `wtoks(" ".join(raw))
   == wtoks(hypothesisText)`.
4. `fusion_renders_readable_greek` — golden κείμενα για στίξη, απόστροφο,
   αριθμούς, επαναλήψεις, αντικατάσταση από άλλο σύστημα.
5. `timings_are_monotonic_and_segment_bounded` — όλες οι περιπτώσεις του §4.3.
6. `fused_transcript_preserves_diarization_contract` — πραγματικό
   `applyDiarization` πάνω σε fused `Transcript`.
7. `three_arms_share_identical_component_results` — ίδιος ήχος σε 4 arms →
   μία κλήση ανά recogniser, ίδια `components.*` hashes στα ίχνη.
8. `shadow_reuses_the_returned_scribe_result` — ένα Scribe αίτημα.
9. `provider_failures_have_defined_results` — ο πίνακας του §4.5.
10. `multipart_request_completes_end_to_end` — πραγματικό Express + πραγματικό
    Python, mock μόνο στα όρια των providers· auth, `model`, `language`,
    cleanup, σχήμα σφαλμάτων.
11. `production_length_meets_resource_budget` — 15-λεπτο transcript (~2,500
    tokens ανά σύστημα) κάτω από 60 s wall / 1 GB· αν αποτύχει, το §7.1
    chunking γίνεται υποχρεωτικό **πριν** τους adapters.
12. `trace_is_joinable_and_complete` — sha256, arm, components, code_rev,
    policy_sha, envelope sha, χωρίς μυστικά.

## 5. Στάδιο 2 — το benchmark

### 5.1 Προαπαιτούμενα

- Κλειδιά: `ELEVENLABS_API_KEY` (**δεν υπάρχει** εδώ — αγορά + αίτημα, απόφαση 16),
  `SONIOX_API_KEY` (υπάρχει στο `~/projects/soniox-tools/.env`),
  `ANTHROPIC_API_KEY` (**δεν υπάρχει** — μόνο CLI auth), `RUNPOD_API_KEY`
  (**η θέση του δεν καταγράφηκε**· το serverless επαληθεύτηκε 2026-08-24 άρα
  υπάρχει — ρώτα τον Harold), `BENCH_API_KEY` (`.env` εδώ).
- Κόστος μίας πλήρους run (14.57 h ήχου, 391 παράθυρα), **με** την cache
  του §4.5: Scribe ≈ $3.2, Soniox ≈ $1.5, serverless ≈ $3 — μία φορά· LLM ≈
  $2 ανά policy instance· pod ≈ $0.5. Σύνολο ≈ **$12**. Χωρίς cache θα ήταν
  ≈ $28 και τα arms δεν θα ήταν paired.

### 5.2 Host

RunPod **CPU** pod με το Docker image του tasks (build από το branch, push σε
`ghcr.io/angelospk/`), port 3000 μέσω `proxy.runpod.net`. Πριν από
οτιδήποτε: watchdog με hard deadline (κανόνας του CLAUDE.md), pod ID στο
ledger record. Έλεγχος με `GET /tasks`. Serverless endpoint με
`min workers = 1` για όσο τρέχει η run (cold start εκτός κρίσιμου μονοπατιού).

### 5.3 Πρωτόκολλο (runbook `docs/runbooks/benchmark-api.md`)

1. `PUT /api/config/providers`: **τέσσερα** `openai-compatible` instances στο
   ίδιο `baseURL: https://<pod>-3000.proxy.runpod.net/v1` με `model` =
   `scribe` | `fusion-rules` | `fusion-policy-opus` | `fusion-policy-sonnet`,
   `language: el`, `concurrency: 2`. `PUT /api/config/keys` ανά instance.
   Το instance `scribe` υπάρχει ώστε η σύγκριση του §5.4 να γίνεται πάνω στην
   **ίδια** Scribe υπόθεση που κατανάλωσε το fusion (διόρθωση Codex: το
   cached `scribe` του bench είναι άλλη κλήση, άλλο κείμενο).
2. **`POST /api/config/providers/test`** και poll στο `/test/status` (3-λεπτο
   clip, ζεστό). Μετά **ένα 15-λεπτο cold-start αίτημα** μέσω CLI
   `transcribe-single` στο pod, και **έξι ταυτόχρονα** 3-λεπτα — αυτά
   απαντούν στο ρίσκο #1 (§7). Αποτυχία με timeout → §7 fallback host, όχι retry.
3. Clone του `2026-08-22-post-june-held-out-test-clean-pack-cont-` με **μόνο**
   τα τέσσερα νέα `providerInstanceIds`. Start. Poll ανά 30 s. Re-POST
   `/start` μόνο για failed items. Σειρά: το bench τρέχει τα instances
   παράλληλα· η cache εγγυάται ότι όποιο arm φτάσει πρώτο σε ένα παράθυρο
   πληρώνει τους recognisers και τα υπόλοιπα τους ξαναχρησιμοποιούν.
4. Ανάγνωση `report.json`. Ίχνη στο `$SC/fusion-bench-<runId>/`, ενωμένα με
   το report μέσω audio sha256 (§4.2).

### 5.4 Σκοράρισμα και πύλη (απόφαση 18, διορθωμένη)

Offline, με `eval/controlled_eval/scoring.py` και paired meeting-clustered
bootstrap (seed παγωμένο, 117 clusters στα 391 / 47 στα 169):

- **Πρωτεύον:** `Δ = WER(fusion-rules) − WER(scribe)`, **άνω όριο του 95% CI
  < 0**, στα **169 confirmation παράθυρα** (οι πέντε πόλεις που δεν επέλεξαν
  prompts — διόρθωση Codex: το 391 είναι κατά 57% μολυσμένο). Τα 391
  αναφέρονται δίπλα ως ο αριθμός του leaderboard.
- **Αναπαραγωγή:** `fusion-rules` εντός ±0.005 του offline 0.11154 στα 391
  (η διαφορά είναι re-transcription θόρυβος Scribe/Soniox — αν είναι
  μεγαλύτερη, σύγκρινε τα `hypothesisText` του νέου run με τα cached ανά
  σύστημα πριν κατηγορήσεις το fusion).
- **Chooser μέσω API:** replay των **frozen** transcripts (fixtures §4.4)
  μέσα από το `chooser.py` ζωντανά, 60-island batches των δύο LLM
  κατηγοριών: island accuracy εντός **5 μονάδων** του `run_*_opus.json`
  in-sample. Απομονώνει το SDK/envelope από τον θόρυβο νέου ASR.
- **Κυριαρχία:** τα 5 μεγαλύτερα παράθυρα < 25% του καθαρού κέρδους.
- **Αναφέρονται, δεν μπλοκάρουν:** deletion/insertion/substitution rates ανά
  instance· `policy-opus` έναντι `policy-sonnet`· κάθε policy νούμερο είναι
  **ένα** run.
- **Arm Scribe-βεβαιότητας (απόφαση 9):** από τα ίχνη, AUROC του
  `scribe_conf_min` ανά νησί ως προς «το Scribe έκανε λάθος στο νησί» (κατά
  την αναφορά). Αν > 0.8, ανοίγει record για arm «κανόνας μόνο όταν
  `scribe_conf_min` < κατώφλι», out-of-fold στην πόλη-partition του
  `autoresearch.py`. Αν όχι, κλείνει με το νούμερο.

Αποτέλεσμα → `docs/reports/2026-XX-XX-fusion-bench.md`, ledger record
ενημερωμένο, **μετά** ξεκινά το §6.

## 6. Στάδιο 3 — shadow και flip σε production

- PR στο upstream `schemalabz/opencouncil-tasks` με `FUSION_MODE=off` default·
  το preview `pr-N.tasks.opencouncil.gr` τρέχει `shadow` σε 5+ πραγματικές
  συνεδριάσεις· **ένα** Scribe call ανά segment (§4.5)· ίχνη στο Spaces
  prefix `fusion-shadow/`.
- Πριν το `on`: (α) `exp-2026-08-26-chooser-deletion-safety` **κλειστό** με
  το listening test — για **οποιοδήποτε** arm. Η εξαίρεση «rules-only αν
  deletions ≤ 1.5x Scribe» αποσύρθηκε: το rules arm είναι ήδη 1.76x και ο
  λόγος από την αναφορά δεν λέει τίποτα για τον ήχο (διόρθωση Codex).
  (β) τρεις επαναλήψεις αν στέλνεται policy· (γ) κόστος ανά συνεδρίαση
  καταγεγραμμένο· (δ) τα shadow ίχνη χωρίς παραβίαση invariant χρόνων σε
  καμία συνεδρίαση.
- Το `/fixTranscript` παραμένει όπως είναι. Καμία αλλαγή στο κύριο app εκτός
  αν το `Word` type αλλάξει version (§4.3).

## 7. Ρίσκα, με σειρά

1. **Σύγχρονο route έναντι proxy timeout / μέγεθος εισόδου.** Κρίσιμο
   μονοπάτι ανά παράθυρο 143 s: max(Scribe ~20 s, Soniox async 30–60 s,
   serverless ~70 s). Στα 15-λεπτα production segments: serverless ≈ 7 min
   και το `align3` πάνω σε ~2,500 tokens ανά σύστημα — **αμέτρητο** (το
   πείραμα έτρεξε σε ~400). Το test 11 του §4.6 το μετρά πρώτο. Αν αποτύχει:
   **chunking** — κόψιμο των τριών ρευμάτων σε κομμάτια ≤ 4 λεπτών στις
   παύσεις του Scribe > 1.5 s, fusion ανά κομμάτι, συρραφή με τη σειρά, και
   το ίδιο κόψιμο μετριέται ξανά στο bench πριν χρησιμοποιηθεί. Το
   `proxy.runpod.net` έχει αντέξει τα GPU runs· ο mini-PC πίσω από Cloudflare
   (~100 s) όχι. Fallback host DO droplet.
2. **Tokenization production ≠ bench.** Test 3 του §4.6.
3. **Η αναφορά δεν είναι ο ήχος.** Κάθε πύλη εδώ είναι agreement-WER. Οι
   διαγραφές (rules 1.76x Scribe, policy 2.6x) κρίνονται μόνο ακουστικά.
4. **Scribe drift.** Το Scribe δεν είναι ντετερμινιστικό· η αναπαραγωγή στο
   §5.4 έχει ανοχή γι' αυτό και μόνο γι' αυτό· η cache του §4.5 εγγυάται ότι
   μέσα σε μία run όλα τα arms βλέπουν το ίδιο κείμενο.
5. **Ο chooser μέσω API ≠ μέσω `claude` CLI.** Παγωμένος φάκελος (§4.1) και
   replay πύλη (§5.4).

## 8. Το artifact για τη Νένα (απόφαση 14)

HTML explorable, ελληνικά, στο `docs/reports/img/2026-08-fusion-explainer.html`
(ή claude-page). Περιεχόμενο: το pipeline του §2 ως διάγραμμα· ένα
πραγματικό νησί με τρεις υποθέσεις και τι διάλεξε κάθε arm· ο πίνακας του §0·
«γιατί β και όχι α» (deletions 0.0169 έναντι 0.0252, ο άτρεχτος listening
test)· τα όρια (agreement ≠ fidelity, 57% των 391 επέλεξαν τα prompts, ένα
run ανά νούμερο). Χωρίς κείμενο μεταγραφών εκτός από το ένα παράδειγμα, με
έγκριση.

## 9. Λίστα εκτέλεσης για τον επόμενο agent

1. §3: `reconstruct.py` από τη νέα θέση αναπαράγει `wer_result_ALL.json`. (30 λεπτά)
2. Παραγωγή των τεσσάρων fixtures + `MANIFEST.json` (§4.4)· το guard-on
   νούμερο γράφεται στο ledger record. (1 ώρα)
3. Branch στο tasks· tests 1–2, 11 κόκκινα· `fusion/` πακέτο μέχρι πράσινο
   (§4.1–4.4). Αν το 11 κοκκινίζει, §7.1 chunking πρώτα. (1 μέρα)
4. Tests 3–10, 12 κόκκινα· audio object, cache, adapters, route,
   `transcribe.ts` flag, Dockerfile (§4.5). (1–2 μέρες)
5. Κλειδιά (§5.1), image push, RunPod CPU pod + watchdog, serverless min
   workers (§5.2). (2 ώρες)
6. Providers test clip → 15-λεπτο cold → 6 ταυτόχρονα → clone με 4 instances
   → run → report (§5.3). (4 ώρες ρολόι)
7. Σκοράρισμα στα 169 και 391, replay chooser, AUROC, report, ledger (§5.4). (3 ώρες)
8. Artifact Νένας (§8). (2 ώρες)
9. PR upstream, shadow (§6).

Ανοιχτά που θέλουν τον Harold: θέση `RUNPOD_API_KEY`, αγορά κλειδιών
ElevenLabs/Anthropic, αίτημα scoped κλειδιών στο OpenCouncil, το 36-clip
listening test (προϋπόθεση του flip για κάθε arm).

## 10. Τροποποίηση 2026-08-31 — plan review Codex `4ea2d051` και η μέτρηση του `align3`

Το σχέδιο των §2–§7 **δεν αποσύρεται· αναδιατάσσεται**. Δύο πράγματα το άλλαξαν: μία
μέτρηση και μία δεύτερη κριτική Codex (effort high, job `4ea2d051`). Η ετυμηγορία της
κριτικής: «δεν εγκρίνω flip με το σημερινό σχέδιο· εγκρίνω rules-only shadow αφού
προηγηθούν πραγματικά δεκαπεντάλεπτα, παγωμένο chunking, και listening test μεγαλύτερο
από 36 clips».

### 10.1 Το ρίσκο 1 έκλεισε ως αποτυχία

[`docs/reports/2026-08-31-align3-scaling.md`](../reports/2026-08-31-align3-scaling.md):
το `align3` περνάει στα 1.000 tokens (12,5 s / 1,13 GB) και **σκάει με MemoryError στα
2.500** κάτω από όριο 12 GB. Η μνήμη είναι `n³` bytes επειδή ο πίνακας `ops` δεσμεύει
`(nb+1)*(nc+1)` bytes ανά επίπεδο και το band δεν τη μειώνει. Ένα δεκαπεντάλεπτο
απαιτεί 15,6 GB.

Συνέπεια: **το chunking του §7 παύει να είναι σχέδιο διαφυγής και γίνεται ο υποψήφιος
για production**. Το test 11 του §4.6 δεν είναι πλέον «μέτρησε αν χρειάζεται
chunking»· είναι «μέτρησε το chunked σύστημα».

### 10.2 Νέο Στάδιο 0, πριν γραφτεί η ενορχήστρωση

1. Τα 36 clips ως **pilot** (όχι πύλη, §10.4). Αν φανεί καθαρό πρόβλημα διαγραφής,
   δεν αξίζει να χτιστεί η ενσωμάτωση.
2. Το `align3` scaling — **έγινε**, §10.1.
3. Cold-cache end-to-end σε 10–20 πραγματικά δεκαπεντάλεπτα, στην **ίδια** διαδρομή
   δικτύου/proxy/serverless με την production: cold start, upload, κάθε provider,
   polling, alignment, p50/p95/max, μνήμη, μέγεθος payload, συμπεριφορά σε timeout.
4. Η απόφαση full-stream ή chunked παίρνεται **εδώ**, όχι μετά το bench.

### 10.3 Δύο chunkings, όχι ένα

- **Chunking κειμένου**: οι τρεις providers βλέπουν όλο το δεκαπεντάλεπτο και κόβονται
  μόνο οι παραγόμενες ακολουθίες. Αλλάζει μόνο ο δικός μας αλγόριθμος. **Αυτό είναι ο
  υποψήφιος.**
- **Chunking ήχου**: ο ήχος κόβεται πριν τις κλήσεις ASR, άρα αλλάζουν και οι τρεις
  υποθέσεις. Είναι **άλλο σύστημα** και απαιτεί πλήρη επαναξιολόγηση.

Το chunking κειμένου θέλει versioned configuration, κοινά χρονικά όρια για τα τρία
ρεύματα, overlap γύρω από κάθε όριο, ντετερμινιστική συρραφή σε ομόφωνα anchors,
αναγκαστική κοπή όταν δεν υπάρχει παύση > 1,5 s, και δικά του fixtures για νησιά
διαφωνίας που τέμνουν όριο. Σειρά αξιολόγησης: full έναντι chunked στα ίδια παγωμένα
provider outputs → πάγωμα κατωφλίων → **μία** τυφλή εκτέλεση → η chunked έκδοση περνάει
αυτοτελώς το `ΔWER vs Scribe` → non-inferiority έναντι full-stream με προκαθορισμένη
ανεκτή απώλεια → ξεχωριστό listening gate για διαγραφές κοντά στα όρια.

### 10.4 Τα 36 clips δεν αποφασίζουν flip

Ακόμη και με **μηδέν** επιβλαβείς διαγραφές στα 36, το μονόπλευρο 95% άνω όριο για την
πραγματική πιθανότητα βλάβης είναι ≈ 8%. Για αποκλεισμό > 5% χρειάζονται ≥ 59
ανεξάρτητα events, για > 2% χρειάζονται 149, για > 1% χρειάζονται 299 — και τα events
της ίδιας συνεδρίασης δεν είναι ανεξάρτητα.

Ελάχιστο για flip του rules arm: **120 fusion-only deletion events**, από ≥ 12
συνεδριάσεις και ≥ 5 πόλεις, ≤ 10 events ανά συνεδρίαση, **δύο** ανεξάρτητοι
ελληνόφωνοι κριτές τυφλοί ως προς το σύστημα, adjudication στις διαφωνίες, και
προκαθορισμένο όριο στο **άνω** confidence bound. Χρειάζονται και control δείγματα από
διαγραφές του incumbent και από κρίσιμες αντικαταστάσεις (αριθμοί, ονόματα, ποσά,
ημερομηνίες, ψήφοι, αρνήσεις): η επικίνδυνη αλλαγή δεν είναι μόνο η διαγραφή. Το policy
arm χρειάζεται **δικό του** δείγμα και δεν κληρονομεί την έγκριση του rules.

### 10.5 Τι κόβεται από το scope

Στόχος ενός μήνα part-time: **rules-only, σε shadow, έτοιμο για canary** — όχι flip.

| κόβεται | γιατί |
|---|---|
| `fusion-policy-opus`, `fusion-policy-sonnet` | κερδίζουν 0.00394 WER στα held-out και σβήνουν 2,6x αντί 1,76x |
| η επιλογή Opus έναντι Sonnet | επιλογή νικητή στο ίδιο set ακυρώνει την κάλυψη του CI |
| το `±0.005` στα 391 ως release criterion | είναι provider-drift diagnostic, πλεονάζον μετά το exact conformance |
| «τα 5 μεγαλύτερα παράθυρα < 25% του κέρδους» | περνάει ενώ όλο το κέρδος έρχεται από μία πόλη· αντικαθίσταται από leave-one-cluster-out, ΔWER ανά πόλη, κατανομή ανά συνεδρίαση |
| κάθε fallback που κάνει fusion με **δύο** arms | δεν έχει αξιολογηθεί ποτέ ως σύστημα |

**Δεν** κόβεται το `stdlib-only` (απόφαση 10): το repo του πελάτη είναι TypeScript/Node
και ένα δέντρο εξαρτήσεων Python στο container τους είναι δικό τους κόστος, όχι δικό
μας gold-plating. Δεν κόβεται ούτε το openai-compatible route (απόφαση 11) — το
απαιτεί το bench — αλλά μένει **λεπτός adapter** πάνω στην ίδια implementation.

### 10.6 Πύλες που έλειπαν

1. **Το held-out έχει καταναλωθεί.** Τα 169 παράθυρα είναι αμερόληπτα μόνο για τις ήδη
   παγωμένες εκδόσεις. Δεν επιτρέπεται να συντονιστεί πάνω τους το chunking, ούτε να
   επιλεγεί arm. Ο chunked υποψήφιος θέλει **νέο τυφλό acceptance set** από νέες
   συνεδριάσεις και πόλεις.
2. **Failure contract, γραμμένο πριν την υλοποίηση.** Τι γίνεται όταν λείπει ένα arm·
   όταν λήξει ο συνολικός χρόνος ενώ το Scribe έχει τελειώσει· όταν αποτύχει το Scribe
   από το οποίο εξαρτώνται και οι χρόνοι· πότε γίνεται ακριβές fallback· ποιο είναι το
   latency budget πριν το upstream deadline· πώς αποφεύγονται διπλές χρεώσεις σε retry·
   τι γίνεται με μερικές ή δηλητηριασμένες εγγραφές cache.
3. **Οι χρόνοι είναι ξεχωριστό προϊόν.** Η interpolation μπορεί να είναι μονότονη και
   λάθος. Πύλη: ποσοστό λέξεων με interpolated χρόνο, μέγιστο interpolated span,
   ακουστικός έλεγχος σε αντιπροσωπευτικά insertions, και ρητό `estimated` flag προς τα
   κατάντη.
4. **Το cache key θέλει περισσότερα** από provider id + params sha: ακριβές
   model/version ή alias, preprocessing/encoding ήχου, έκδοση σχήματος αιτήματος,
   έκδοση normalization/parser, timestamp και checksum της ωμής απάντησης. Το ζεστό run
   μετράει τον αλγόριθμο· το κρύο μετράει το σύστημα. Χρειάζονται και τα δύο.
5. **Το PR preview δεν είναι production shadow.** Το shadow θέλει τον πραγματικό worker,
   τα πραγματικά όρια timeout/μνήμης/concurrency, cold starts και provider quotas, σε
   ~100 δεκαπεντάλεπτα jobs από 10+ συνεδριάσεις και πολλές πόλεις. Οι 5 συνεδριάσεις
   βρίσκουν integration bugs, δεν στηρίζουν SLO.

### 10.7 Canary αντί για flip

Το §6 τελείωνε σε `flip`. Τελειώνει τώρα σε **canary**: μικρό ποσοστό συνεδριάσεων,
αυτόματο ακριβές fallback στο Scribe σε timeout / αποτυχία provider / άκυρη
ευθυγράμμιση / ανωμαλία εξόδου, και kill switch που δεν απαιτεί deployment.

### 10.8 Τι δεν κερδίζει ο πελάτης

Το fusion εξαρτάται από το Scribe **και** ως ψήφο **και** ως μοναδική πηγή χρόνων. Αν
«αντικατάσταση» σημαίνει μικρότερο κόστος, λιγότερη εξάρτηση από προμηθευτή ή αντοχή σε
outage, **δεν** επιτυγχάνεται και πρέπει να ειπωθεί ρητά. Επιτυγχάνεται μόνο καλύτερο
κείμενο για τον χρήστη.

### 10.9 Μετρήσεις 2026-09-02 (βήμα 2)

Τα τέσσερα fixtures του §4.4 παράχθηκαν (`make_fixtures.py`, συν ένα πέμπτο
`fixture_inputs_391.json` με τα κανονικοποιημένα tokens ανά σύστημα — απαραίτητο για
να ξανατρέξει ο αλγόριθμος από εισόδους). Το guard-on μετρήθηκε: **0.11205**
(sidn [5036, 1848, 5519, 110694], guard σε 75 από 13.414 νησιά), όχι το ≈0.1117 του
πίνακα §4.4 — ο ορισμός του guard παγώνει στο `make_fixtures.py` γιατί το pricing
script δεν διασώθηκε ποτέ. Δεύτερη κριτική Codex (`f784ace4`) πρόσθεσε: chunking
πριν την ενορχήστρωση, replay fake provider στο ίδιο interface με τους πραγματικούς,
«benchmark-shaped integration gate» χωρίς δίκτυο πριν από κάθε πληρωμένο run,
τρία stacked PRs, `/run` αντί `/runsync` στο RunPod, layered cache, και ότι το
`confidence` κρατά τη σημασία provider-confidence (το agreement πάει σε νέο πεδίο,
αλλιώς `version: 5`).

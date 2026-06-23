# Meeting talking points — 2026-06-23

Όλα τα κύρια σημεία για τη συζήτηση: αποφάσεις, ανοιχτά ζητήματα, και τα πειράματα,
με links για τις λεπτομέρειες. Διάβασέ το πριν το meeting· κάθε σημείο δείχνει πού
να ψάξεις βαθύτερα.

Πηγές-κλειδιά (όλα τα νούμερα βγαίνουν από εδώ):
- Split & publish plan: [dataset-split-and-publish-plan.md](../specs/dataset-split-and-publish-plan.md)
- Fine-tuning dry-run plan: [finetuning-dryrun-plan.md](../specs/finetuning-dryrun-plan.md)
- Mini-PC auto-research spec: [minipc-finetune-autoresearch.md](../specs/minipc-finetune-autoresearch.md)
- Auto-research αποτελέσματα: [report.md](../../data/reports/finetune-research/report.md)
- Metric HIR: [metric-hir.md](../decisions/metric-hir.md)
- Coverage ανά πόλη: `ui/static/coverage.json` + σελίδα UI `/stats/coverage`
- Έτοιμο για slides: [2026-06-23-dataset-and-metric-report.md](2026-06-23-dataset-and-metric-report.md)

---

## 1. Το πλαίσιο — δύο πράγματα, όχι ένα

Δουλεύουμε δύο ξεχωριστά artifacts που μπερδεύονται μεταξύ τους:

1. **Το fix-task** — το LLM που καθαρίζει το raw ASR κείμενο (ήδη production, Sonnet).
   Εκεί έτρεξε το glossary A/B του Σαββατοκύριακου.
2. **Το ASR fine-tuning dataset** — ζεύγη `(audio, corrected_text)` για fine-tune
   του ακουστικού μοντέλου (Whisper-v3-large). Αυτό είναι το HuggingFace "post" και
   το split.

Το meeting αφορά κυρίως το #2. Το #1 τροφοδοτεί το #2 μόνο ως triage: μας λέει ποια
λάθη τα πιάνει ήδη το LLM (χαμηλή αξία ως ASR target) vs ποια είναι γνήσια ακουστικά
(υψηλή αξία). Λεπτομέρειες: [split plan §0](../specs/dataset-split-and-publish-plan.md).

## 2. Το metric — Human Intervention Rate (HIR)

Το πραγματικό κόστος του project είναι ο χρόνος του reviewer. Ένας αριθμός το πιάνει:

> **HIR = (utterances που διόρθωσε άνθρωπος) / (σύνολο utterances)**, σε πλήρως
> reviewed meeting. Baseline σήμερα: **28.1%** (FPY = 1 − HIR ≈ 71.9%).

- Micro HIR 28.1%, Macro 30.1% (διάμεσος 28%, p10 19% → p90 45%).
- Υπολογίζεται μόνο σε meetings με `humanReview=true` (αλλιώς ο παρονομαστής χαλάει).
- Twin metric για πιο θετική αφήγηση: First-Pass Yield = το ποσοστό που το pipeline
  πετυχαίνει χωρίς ανθρώπινο άγγιγμα.

Πλήρης ορισμός + μαθηματικά (Wilson CI, clustered bootstrap): [metric-hir.md](../decisions/metric-hir.md).

**Για συζήτηση:** headline = micro ή macro; τι μετράμε ως "human touch" (τελική
κατάσταση vs οποιοδήποτε άγγιγμα στην αλυσίδα); κατώφλι "reviewed".

## 3. Τι δεδομένα έχουμε πραγματικά

Από το meeting JSON όλων των public meetings (`eval/fetch_speakers.py`):

- **571k utterances, 327 public meetings, ~378h**, 544 ταυτοποιημένοι speakers.
- **90.5%** των utterances έχουν speaker → speaker-disjoint split εφικτό.
- **0 speakers** σε >1 πόλη → split ανά πόλη = **αυτόματα speaker-disjoint**, μηδέν
  cross-city leakage. Αυτό κάνει το split εύκολο.
- Όγκος δεν είναι το πρόβλημα: ~365h corrected audio συνολικά. Ο περιορισμός είναι
  **ποιότητα labels + σωστό split**, όχι ποσότητα.

Top πόλεις (ώρες / HIR): chania 115h/22.5%, athens 75h/33.1%, sparta 29h/25.5%,
chalandri 25h/21.2%, zografou 20h/23.7%, orestiada 17h/25.6%, argos 17h/28.5%.
Πλήρης πίνακας: `ui/static/coverage.json` + σελίδα UI `/stats/coverage`.

### 3α. Πώς να εξηγήσεις το coverage (από πού βγαίνει κάθε νούμερο)

Τα νούμερα δεν είναι εκτιμήσεις — βγαίνουν από δύο fetch scripts που τραβάνε το
ίδιο το meeting JSON του OpenCouncil και τα μετράνε. Με μια πρόταση το καθένα:

**Συνεδριάσεις** (από `eval/fetch_meeting_meta.py` → `data/eval/meeting_meta.json`):

| | τιμή | τι είναι / από πού |
|---|---|---|
| Public | 327 | meetings που είναι δημόσια προσβάσιμα (η λίστα meetings· τα private δεν δίνουν δεδομένα) |
| Private (εκτός) | 84 | μη-δημόσια → εκτός dataset (411 σύνολο − 327 public) |
| Reviewed ✓ | 212 | έχουν `taskStatus.humanReview = true` → πλήρως ελεγμένα από άνθρωπο |
| Μη-reviewed ✗ | 115 | `humanReview = false` → κάποιος μπορεί να άγγιξε λίγα, αλλά δεν τελείωσε ο έλεγχος (327 − 212) |
| Πόλεις | 22 | διακριτές πόλεις στα public meetings |
| Speakers | 544 | ταυτοποιημένα `person_id` (από `eval/fetch_speakers.py`) |

**Ώρες ήχου** (από `eval/fetch_speakers.py` → `data/eval/speakers.parquet`: αθροίζει
τη διάρκεια κάθε utterance, ομαδοποιημένη κατά `lastModifiedBy` και το `humanReview` gate):

| | ώρες | τι είναι |
|---|---|---|
| Σύνολο public | 378,5h | όλο το audio των 327 public meetings |
| Reviewed | 254,4h | μόνο τα 212 reviewed meetings — **αυτό είναι το αξιόπιστο μέρος** |
| ↳ no-edit backbone | 109,3h | `lastModifiedBy = none`: ASR που ο άνθρωπος άφησε ανέγγιχτο → καθαρό ground truth |
| ↳ human-verified | 93,0h | `lastModifiedBy = user`: ο άνθρωπος έκανε την τελική διόρθωση |
| ↳ task-final | 52,1h | `lastModifiedBy = task`: το LLM fix-task έκανε την τελική αλλαγή, ο άνθρωπος την αποδέχτηκε |
| Μη-reviewed (εκτός) | 124,1h | τα 115 μη-reviewed meetings → **τα πετάμε**, δεν τα εμπιστευόμαστε |

**Η απλή εξήγηση σε μία γραμμή:** από τις 378,5h public, εμπιστευόμαστε τις 254,4h
που ένας άνθρωπος έλεγξε ολόκληρες· οι υπόλοιπες 124,1h δεν τελείωσαν review, οπότε
ένα `no-edit` εκεί δεν σημαίνει «σωστό», σημαίνει «κανείς δεν κοίταξε» → εκτός.
Οι 254,4h χωρίζονται σε τρία κομμάτια ανάλογα με το ποιος έκανε την τελευταία αλλαγή
(`109,3 + 93,0 + 52,1 = 254,4`). Το no-edit backbone (109,3h) είναι ο κορμός του
training set· τα human-verified (93h) είναι οι πολύτιμες διορθώσεις. Το ίδιο gate
ορίζει και το HIR (§2 + §4).

## 4. Η παγίδα στα labels — το `humanReview` gate

Ερώτηση από Discord: το `thessaloniki/apr1_2026` έχει >10 human-edited utterances
αλλά δεν είναι διορθωμένο εξ ολοκλήρου — το αποκλείουμε; **Ναι**, και υπάρχει καθαρό
γενικό σήμα: `taskStatus.humanReview`.

- Σε μη-ολοκληρωμένο meeting, ένα `no-edit` utterance **δεν** σημαίνει «το ASR ήταν
  σωστό» — σημαίνει «κανείς δεν το κοίταξε». Αν το βάλουμε ως ground truth,
  μαθαίνουμε λάθη στο μοντέλο.
- Κανόνας: μπαίνει στο trusted dataset / στον HIR παρονομαστή **μόνο αν
  `humanReview=true`**. Από 327 public: **212 reviewed, 115 όχι** (124h / 31% utts εκτός).

**Δύο διαφορετικά counts, διαφορετικές δουλειές** (να μην μπερδευτούν):
- **313 meetings** αξίζουν curation (≥10 human-edited utterances) → αυτά τροφοδοτούν
  το **UI** (per-correction include/exclude). Το UI **δεν** διορθώνει transcripts·
  είναι εργαλείο curation: ποιες υπάρχουσες διορθώσεις μπαίνουν στο training set.
- **212 meetings** είναι fully-reviewed → μόνο εκεί εμπιστευόμαστε το **no-edit
  backbone** + το **HIR metric** (per-meeting trust, εφαρμόζεται στο dataset build).

Λεπτομέρειες: [split plan §0d](../specs/dataset-split-and-publish-plan.md).

## 5. Το split — πρόταση

**Settled:**
- **TEST = μελλοντικά δεδομένα** (μετά 1/6/2026). Σήμερα **άδειο επίτηδες** — τα June
  meetings μπαίνουν αργότερα. Άρα **δεν χρειάζεται πόλη για test**· είναι rolling
  benchmark, παγώνει μόλις φτάσει αρκετό μέγεθος.
- **VAL = orestiada + argos** ολόκληρες (~27h). Ολόκληρες πόλεις = αυτόματα
  speaker- & meeting-disjoint, μηδέν leakage, απλό. Και οι δύο έχουν HIR κοντά στο
  28% (αντιπροσωπευτικές, όχι outliers).
- **TRAIN = οι άλλες 8 κύριες public πόλεις** — corrections τώρα + no-edit backbone μετά.

**Γιατί όχι ακριβώς το Notion split** (Argos+Vrilissia ολόκληρες → val, +30% speakers):
1. Το **Vrilissia είναι κυρίως private** → val πάνω σ' αυτό δεν δημοσιεύεται/
   αναπαράγεται. Το βγάλαμε.
2. **Whole-city → val μπερδεύει** unseen-speaker / unseen-meeting / unseen-city σε
   έναν αριθμό. Το unseen-city ανήκει στο TEST, το unseen-speaker στο VAL.
3. Το **"30% των speakers κατά πλήθος" είναι λάθος μονάδα** — οι speakers είναι
   power-law σε λεπτά· 30% του count μπορεί να είναι 5% ή 50% του audio. Αν θέλουμε
   unseen-speaker val, γίνεται κατά **speaker-λεπτά** με floor ≥10min, όχι κατά πλήθος.
4. **30% είναι υπερβολικό** — το val είναι μόνο για model selection· ~12% αρκεί και
   κρατά πιο πολύ reviewed audio για training.

Λεπτομέρειες + test strata (production benchmark, unseen-city, future): [split plan §0c](../specs/dataset-split-and-publish-plan.md).

**Για συζήτηση:** χρειαζόμαστε τη λίστα meetings του `bench.opencouncil.gr` —
αν είναι σταθερό set, το παγώνουμε ως test· αν αλλάζει, ορίζουμε δικό μας.

## 6. Σύνθεση dataset — ποιες σειρές μπαίνουν στο training

Τρία tiers labels (Codex refinement):
- ✅ **No-edit utterances = ο κορμός.** Raw ASR που ο άνθρωπος άφησε ανέγγιχτο σε
  reviewed meeting = καθαρό, άφθονο `(audio, text)`. Αποτρέπει catastrophic forgetting.
- ✅ **Human-verified finals** — γνήσια λάθη που έπιασε ο άνθρωπος· υποψήφια για ήπιο
  upweighting.
- ❌/⚠️ **`task_only`** (LLM-edited, χωρίς human sign-off) — labels μη επικυρωμένα,
  μετρήσαμε ότι το task **overcorrects**. Default: exclude· ίσως κρατάμε μόνο
  NE/acronym ως weak/down-weighted tier. Το ξεκαθαρίζει το **ablation A**.

Σύνθεση (Grok best-practice): ~**70–80% clean / 20–30% corrected**, LoRA αντί full
fine-tune, rehearsal με γενικά ελληνικά (HParl/Common Voice/FLEURS) για να μην
«ξεχάσει» το μοντέλο.

**Μεγαλύτερος κίνδυνος (Codex), να τον έχουμε μπροστά μας:** να αντιμετωπίσουμε το
correction-provenance και το "no-edit" ως αξιόπιστα labels ενώ είναι **workflow
artifacts** → εύκολο φαινομενικό κέρδος που στην πραγματικότητα αντανακλά selection
bias / templated κείμενο / σύνθεση πόλεων, όχι καλύτερο ASR.

**Απλούστερο αξιόπιστο αποτέλεσμα-στόχος:** *human-verified, acoustically-supported
corrections βελτιώνουν στοχευμένα ελληνικά ASR λάθη πάνω από ένα reviewed no-edit
backbone, υπό meeting- & speaker-disjoint αξιολόγηση.*

Πλήρες include/exclude + ablations A & B: [split plan §2 + reviewer notes](../specs/dataset-split-and-publish-plan.md).

## 7. Το "post" — δημοσίευση στο HuggingFace

- **Δημοσιεύουμε** τις 10 κύριες public πόλεις· audio + σωστό κείμενο μόνο (όχι το
  λάθος κείμενο).
- **Card:** ώρες, #πόλεις/#meetings/#speakers, μεθοδολογία split, κατανομή κατηγοριών,
  provenance (no-edit vs human-verified vs excluded task-only), τίμια caveats.
- **Reproducibility:** δημοσιεύουμε το split file + το build script.

**Για συζήτηση:** license + privacy (ονόματα speakers = PII· 84 private meetings
εκτός· να επιβεβαιώσουμε το license του OpenCouncil για derived audio+transcripts).
Λεπτομέρειες: [split plan §3](../specs/dataset-split-and-publish-plan.md).

## 8. Πειράματα — τι τρέξαμε ήδη

### 8α. Fix-task glossary A/B (Σαββατοκύριακο)
1000 corrections, on-box Sonnet. **Το glossary block ΔΕΝ βοηθάει συνολικά**
(52.8% → 51.3% edit-application). Βοηθάει **μόνο** στο `named_entity` (+9.9pp).
Held-out prompt-tuning: **στατιστικά μη σημαντικό** (HIR −2.0pp, McNemar p=0.44,
CI περνά το 0). → Η μόχλευση είναι στο ASR fine-tune + στοχευμένο glossary μόνο για
entities, όχι στο γενικό prompt-tuning. Λεπτομέρειες: [report §6](2026-06-23-dataset-and-metric-report.md).

### 8α′. Διόρθωση ονομάτων: NER-gate vs fuzzy/glossary
Ξεχωριστή γραμμή πειραμάτων (2026-06-21): μπορούμε να διορθώσουμε ονόματα
**ντετερμινιστικά** (χωρίς LLM quota), κάνοντας fuzzy-match τα ASR tokens σε μια
λίστα ονομάτων — αλλά **μόνο πάνω σε tokens που είναι όντως ονόματα**, ώστε να μη
«διορθώνουμε» άσχετες λέξεις που τυχαία ακούγονται παρόμοια (αυτό ακριβώς που κάνει
λάθος το σκέτο fuzzy matching). Το gate είναι ένα ελληνικό NER:
**GreekBERT** (`amichailidis/bert-base-greek-uncased-v1-finetuned-ner`) ή GLiNER.

- **Το NER gate μόνο του δεν φτάνει.** Με το πυκνό global glossary (~5.894 ονόματα),
  το fuzzy — **ακόμη και NER-gated** — χαλάει καθαρό κείμενο: πολλές σωστές λέξεις
  πέφτουν φωνητικά κοντά σε *κάποιο* από τα 5.894.
- **Το πραγματικό κλειδί είναι το μέγεθος του candidate set.** Αν αντί για global
  glossary βάλουμε μικρό **per-meeting roster** (τα ονόματα που όντως ειπώθηκαν σε
  εκείνη τη συνεδρίαση, ~δεκάδες), precision & retention ανεβαίνουν μονότονα.
- **Με το πραγματικό roster** (από `partiesWithPeople` του meeting): precision 0.42,
  collateral 1.2/100, clean-text retention **0.955** (vs 0.855 με το global), recall ~0.12.
- **Verdict:** όχι deployable μόνο του (95.5% retention = ακόμη χαλάει ~1 στα 20
  καθαρά, recall ~0.12 χάνει τα περισσότερα). Καλύτερος ρόλος: **συντηρητικό πρώτο
  πέρασμα δίπλα στο LLM.** Ίδιο μάθημα με το glossary A/B: **scope στενά (μικρό
  per-meeting set), ποτέ dump μεγάλης λίστας.**

Λεπτομέρειες + αριθμοί: [dynamic-vocabulary-and-entities.md](../reference/dynamic-vocabulary-and-entities.md).
Κώδικας: `eval/{fuzzy_correct,ner_gate_eval,roster_gate_eval,fetch_rosters}.py`.

### 8β. Mini-PC auto-research (Track 2) — ΕΤΡΕΞΕ ✅
16 fine-tune runs, ~1h, whisper-base CPU (Karpathy-style loop: μετέβαλε έναν data
άξονα, fine-tune, score val, keep/discard, επανέλαβε). Headline:
- **Fine-tuning σε ~17 λεπτά corrections ρίχνει το WER του κανονικού λόγου αξιόπιστα:**
  `val_reg` 0.674 → ~0.43 norm WER (~24 πόντοι) σε **κάθε** run. Domain adaptation
  δουλεύει — και **δεν** χαλάει το γενικό ASR (η ανησυχία για correction-bias δεν
  επιβεβαιώθηκε σ' αυτή την κλίμακα).
- **Οι δύσκολες residual περιπτώσεις (`val_corr`) δεν κουνιούνται:** ~0.61 ± 0.06,
  και το cross-config spread (0.064) ≈ seed spread ενός config (0.062) → **κανένας
  data άξονας δεν είναι στατιστικά διαχωρίσιμος** σε 56 val clips.
- **Τι μεταφέρεται στο large-v3:** το composition lever (corrections + no-edit
  backbone). Το lr/steps **δεν** μεταφέρεται.
- **Blocker επόμενου γύρου:** μεγαλύτερο `val_corr` (όλο το ~9.9k pool orestiada+argos,
  όχι 56) + bootstrap CIs πριν εμπιστευτούμε οποιοδήποτε mixture ranking.

Πλήρης leaderboard + per-category + caveats: [auto-research report](../../data/reports/finetune-research/report.md).

### 8γ. Kaggle large-v3 LoRA (Track 1) — σε εξέλιξη
Notebook: `notebooks/whisper_finetune_kaggle.ipynb`. Τρέχει το πλήρες pipeline στις
~1,905 included corrections: fetch `/api/export` → cut clips → LoRA fine-tune
large-v3 → WER στο val (orestiada+argos) before/after → save adapter. Τώρα με
**SMOKE mode** για γρήγορο smoke test πριν το πλήρες run. Σχέδιο: [finetuning-dryrun-plan.md](../specs/finetuning-dryrun-plan.md).

## 9. N-best / confidence output (το "καλύτερες απαντήσεις, όχι μία")

Με **faster-whisper**: `beam_size=5, word_timestamps=True` → κάθε segment έχει
`avg_logprob`, `no_speech_prob`, per-word `probability`· το beam δίνει και N-best.
Σχέδιο: κανονικό κείμενο για σίγουρα spans· για spans κάτω από **threshold**
(π.χ. `avg_logprob < -1.0` ή min word prob < 0.7) βγάζουμε compact JSON
`{text, alternatives:[…], conf}` ώστε το fix-task LLM να εστιάζει μόνο στα αβέβαια.
Το threshold το κρατά από το να ανάβει παντού. Λεπτομέρειες: [dry-run §6](../specs/finetuning-dryrun-plan.md).

**Caveat (Codex):** να επιβεβαιώσουμε ότι το faster-whisper API εκθέτει όντως beam
alternatives — οι word/token probabilities δεν είναι N-best hypotheses.

## 10. Αποφάσεις — ξεκαθαρισμένα (να μην ξανασυζητηθούν)

- [x] Metric = HIR (28.1% baseline) + WER ως ακουστικός proxy.
- [x] TEST = temporal (rolling, άδειο τώρα). Όχι πόλη για test.
- [x] VAL = orestiada + argos ολόκληρες. TRAIN = οι άλλες 8.
- [x] train/val = speaker+meeting-disjoint, όχι per-utterance.
- [x] `humanReview=true` gate για backbone & metric.
- [x] Βγάζουμε Vrilissia από το val (κυρίως private).
- [x] Glossary δεν είναι γενικό win (μόνο named_entity)· διόρθωση ονομάτων =
  scope στενά (per-meeting roster) + NER gate, ποτέ dump· συντηρητικό assist, όχι standalone.
- [x] Likely Whisper-v3-large + LoRA, freeze encoder.
- [x] Δημοσιεύουμε audio + σωστό κείμενο μόνο, 10 public πόλεις.
- [x] Δεν διορθώνουμε timestamps (~10% λαθών → τα πετάμε).
- [x] Domain adaptation δουλεύει & δεν χαλάει γενικό ASR (από Track 2).

## 11. Ανοιχτά για συζήτηση

- [?] **Residual-WER audit στα no-edit** — πόσο εμπιστευόμαστε τον κορμό; Χρειάζεται
  άνθρωπο / stronger-model re-transcription σε δείγμα. Ο μεγαλύτερος μεθοδολογικός κίνδυνος.
- [?] **task_only**: exclude εντελώς ή weak/down-weighted tier (ίσως μόνο NE/acronym); → ablation A.
- [?] **Audio normalization**: per-meeting vs per-interval; να ταιριάζει (ή όχι) με production.
- [?] **bench.opencouncil.gr**: σταθερό set → freeze ως test, ή ad-hoc → ορίζουμε δικό μας.
- [?] **License + privacy** για το HF post (PII, private meetings).
- [?] **Val size**: μόνο included orestiada+argos (154 utts) ή όλο το ~9.9k pool για
  σταθερότερο WER; (Track 2 το επιβεβαίωσε ως blocker.)
- [?] **HIR**: micro ή macro headline· ορισμός "human touch"· κατώφλι "reviewed".
- [?] **Named-entity metric**: ορισμός denominator + matching rules εκ των προτέρων.
- [?] **Export auth**: το `/api/export` είναι public+unauth· βάζουμε `?token=`;

## 12. Επόμενα βήματα (προτεραιότητα)

1. [ ] Επιβεβαίωση λίστας `bench.opencouncil.gr` → test feasibility.
2. [ ] Μεγαλύτερο val_corr (όλο το orestiada+argos pool) + bootstrap CIs.
3. [ ] Παράξουμε το canonical split CSV (meeting + speaker → train/val/test).
4. [ ] `docs/decisions/data.md`: κανόνας include/exclude + ablation plan.
5. [ ] HF card skeleton + λύση license/privacy.
6. [ ] Ολοκλήρωση Kaggle large-v3 run (smoke → full), σύγκριση με baseline.

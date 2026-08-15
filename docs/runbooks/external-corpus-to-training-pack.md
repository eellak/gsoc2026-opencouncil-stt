# Runbook: εξωτερικό σώμα ομιλίας → training pack

Πώς μπαίνει ένα δημόσιο σώμα ομιλίας στο fine-tune, από το HuggingFace μέχρι ένα
pack που τρώει ο trainer. Γραμμένο πάνω στο HParl (2026-08-14)· τα βήματα είναι τα
ίδια για κάθε επόμενη πηγή.

**Τι διαβάζεις πρώτα**

- [`docs/reference/external-source-packs.md`](../reference/external-source-packs.md) —
  το **συμβόλαιο** του pack (layout, πεδία, κανόνες). Αυτό είναι η αυθεντία για τη μορφή.
- [`docs/reports/2026-08-14-hparl-audio-text-probe.md`](../reports/2026-08-14-hparl-audio-text-probe.md)
  — η πρώτη πλήρης εκτέλεση, με τα νούμερα.
- [`docs/specs/2026-08-14-hparl-stage1-prereg.md`](../specs/2026-08-14-hparl-stage1-prereg.md)
  — τι γίνεται **μετά** το pack, και το gate. Καμία GPU χωρίς αυτό.

Όλα τρέχουν με `.venv-eval/bin/python`. Ήχος και κείμενο μένουν κάτω από
`~/.cache/oc-public/` — **ποτέ στο git**.

---

## Βήμα 0 — έλεγχος πριν κατεβάσεις τίποτα

1. **Άδεια, στην πηγή.** Το mirror δεν είναι αυθεντία. Το HParl mirror αυτοαντιφάσκει
   μέσα στην ίδια κάρτα (YAML `cc-by-4.0`, README `CC BY-NC 4.0`). Βρες τον αρχικό
   φορέα (π.χ. CLARIN) και κρίνε το NC έναντι της εμπορικής διάστασης του OpenCouncil.
   Αν είναι NC και υπάρχει εμπορική διάσταση, **σταματάς εδώ**.
2. **Πεδίο.** Κοινοβούλιο ≠ δημοτικό συμβούλιο. Ό,τι είναι εκτός πεδίου μπαίνει ως
   **στάδιο-1 προσαρμογή**, όχι ως επιπλέον in-domain δεδομένα.
3. **Ledger.** Ψάξε αν η πηγή έχει ήδη κριθεί. Το HParl είχε υποβαθμιστεί ρητά στις
   2026-08-11 πριν ξανανοίξει.
4. **Σχήμα και μέγεθος**, χωρίς πλήρες download:

```bash
.venv-eval/bin/python -c "
from huggingface_hub import HfApi
i = HfApi().dataset_info('<repo>', files_metadata=True)
print(i.cardData); print(sum(f.size or 0 for f in i.siblings)/1e9, 'GB')"
```

5. **Κοίτα το κείμενο πριν τον ήχο.** Το φθηνότερο πείραμα του project. Range-read
   μόνο τη στήλη κειμένου ενός shard και μέτρησε: τόνοι; στίξη; placeholders
   (`[UNK]`, `<spoken_noise>`); μήκος. Το `ddamianos/hparl` απορρίφθηκε εδώ, χωρίς να
   κατέβει ούτε ένα byte ήχου.

```python
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
pf = pq.ParquetFile(HfFileSystem().open("datasets/<repo>/data/<shard>.parquet", "rb"))
s = pf.read(columns=["<text_col>"]).column(0).to_pylist()
```

---

## Βήμα 1 — φιλτράρισμα: συμφωνεί το κείμενο με τον ήχο;

Πρότυπο: [`eval/hparl2_filter.py`](../../eval/hparl2_filter.py). Κάνει, ανά γραμμή:
MP3 (mono 16 kHz, 32 kbps ≈ 4 kB/s), Soniox, αφαίρεση tags, alignment έναντι του
κειμένου της πηγής, και κρατάει `align ≥ 0.95`.

```bash
# δείγμα ενός shard — πάντα πρώτα αυτό
.venv-eval/bin/python eval/hparl2_filter.py --n 150

# πλήρες/μεγάλο πέρασμα, resumable
SHARDS="data/train-00000-of-00022.parquet,data/train-00001-of-00022.parquet,..."
setsid nohup .venv-eval/bin/python eval/hparl2_filter.py \
  --shard "$SHARDS" --n 357 --workers 16 --append \
  >> ~/.cache/oc-public/<source>/pilot.log 2>&1 < /dev/null &
```

**Concurrency: 16.** Μετρημένο, όχι υποθετικό — 3 workers δίνουν 43 γραμμές/λεπτό,
12 δίνουν 126, 24 δίνουν 127, 40 δίνουν 176, με μηδέν σφάλματα παντού. Το `3` που
κληρονομήθηκε από τον gap2 verifier κόστιζε ~3x. Πάνω από ~16 το κέρδος είναι μικρό
και κάθε κλήση είναι ξεχωριστό python subprocess.

**Επανεκκίνηση:** το `--append` παραλείπει row ids που υπάρχουν ήδη, και ένα shard που
έχει ολοκληρωθεί απορρίπτεται από το parquet footer χωρίς να ξανακατέβει. Ασφαλές
σταμάτημα:

```bash
for pid in $(pgrep -f "hparl2_filter.py --shard"); do
  [ "$(cat /proc/$pid/comm)" = "python" ] && kill $pid
done
```
(`pkill -f` σκοτώνει και το ίδιο σου το shell — έχει συμβεί.)

**Τι διαβάζεις στην έξοδο.** Όχι μόνο το WER: τα σφάλματα χωρίζονται σε
*κείμενο-χωρίς-ήχο* (συντακτικές προσθήκες — δηλητηριάζουν) και
*ήχο-χωρίς-κείμενο* (η πηγή έκοψε ομιλία — διδάσκει διαγραφές). Είναι διαφορετικά
προβλήματα και δεν αθροίζονται.

**Δύο παγίδες που ήδη μας βρήκαν:**
- Το `align ≥ 0.95` σε γραμμή διάμεσου 9 tokens είναι **πρακτικά exact-match**:
  διαλέγει εύκολο ήχο, και το αποτέλεσμα είναι δείγμα *συμφωνίας με το Soniox*, όχι
  τυχαίο δείγμα. Γράψ' το στα caveats του pack.
- Τα placeholder tags **είναι** σήμα, αλλά όχι επαρκές φίλτρο: keep rate 40,8% σε
  γραμμές με `<spoken_noise>` έναντι 54,1% χωρίς (n=10.133). Στο πρώτο δείγμα των 150
  γραμμών φαινόταν 48% vs 52% και είχε γραφτεί «δεν είναι ο διαχωριστής» — ήταν
  θόρυβος μικρού δείγματος. **Μην συμπεραίνεις πλευρικούς ισχυρισμούς από n=150·**
  ο ρυθμός keep σταθεροποιείται πολύ πριν από αυτούς. Η εισαγωγή μένει στο σκορ.

---

## Βήμα 2 — επισκευή στόχου (στίξη, κεφαλαία)

Πρότυπο: [`eval/hparl2_punctuate.py`](../../eval/hparl2_punctuate.py). Η λογική:
**λέξεις της πηγής + στίξη του ASR**. Η πηγή έχει τη σωστή ακολουθία λέξεων, το ASR
έχει τη δομή.

```bash
.venv-eval/bin/python eval/hparl2_punctuate.py --llm --batch 25
```

Δύο διαδρομές, και οι δύο τρέχουν:
- **`transfer()`** — ντετερμινιστική μεταφορά στίξης σε tokens που ταιριάζουν. Το
  fallback και το τίμιο baseline.
- **`gpt-5.6-luna`** μέσω του codex bridge (δεν υπάρχει OpenAI key στο workspace· ο
  bridge είναι ο μόνος δρόμος). Χρειάζεται επειδή η ντετερμινιστική διαδρομή **δεν
  μπορεί να κρίνει αν το απόσπασμα είναι ολοκληρωμένη πρόταση**.

> **Ο κανόνας που κάνει αυτό ασφαλές: hard word guard.** Η έξοδος του LLM γίνεται
> δεκτή μόνο αν η `greek_normalize` ακολουθία tokens είναι **ταυτόσημη** με της πηγής·
> αλλιώς η γραμμή πέφτει στο `transfer()` και μετριέται. Χωρίς αυτό, ένα βήμα
> «επιμέλειας» μπορεί σιωπηλά να ξαναγράψει τον στόχο. Στο HParl: 74/74 πέρασαν.

**Τα αποσπάσματα είναι ως επί το πλείστον κομμένα** (μόνο 26% ολοκληρωμένες προτάσεις
στο HParl). Αν βάλεις τελεία παντού, μαθαίνεις στο μοντέλο να κλείνει πρόταση κάθε
~5 δευτερόλεπτα. Το prompt το απαγορεύει ρητά.

---

## Βήμα 3 — ανθρώπινος έλεγχος

```bash
.venv-eval/bin/python scripts/hparl2_review_page.py --n 40            # έλεγχος του gate
.venv-eval/bin/python scripts/hparl2_review_page.py --n 30 --only-kept # έλεγχος στίξης
cd ~/.cache/oc-public/<source>/review && python3 -m http.server 8123
```

Η σελίδα δειγματοληπτεί **στρωματοποιημένα σε όλο το εύρος alignment**, ώστε να
ελέγχεται το ίδιο το κατώφλι και όχι μόνο οι εύκολες γραμμές. `space` play, `k`
σωστό, `x` λάθος, `j` επόμενο· οι κρίσεις μένουν σε localStorage και το **Export**
κατεβάζει όλες όσες υπάρχουν, όχι μόνο της τρέχουσας σελίδας.

**Το Export είναι το παραδοτέο.** Χωρίς το JSON, το κατώφλι μένει αβαθμονόμητο έναντι
ανθρώπινης κρίσης, όσο καλή κι αν είναι η προφορική εντύπωση.

---

## Βήμα 4 — χτίσιμο του pack

```bash
.venv-eval/bin/python scripts/build_training_pack.py --source <source> --pack-id <source>-v1
```

Βγάζει `audio/ + train.jsonl + meta.json + README.md` κάτω από
`~/.cache/oc-public/training-sets/<pack-id>/`, με admission gate, dedupe, sha256 και
τα caveats **μέσα στο `meta.json`** — να ταξιδεύουν με τα δεδομένα, όχι με τη μνήμη
κάποιου.

Νέα πηγή = μία εγγραφή στο `SOURCES` του builder (τίτλος, path, άδεια, πεδίο, report,
caveats). Ο trainer δεν αλλάζει.

Μετά: καταχώρησε artifact στο [`research/ledger.json`](../../research/ledger.json) με
το `train_jsonl_sha256`, και σύνδεσε το report.

---

## Βήμα 5 — εκπαίδευση

```bash
PACK_MANIFEST=~/.cache/oc-public/training-sets/<pack-id>/train.jsonl \
PACK_ARM=pn \
python notebooks/train_runpod.py
```

`PACK_ARM=pn` όταν η πηγή δεν έχει word-level timings· το `p` (Whisper timestamp
tokens) απαιτεί χρονισμούς και ο trainer το ελέγχει ρητά.

Πριν οποιοδήποτε pod: [`docs/runbooks/runpod-training-pod.md`](runpod-training-pod.md).
**Ένα pod χρεώνει από τη δημιουργία** — watchdog με σκληρή προθεσμία *πριν* ανεβάσεις
οτιδήποτε, και κατέγραψε το pod ID.

---

## Κανόνες που δεν διαπραγματεύονται

- Το pack μπαίνει **ως δικό του arm**, ποτέ σιωπηλά ανακατεμένο στις in-domain
  διορθώσεις. Δύο στάδια, όχι μίγμα, όταν τα πεδία διαφέρουν έντονα.
- **Το deletion rate είναι η μετρική που φυλάς** όταν το pack είναι από κοντά
  αποσπάσματα. Μοντέλο που ρίχνει το WER παραλείποντας δύσκολα σημεία φαίνεται
  καλύτερο και είναι χειρότερο.
- Ξεχώριζε πάντα **συμφωνία-με-OpenCouncil** από **πιστότητα-στον-ήχο**. Το φίλτρο
  εδώ μετράει το πρώτο είδος συμφωνίας έναντι ενός vendor, όχι αλήθεια.
- 3 seeds ανά arm. Το μετρημένο per-seed εύρος είναι 2,1 μονάδες WER.
- Αν κάτι αποτύχει δύο φορές με τον ίδιο τρόπο, σταμάτα και γράψε τι έσπασε.

# Σχέδιο: serving-stack βελτιώσεις χωρίς επανεκπαίδευση, μετρημένες βήμα-βήμα

2026-08-12. Ελεγμένο από Codex (high effort, δύο περάσματα: ανάλυση λαθών + review
αυτού του σχεδίου). Στόχος: να κατέβει το benchmark WER (τώρα 0.1404 pooled
wer-nofillers, Scribe v2 0.1330) με αλλαγές στο *πώς καλείται και μετα-επεξεργάζεται*
το `artifact-ct2-fixed`, όχι στα βάρη. Κάθε αλλαγή παίρνει δικό της μετρημένο δέλτα.

Θεμέλιο: [2026-08-11-error-analysis-vs-scribe](../reports/2026-08-11-error-analysis-vs-scribe.md)
και νέα per-item ανάλυση στο πλήρες report (2026-08-12): οι διαγραφές μας είναι 3.5:1
έναντι προσθηκών (Scribe 1.6:1), 46% των αντικαταστάσεων είναι near-miss ≤2 χαρακτήρων
(επώνυμα roster + ομόηχα), και το έλλειμμα προς Scribe είναι ~450 λάθη στις seen πόλεις
συν ~139 στις unseen — δηλαδή δεν κλείνει μόνο με δουλειά στα ονόματα.

## Πρωτόκολλο μέτρησης

- **Ανάπτυξη/επιλογή ΜΟΝΟ στα 39 frozen validation παράθυρα** (manifest:
  `research/eval-freeze-2026-08/manifest.json`), τοπικά, CPU int8, ίδιο harness με το
  decode-ablation. Control = `~/.cache/oc-public/decode-ablation/eval-A.json`.
- **Τα 7 temporal holdout παράθυρα μένουν σφραγισμένα** μέχρι να παγώσει ΟΛΟ το τελικό
  stack· ανοίγουν μία φορά, όχι ανά arm.
- **Gate ανά arm, προδηλωμένο πριν φανεί αριθμός.** Κοινό local gate (Codex): point
  estimate ΔWER < 0 ΚΑΙ one-sided 95% upper bound Δ < +0.0010 (non-inferiority — το
  «CI περιέχει το μηδέν» ΔΕΝ είναι απόδειξη αβλάβειας), συν το arm-specific gate.
  Έλεγχος single-window domination (κανένα παράθυρο >50% του κέρδους) και
  leave-one-out σταθερότητα προσήμου. Bootstrap με clustering σε επίπεδο meeting.
- **Cumulative-stack τοπικό gate**: τα arms δεν αθροίζονται· μετά τα standalone
  screens τρέχει σκάλα control → +B → +B+C → +B+C+E (το D μόνο με ρητό integration
  point) και κάθε σκαλί περνά έναντι του προηγούμενου.
- **Benchmark = επιβεβαίωση, όχι ανάπτυξη.** Έως 4 GPU runs σε ΕΝΑ ίδιο pod, ως
  σκάλα: (1) νέο same-pod control, (2) control+πρώτος επιζών, (3) +δεύτερος,
  (4) τελικό stack — το τελευταίο σκαλί ΕΙΝΑΙ το τελικό νούμερο. Η υπάρχουσα γραμμή
  `oc-runpod-fixed-2026-08-10` είναι context (άλλο pod/stack), όχι baseline των δελτών.
  Κάθε run καρφώνει hashes (adapter, CT2/faster-whisper εκδόσεις, config, pod).
- **«Νίκη επί Scribe»** αναφέρεται σε δύο επίπεδα: point win (τελικό pooled <0.1330)
  και confirmed win (paired bootstrap upper bound του τελικό−Scribe < 0).

## Arms

### A — Feasibility audit λεξικού ονομάτων (καθαρή ανάλυση, μηδέν GPU)

Ό,τι ορίζει το [name-repair plan](2026-08-11-name-repair-plan.md), βήμα 0, συν
προσθήκες Codex: εκτός από τα ~73 χαμένα gold mentions, μετριούνται και τα non-name
spans που ο κανόνας θα σκεφτόταν να αλλάξει (false-candidate rate), tie/abstention
rate, clustering ανά meeting/πρόσωπο. Πύλη (υπάρχουσα): ≥15% επιλέξιμα με τον
συντηρητικό κανόνα ΚΑΙ ≥80% των oracle-επιδιορθώσιμων διαλέγουν το σωστό πρόσωπο.
Τα κατώφλια είναι screen· την τελική κρίση δίνει το E σε full-output WER.

### E — Post-hoc roster repair (μόνο αν περάσει το A)

Ντετερμινιστικός μετασχηματισμός κειμένου, κανόνας και ασφάλειες όπως στο name-repair
plan (βήματα 1-2). Προδηλώνονται: no-op χωρίς roster, idempotence
(`repair(repair(x))==repair(x)`), καμία αλλαγή ήδη-σωστών μορφών roster, gate σε
καθαρό WER + harmful changes + mention precision (όχι recall μόνο). Αναπτύσσεται στα
39-window control hypotheses. **Δεν αναπτύσσεται/κουρδίζεται πάνω στο 260-window
report.json** — μία frozen εφαρμογή εκεί μετράει ως benchmark look. Επανεκτιμάται
incrementally μετά τα B/C (ο πληθυσμός λαθών του αλλάζει).

### B — Hotwords biasing με per-meeting roster

Ένα primary arm: roster μέσω `hotwords` του faster-whisper, όλα τα άλλα ίδια με το
control. Δεσμεύσεις (Codex, από τον upstream κώδικα): `initial_prompt` ακριβώς ό,τι
έχει το control, `prefix=None` με runtime assertion (τα hotwords αγνοούνται αν
υπάρχει prefix), budget ≤160 tokens του tokenizer ελεγμένο πριν το decode (όχι σιωπηλό
truncation), σταθερή μορφοποίηση/σειρά, dedup μετά από frozen κανονικοποίηση, χωρίς
DF filtering όταν το roster χωράει.

**Τροποποίηση 2026-08-12, πριν από κάθε decode/αριθμό** (εύρημα υλοποίησης: με
«πλήρη ονόματα πρώτα, αλφαβητικά» τα 160 tokens χωρούν ~12 από 129-141 εγγραφές, τα
περισσότερα επώνυμα μένουν έξω): η σειρά γίνεται **μοναδικά κανονικά επώνυμα πρώτα**
(ένα ανά πρόσωπο, αλφαβητικά μετά την frozen κανονικοποίηση· παραλείπονται εγγραφές
με αρχικό-τελεία, κόμματα και ονόματα παρατάξεων), και πλήρη ονόματα μόνο αν
περισσέψει budget. Μεγιστοποιεί την κάλυψη προσώπων ανά token — το DS-WER μετράει
επώνυμα και το audit έδειξε ότι εκεί είναι τα near-miss λάθη.

**Τροποποίηση 2η, 2026-08-12, επίσης πριν από κάθε decode** (εύρημα: ~7.5 tokens ανά
επώνυμο σημαίνει ότι στα 160 χωρούν ~21 από 32-36 πρόσωπα, και η αλφαβητική ουρά
αποκλείεται συστηματικά — συσχετισμένο τυφλό σημείο· απόφαση Codex): η σειρά των
επωνύμων ΔΕΝ είναι αλφαβητική αλλά κατάταξη κατά
`SHA-256(frozen_salt || meeting_id || canonical_surname)`, greedy μέχρι τα 160 tokens,
ποτέ κομμένο επώνυμο. Budget παραμένει 160 (τα hotwords μοιράζονται το 448-token
decoder context — μεγαλύτερο prompt είναι δικό του ρίσκο). Προαιρετικό secondary arm,
ρητά exploratory: ίδια πολιτική στα 200 tokens, για την καμπύλη κάλυψης-μήκους.
Αλφαβητική σειρά και two-pass relevance selection απορρίπτονται. Στιγμιότυπο
υλοποίησης (παγωμένο μαζί με την πολιτική): salt `oc-hotwords-2026-08-12`,
meeting_key το πλήρες `city/meeting_id`, surname_key σε NFC+casefold, greedy
first-fit (επώνυμο που δεν χωράει παραλείπεται ολόκληρο και η σάρωση συνεχίζει).
Μόνο επώνυμα στο primary arm — καθόλου πλήρη ονόματα. Χωρίς
DF filtering όταν το roster χωράει (αλλιώς μόνο frozen λίστα συγκρούσεων με κοινές
λέξεις, ποτέ υπολογισμένη από validation/benchmark). Gate: κοινό WER gate + mention
recall πάνω κατά προδηλωμένο ποσό + mention precision non-inferior + insertions και
out-of-roster name insertions να μην ανεβαίνουν ουσιωδώς. Roster coverage στο
benchmark: 183/203 city/meeting (31/33 unseen)· τα ακάλυπτα μένουν exact no-op και
αναφέρονται χωριστά. Προϋπάρχον σήμα (εποχή σπασμένου adapter, n=59):
[2026-07-25-hotwords-biasing](../reports/2026-07-25-hotwords-biasing.md).

### C — Shifted-window consensus decode (insertion-only v1)

Δεύτερο πέρασμα με μετατόπιση **ενός** προδηλωμένου offset (15s για 30s chunks),
συγχώνευση συντηρητική: timestamps σε συντεταγμένες αρχικού ήχου, anchors ≥2 ακριβών
λέξεων, δεκτές ΜΟΝΟ προσθήκες σε ακάλυπτο VAD-supported κενό μεταξύ anchors (roll-call
ουρά: ένα anchor + ισχυρότερος κανόνας), καμία αντικατάσταση λεξικών διαφωνιών στην
v1, απόρριψη διπλότυπων φράσεων σε γειτονικό χρόνο, μονοτονία timestamps. Μηχανικός
trigger από VAD-coverage vs token-coverage, με log του trigger rate και του κόστους.
Προκαταρκτικό (πριν γραφτεί merge): έλεγχος ότι οι στοχευόμενες διαγραφές βρίσκονται
όντως μέσα σε VAD-positive ήχο. Προσοχή: αν το control δεν είχε `word_timestamps`,
πρώτα timestamp-enabled control ή απομόνωση του timestamp-only effect. Gate: ΔD<0,
ΔI/ανακτηθέντα<0.5, κοινό WER gate με τα S να συμμετέχουν, όρια RTF/trigger-rate.

### D — N-best rescoring (προαιρετικό, κερδίζει τη θέση του)

**Prereg selector 2026-08-12 (Codex), παγωμένο μετά το oracle screen (ceiling 0.0071)
και πριν από κάθε scoring του selector:** log-linear μίας παραμέτρου πάνω σε distinct
(μετά την frozen κανονικοποίηση) υποθέσεις ανά chunk:
`argmax_h [(A(h)-A(top1))/s_A + λ·(L(h)-L(top1))/s_L]`, όπου A = CT2 score
κανονικοποιημένο ανά token (λ=0 πρέπει να αναπαράγει ακριβώς το beam8-top1) και
L = per-word logprob από word 4-gram KenLM (modified Kneser-Ney). λ από το frozen
πλέγμα {0, 0.25, 0.5, 1, 2, 4}, ισοπαλίες στο μικρότερο λ και μετά στο χαμηλότερο
beam rank, εναλλακτική μόνο με score αυστηρά > 0. ΚΑΝΕΝΑ άλλο feature στο
confirmatory (roster/μήκος/επανάληψη = exploratory μόνο). Cross-fitting:
leave-one-meeting-out (31 folds), αντικειμενική συνάρτηση το pooled window-level
S+D+I των υπόλοιπων 30 meetings· αναφέρονται ΜΟΝΟ τα stitched out-of-fold. Gate:
ανάκτηση ≥25% του oracle ceiling (≥22 καθαρά λάθη, ~−0.0018) ΚΑΙ το κοινό
non-inferiority (upper bound < +0.0010) με meeting-clustered paired bootstrap χωρίς
refit μέσα στα replicates. LM corpus: μόνο seen-city κείμενα, παγωμένο manifest με
hashes πριν το scoring, ρητός αποκλεισμός κάθε υλικού argos/orestiada.

Πρώτα oracle ceiling με fixed `beam_size=8`/8 hypotheses, T=0 — απαιτεί μικρό wrapper
γιατί το faster-whisper καταναλώνει μόνο το `sequences_ids[0]`. Προχωρά ΜΟΝΟ αν το
ceiling ≥0.0040 pooled ΚΑΙ ένας frozen/cross-fitted selector (LM + roster + length
penalty, επιλογή μόνο μεταξύ ακουστικών υποθέσεων) πιάνει προδηλωμένο κλάσμα του
oracle και περνά το κοινό gate. Αλλιώς κόβεται χωρίς benchmark slot.

## Σειρά εκτέλεσης

1. Αναπαραγωγή control (replay = eval-A.json ακριβώς, ή εξήγηση κάθε διαφοράς).
2. A (audit) → 3. E offline στα 39 → 4. B → 5. C → 6. D αν δικαιολογείται →
7. frozen cumulative σκάλα τοπικά → 8. temporal holdout μία φορά →
9. benchmark σκάλα (≤4 runs, ένα pod).

## Έλεγχοι πριν από κάθε scoring (TDD, επιλογή Codex)

Scorer fixture με χειροϋπολογισμένο WER/S/D/I· bootstrap fixture με fixed seed·
control replay· no-op σε άδειο/λάθος roster· hotword budget assertion· idempotence
του repair· C merge identity (ίδια περάσματα → baseline)· C ποτέ εκτός VAD gap·
endpoint byte-identical στο control με τα interventions κλειστά.

## Κίνδυνοι επαναχρησιμοποίησης των 39 παραθύρων

4-5 arms στο ίδιο set = winner's curse. Μετριασμός: παγωμένα configs/gates πριν από
κάθε αποκάλυψη αριθμού, όχι κούρδισμα μετά από αποτυχία, λίγες παραλλαγές ανά
οικογένεια, τοπικά CIs = screening όχι confirmatory, Holm/simultaneous intervals για
per-arm ισχυρισμούς, και το 260 benchmark ως το ΕΝΑ confirmatory στάδιο.

## Ledger

- A/E: `exp-2026-08-11-name-repair` (υπάρχον OPEN)
- B: `exp-2026-07-25-hotwords` (υπάρχον OPEN)
- C/D + σκάλα: `exp-2026-08-12-serving-stack` (νέο OPEN)

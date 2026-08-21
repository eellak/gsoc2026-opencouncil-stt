# Τα residual correction errors συγκεντρώνονται στα προβληματικά boundaries

Η ανάλυση επαναχρησιμοποίησε χωρίς νέο decode τις 151 correction rows του παγωμένου
training sample και τα ίδια fixed/base hypotheses. Μετρά agreement με τα training
labels, όχι fidelity προς τον ήχο.

## Αποτέλεσμα

Το συνολικό fixed WER είναι 22,61% (171 S / 30 D / 64 I, 1.172 reference tokens).
Η ισχυρότερη διαφορά είναι στα boundaries:

| stratum | rows | tokens | fixed WER | base−fixed |
|---|---:|---:|---:|---:|
| boundary `ok` | 90 | 726 | 17,36% | +17,77 π.μ. |
| `suspect_bleed_in` | 15 | 94 | 25,53% | +27,66 π.μ. |
| `suspect_cut_end` | 18 | 120 | 31,67% | +33,33 π.μ. |
| `suspect_cut_start` | 23 | 195 | 35,90% | +29,23 π.μ. |
| συνολική adjustment >1 s | 25 | 198 | 38,89% | +26,77 π.μ. |

Στα `suspect_cut_start`, οι insertions μόνο είναι 12,31%, έναντι 3,31% στα `ok`.
Αυτό είναι συμβατό με label/audio boundary mismatch, αλλά δεν το αποδεικνύει χωρίς
ακρόαση. Οι μικρές clips έχουν επίσης μεγαλύτερο residual: 27,01% κάτω από 2 s,
24,55% στα 2–4 s και 16,57% στα 4–8 s. Η sample δεν περιέχει καμία overlap row,
οπότε ο predeclared overlap έλεγχος είναι μη μετρήσιμος.

Στις επικαλυπτόμενες category strata, τα υψηλότερα eligible WER είναι
`other_lexical` 30,80%, `acronym_abbreviation` 28,95% και `named_entity` 24,14%.
Δεν αθροίζονται και δεν είναι ανεξάρτητα από duration/boundary.

Το base−fixed είναι θετικό σε κάθε παραπάνω stratum. Άρα το adapter έχει μάθει και
αυτές τις κατηγορίες· το residual δεν είναι απλώς πλήρης αποτυχία εκμάθησης.

## Απόφαση

Δεν αφαιρούμε ή downweightάρουμε rows από αυτό το aggregate. Πάγωσε model-blind
queue 36 clips από `adjustment>1`, `suspect_cut_start` και `other_lexical` στο
`~/.cache/oc-public/training-residual-audit-2026-08/review.jsonl`. Ο reviewer βλέπει
μόνο ήχο και training label, όχι hypothesis, WER, model ή stratum.

Αν η ακρόαση επιβεβαιώσει κακά boundaries/labels, το επόμενο arm γίνεται dense core
με προστατευμένη και διορθωμένη boundary lane. Αν τα labels είναι πιστά, το εύρημα
στηρίζει dense context ως mechanism αντί για data deletion. Μέχρι τότε το audit είναι
OPEN και δεν αλλάζει training data.

Frozen design: [training residual audit](../specs/2026-08-19-training-residual-audit.md).
Aggregate: `eval/results_training_residual_audit.json`.

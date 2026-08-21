# Ο σημερινός control έχει seed range 0,285 μονάδες, όχι 2,1

2026-08-18 · ticket
[«Πώς κρίνουμε αν ένας training arm κέρδισε»](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/44)
· aggregates: [`eval/results_seed_calibration.json`](../../eval/results_seed_calibration.json)

## Απάντηση

Το **2,1** δεν ήταν variance ή standard deviation. Ήταν το εύρος τριών paired
treatment effects του παλιού mixture experiment. Στο σημερινό evaluation harness,
οι τρεις ήδη εκπαιδευμένες control replicas της ίδιας πλήρους συνταγής δίνουν:

| control | seed | WER | del | ins | sub |
|---|---:|---:|---:|---:|---:|
| `A_s13` | 13 | 0,15557 | 0,06070 | 0,01864 | 0,07623 |
| `A_s29` | 29 | 0,15582 | 0,06062 | 0,02032 | 0,07489 |
| `A_s47` | 47 | 0,15842 | 0,06129 | 0,02351 | 0,07363 |

Σε 39 παγωμένα validation windows / 31 meetings / 11.911 reference tokens:

- seed-effect range WER: **0,285 μονάδες**,
- sample SD των τριών control WER: **0,158 μονάδες**,
- ίδιο seed 13, δεύτερο full run (`A_s13` έναντι `artifact-adapter-fixed`):
  **−0,336 μονάδες**, περιγραφικά μόνο,
- το `win_argos_feb27_2_2026_1959208` κουβαλά **23 από τα 34 net errors**, δηλαδή
  **67,6%** του range μεταξύ καλύτερου και χειρότερου seed.

Άρα η σημερινή control διακύμανση είναι πολύ μικρότερη από τη συντομογραφία «2,1
μονάδες ανά seed», αλλά το αποτέλεσμα δεν δικαιολογεί τον αντίθετο υπερβολικό
ισχυρισμό ότι ένα seed αρκεί.

## Τι δεν μπορεί να υπολογίσει αυτή η calibration

Η SD τριών **control outcomes** δεν είναι η SD paired **treatment effects**. Ένα
control-only power formula επιστρέφει παράλογα ένα run για effect μίας μονάδας· αυτό
δείχνει ότι λείπει το σωστό estimand, όχι ότι confirmation γίνεται με ένα seed.

Ο κανόνας είναι επομένως:

1. τρία paired seeds στο 300-step `large-v3` screen,
2. εκτίμηση της paired-effect SD από αυτά τα deltas,
3. τότε μόνο power calculation για 80% ισχύ και two-sided 95% interval,
4. μέχρι τότε κάθε arm γράφεται `SCREEN`, ποτέ `CONFIRMED WIN`.

Η πλήρης evidence ladder και τα promotion gates ζουν στην απόφαση
[`docs/decisions/training-evidence.md`](../decisions/training-evidence.md).

## Επιφυλάξεις

- Μόνο τρία διαφορετικά seeds· η SD είναι ασταθής.
- Το same-seed repeat έχει ίδια πεδία `run_meta` και διαφορετικό weight hash, αλλά
  δεν καταγράφηκαν identical batch-order και hardware attestations. Μετρά συνολική
  run nondeterminism περιγραφικά, όχι καθαρή hardware ή seed συνιστώσα.
- Το μεγαλύτερο παράθυρο κουβαλά 67,6% του net range. Κανένα headline delta από
  αυτά τα controls δεν επιβιώνει ως γενικός ισχυρισμός χωρίς dominance check.
- Agreement-with-OpenCouncil, όχι fidelity-to-audio.
- Κοινό local CPU int8 stack, frozen beam-5 config και κοινά per-window decode seeds.
- Τα hypothesis texts μένουν μόνο στο `~/.cache/oc-public/seed-calibration-2026-08/`.
  Κανένα transcript ή audio δεν μπήκε στο git.
- Μηδέν GPU, μηδέν paid API, μηδέν sealed holdout windows.

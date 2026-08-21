# Handoff: πού είμαστε και τι πιάνεις μετά

> **Superseded for the current 18–23 August project scope on 2026-08-19.** Read
> [`2026-08-19-handoff-claude.md`](2026-08-19-handoff-claude.md) first. This file is
> historical context for the state before the training experiments.

2026-08-18, μετά το commit `0d686b15`. Προθεσμία GSoC **23/8**.

## Διάβασε πρώτα, με αυτή τη σειρά

1. `CLAUDE.md` — το πρωτόκολλο. Δεν είναι προαιρετικό.
2. `research/ledger.json` — η αυθεντία. 65 πειράματα. Ψάξε εκεί **πριν** προτείνεις οτιδήποτε.
3. `CURRENT.md` — τι μπλοκάρει.

Ο χάρτης και τα tickets ζουν στα GitHub Issues του `eellak/gsoc2026-opencouncil-stt`
(label `wayfinder:map` για τον χάρτη, `wayfinder:*` για τα υπόλοιπα).

## Πού είμαστε σε τρεις γραμμές

Το μοντέλο μόνο του δεν βελτιώθηκε· η **σύνθεση τριών ASR** ναι, και πολύ: WER
0.1201 → **0.1005** στα 247 παράθυρα, με CI που αποκλείει το μηδέν και τα τρία είδη
σφάλματος να πέφτουν μαζί. Ο mentor έκρινε στις 18/8 ότι αυτό είναι **εντός scope**.
Το ταβάνι τέλειου διαιτητή ανά στήλη είναι 0.0475, αλλά περίπου το μισό της απόστασης
δεν κλείνει με **καμία** ψηφοφορία πάνω σε αυτά τα τρία κείμενα.

## Ο ένας περιορισμός που ορίζει κάθε training απόφαση

Το παλιό **2,1** ήταν range τριών treatment effects, όχι seed variance. Η σημερινή
CPU calibration των τριών control replicas δίνει range **0,285 μονάδες WER** και
sample SD **0,158**, αλλά αυτό δεν είναι ακόμη paired-effect SD. Μια εκπαίδευση με
έναν seed παραμένει μόνο screen· confirmation χρειάζεται paired pilot, power 80% και
two-sided 95% CI σύμφωνα με `docs/decisions/training-evidence.md`.

## Τι πιάνεις, με σειρά

| # | ticket | γιατί αυτή η σειρά |
|---|---|---|
| [#44](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/44) | ο κανόνας μέτρησης | χωρίς αυτό, τα #40/#41/#42 παράγουν νούμερα που δεν διαβάζονται |
| [#36](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/36) | WER στο training set | CPU, μηδέν κόστος, ο mentor το ζήτησε ρητά |
| [#37](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/37) | το email της Πέμπτης | μπλοκάρεται από το #36 |
| [#41](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/41) | παράθυρα 30s στην εκπαίδευση | το πρώτο βήμα είναι μέτρηση σε CPU, χωρίς εκπαίδευση |
| [#42](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/42) | καθαρότερο dataset | το ίδιο: μέτρα τι επιβιώνει πριν ξοδέψεις GPU |
| [#40](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/40) | CPT / broad adaptation | διάλεξε **έναν** μηχανισμό από τους τρεις |
| [#43](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/43) | διαχωρισμός ομιλητών στην επικάλυψη | ο φραγμός είναι το **enrollment**, όχι ο διαχωριστής |
| [#39](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/39) | `exclusive:true` στο pyannote | παράλληλο, δωρεάν |

Κάθε ticket έχει μέσα του τη δουλειά. Μην τα ξαναγράψεις — υλοποίησέ τα.

## Δρόμοι ήδη κλειστοί με μετρημένο αρνητικό. Μην τους ξαναπροτείνεις

- **περισσότερα δεδομένα** — ~1.300 ώρες αγοράζουν ~0,5 μονάδες
- **στοχευμένη εκπαίδευση διαγραφών** — *ανέβασε* τις διαγραφές 0.0600 → 0.0788, CI αποκλείει το μηδέν
- **εξωτερικά πακέτα πάνω σε αυτό** — κάθε paired CI περιλαμβάνει το μηδέν
- **hotword biasing στα ονόματα** — ανάκληση 51% → 65%, αλλά +0,34 WER σε διαγραφές
- **LLM ως συνθέτης ή διαιτητής** — +0.000467, CI αποκλείει το μηδέν, δηλαδή χειρότερα
- **beam 2** — χειρότερο από beam 5 κατά +0.0095

## Σκληροί κανόνες

- Κείμενο μεταγραφών και ήχος **ποτέ** στο git. Caches στο `~/.cache/oc-public/`.
- Τα σφραγισμένα παράθυρα αξιολόγησης μένουν σφραγισμένα. Το W **δεν** έχει test νούμερο,
  εκ σχεδιασμού· ο υπολογισμός του ξοδεύει το holdout και θέλει ρητή απόφαση του χρήστη.
- `eval/controlled_eval/msa.py` **δεν** επεξεργάζεται· το sha256 του (`3751fe5a13320e2b`)
  κλειδώνει cache 18 MB.
- Πάγωσε την πύλη **πριν** δεις νούμερο. Χαμηλότερο WER με υψηλότερες διαγραφές είναι
  **αποτυχία**, όχι συναλλαγή.
- Έλεγξε **κυριαρχία ενός παραθύρου** πριν αναφέρεις delta. Ένα παράθυρο έχει ήδη δώσει
  το 67% ενός τίτλου εδώ.
- **Καμία δαπάνη GPU χωρίς ρητή απόφαση του χρήστη.** Ένα pod χρεώνει από τη δημιουργία.
  Αν χρειαστεί pod, δες `docs/runbooks/2026-08-18-chunking-arms-247-gpu.md` — περιγράφει
  πώς απέτυχε δύο φορές στη σειρά.
- Κάθε ticket που κλείνει ενημερώνει το ledger record στην **ίδια** αλλαγή, και τρέχει
  `python3 scripts/check-research-state.py`.

## Τι υπάρχει ήδη και δεν το ξαναφτιάχνεις

- `eval/chunking_decode.py` — αρμοί V / P / Π / Ε με tests. Προσοχή: το `eval_rows()` είναι
  κλειδωμένο στα 39 παράθυρα και πετάει `ValueError` αλλιώς.
- `eval/tsfusion/` — η σελίδα διάγνωσης, 110 tests. Δες
  `docs/reports/2026-08-18-timestamp-diagnostic.md`.
- `eval/controlled_eval/fusion_lab.py` — το υπόστρωμα των 247 παραθύρων και το W.
- `eval/controlled_eval/anchored.py`, `anchor_timings.py` — γραμμένα, **ποτέ τρεγμένα**.

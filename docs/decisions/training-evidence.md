# Αποδεικτική σκάλα για training arms

Απόφαση 2026-08-18. Κάθε training arm περνά διαδοχικές βαθμίδες· ένα φθηνό
πείραμα αποφασίζει μόνο αν αξίζει η επόμενη δαπάνη, όχι αν η ιδέα κέρδισε.

## Γλώσσα αποτελεσμάτων

- **`SCREEN — ADVANCE`**: χαμηλότερο σημειακό validation WER, χωρίς αύξηση
  διαγραφών και με όλους τους φρουρούς περασμένους. Επιτρέπει μόνο την επόμενη
  βαθμίδα.
- **`SCREEN — STOP`**: αποτυχία λειτουργικής πύλης. Σταματά η δαπάνη· δεν
  αποδεικνύεται ότι ο μηχανισμός είναι γενικά βλαβερός.
- **`CONFIRMED WIN`**: προδηλωμένο, επαρκώς powered paired-seed αποτέλεσμα με
  95% CI κάτω από το μηδέν και όλους τους φρουρούς περασμένους.

Οι λέξεις «υποσχόμενο», «κέρδισε» και «αρνητικό εύρημα» δεν χρησιμοποιούνται για
μονο-seed ή υποτροφοδοτημένα arms.

## Τι αποφασίζει

Πρωτεύον endpoint είναι το WER στο παγωμένο validation set. Κάθε run αναφέρει
και training-set WER, όπως ζήτησε ο mentor, αλλά αυτό δεν επιλέγει arm.

Χαμηλότερο WER μαζί με υψηλότερο σημειακό deletion rate είναι αποτυχία, όχι
συναλλαγή. Αναφέρονται επίσης `(S+D)/N`, insertion rate, substitution rate,
hypothesis/reference length ratio και κυριαρχία ενός παραθύρου. Το `(S+D)/N`
είναι διαγνωστικό, όχι άτρωτο στο hypothesis padding.

## Η σκάλα

1. **Local mechanism preflight, μηδέν GPU.** Ένας απομονωμένος μηχανισμός·
   έλεγχοι feasibility, data yield, leakage, label quality και invariants.
2. **Tiny proxy search, μηδέν GPU.** Έως 8 configs × 3 seeds και έως 2 ώρες CPU.
   Το proxy μπορεί να ασκήσει veto σε αδρανές ή καταστροφικό config· η κατάταξή
   του δεν μεταφέρεται στο `large-v3` ως εύρημα.
3. **Short-horizon `large-v3` screen.** Έως 3 candidates, 300 παγωμένα steps ×
   3 paired seeds. Ιστορική τάξη κόστους: 6–7 GPU-hours / περίπου $3 συνολικά.
4. **Medium screen.** Έως 2 survivors, συνέχεια μέχρι 1.800 steps. Ιστορική
   τάξη πρόσθετου κόστους: περίπου 14 GPU-hours / $6–7.
5. **Full training.** Ένας survivor, 7.242 steps × 3 seeds. Ιστορική τάξη
   κόστους: περίπου 22 GPU-hours / $10 όταν τα υπάρχοντα controls είναι έγκυρα,
   διπλάσια όταν χρειάζονται νέα.
6. **Confirmation.** Το πλήθος paired seeds προκύπτει από την τρέχουσα
   calibration. Τρία πλήρη seeds παραμένουν `SCREEN` όταν δεν δίνουν 80% ισχύ.

Κάθε μετάβαση σε νέα GPU βαθμίδα χρειάζεται νέα, ρητή απόφαση του χρήστη. Τα
ιστορικά ποσά είναι εκτιμήσεις, ποτέ προέγκριση δαπάνης.

## Promotion gate

Στο short-horizon screen και σε κάθε επόμενο checkpoint απαιτούνται όλα:

1. mean paired ΔWER `< 0`,
2. τουλάχιστον 2/3 seeds με ΔWER `< 0`,
3. mean deletion delta `≤ 0`,
4. insertion delta `< +0.0005`,
5. καμία αντιστροφή προσήμου σε leave-one-window-out,
6. κανένα παράθυρο πάνω από 25% του καθαρού κέρδους.

Arm που περνά στα 300 steps αλλά αποτυγχάνει στα 1.800 σταματά. Το σφραγισμένο
test set δεν ανοίγει χωρίς ξεχωριστή απόφαση του χρήστη.

## Έρευνα εκτός repository

Αν μια απόφαση χρειάζεται φρέσκια εξωτερική έρευνα, η διαδρομή είναι `ssh laptop`
και εκτέλεση του Grok Research skill στο MacBook. Τα τοπικά ledger records και οι
προδηλώσεις παραμένουν η αυθεντία για ό,τι έχει ήδη μετρηθεί.

## 2026-08-19 — Agreed decision tree for the remaining training work

**Status: accepted by the user after a grilling session.** The primary endpoint
remains validation WER. Training-set WER is diagnostic only and never promotes an
arm.

The running dense screen compares the same rows in two shapes: A is isolated short
utterances and B is the existing dense packs. At fixed optimizer steps, B also
delivers more labelled speech per update. Therefore a passing B supports only the
claim that **dense packing + useful-token density uses a fixed training budget more
effectively**; it does not isolate long context as the cause. An equal-labelled-
seconds ablation is required before making that narrower causal claim.

The 39 frozen windows remain the fast screen substrate. The existing promotion gate
above is unchanged. A recipe is worth shipping only if it later improves the strict,
audio-faithful validation by at least **0.5 absolute WER points** (`ΔWER <= -0.005`)
without a material regression on protected slices. A screen may advance with the
directional gate; it cannot satisfy this shipping claim by itself.

The execution order is:

1. finish and report the current dense 300-step paired-seed screen;
2. complete the frozen 36-item blind boundary/label listening audit;
3. freeze a hybrid-data arm consisting of a clean core plus an audited protected
   name/hard-speech/boundary lane;
4. compare that arm with an equal-source-hours control under the same recipe and
   compute budget;
5. request explicit approval before every medium, full or holdout stage.

No stage launches the next GPU stage automatically. If both the dense screen and one
properly constructed hybrid-data screen fail their preregistered gates, training
research stops for this cycle and the existing model/fusion stack is shipped. The
seven temporal holdout windows remain sealed until a candidate has passed both its
screen and the strict validation gate.

# Τα training windows είναι πράγματι εκτός κατανομής — ο dense arm χτίζεται χωρίς νέα δεδομένα

2026-08-18. CPU-only audit με παγωμένους ορισμούς στο
[spec](../specs/2026-08-18-training-window-density-audit.md). Κώδικας:
`eval/training_window_density.py`. Aggregate-only αποτέλεσμα:
`eval/results_training_window_density.json`. Δεν έγινε εκπαίδευση, δεν χρησιμοποιήθηκε
GPU/API και δεν ανοίχτηκε κανένα sealed holdout.

## Ποιο training set μετρήθηκε

Το `data/hf-dataset/public/train.parquet`: **28.967 rows**, 13.929 διορθώσεις και
15.038 `no_edit`. Είναι ακριβώς η ταυτότητα του control arm A: τα raw spans αθροίζουν
**22,4725 ώρες**, όπως στην αναφορά του mixture experiment.

Η φράση «2,79 δευτερόλεπτα ομιλίας» χρειάζεται μία διόρθωση ορολογίας. Τα **2,793 s**
είναι το μέσο `end-start` του annotation, άρα proxy του speech span — όχι μέτρηση VAD.
Ο trainer κόβει στο διορθωμένο boundary και προσθέτει ακόμη 0,2 s σε κάθε πλευρά. Το
μέσο intended audio clip που περνά στον feature extractor είναι επομένως **3,553 s**,
πριν από το τελικό clamp στο τέλος του source recording.

| ποσότητα | μέσος | διάμεσος | occupancy σε 30 s |
|---|---:|---:|---:|
| raw speech-span proxy | 2,793 s | 2,064 s | 9,31% mean |
| aligned span | 3,153 s | 2,480 s | — |
| intended clip audio | **3,553 s** | **2,880 s** | **11,84% mean / 9,60% median** |
| digital padding | — | — | **88,16% mean / 90,40% median** |

Άρα το παλιό «91% padding» είναι σωστή τάξη μεγέθους, αλλά χρησιμοποιεί το raw
speech-span proxy. Για το audio που πραγματικά δίνει ο trainer στον encoder, η καλύτερη
εκτίμηση από το manifest είναι **88% digital padding**.

## Πού κάθεται μέσα στα 30 s

Το waveform ξεκινά πάντα στο encoder time 0 και όλο το digital padding μπαίνει δεξιά.
Το raw speech proxy αρχίζει στη διάμεσο στα **0,40 s** και τελειώνει στα **2,51 s**.
Το **87,6%** όλου του raw-span χρόνου βρίσκεται στα πρώτα 5 s. Το intended clip audio
είναι ακόμη πιο συγκεντρωμένο: **89,1%** του support του βρίσκεται στα πρώτα 5 s, και
κανένα clip δεν φτάνει στη ζώνη 25–30 s.

Δεν πρόκειται απλώς για «πολύ padding». Η εκπαίδευση δείχνει σχεδόν πάντα:

`[λίγο audio στην κεφαλή][μακρύ digital silence μέχρι τα 30 s]`

## Τι βλέπει το inference

Στα ίδια ήδη-read 247 benchmark clips του confidence substrate ανακατασκευάστηκαν
**1.148** faster-whisper windows με τον υπάρχοντα frozen seek replay. Και τα 247 WAV
headers ήταν διαθέσιμα.

- μέσο available source audio: **27,356 s / 91,19% occupancy**;
- διάμεσος: **30 s / 100% occupancy**;
- **948/1.148 (82,6%)** windows είναι ολόκληρα 30 s;
- **972/1.148 (84,7%)** έχουν τουλάχιστον 25 s source audio;
- ως περιγραφικός μόνο proxy, **83,3%** έχουν emitted word και στη ζώνη 25–30 s,
  με διάμεσο τελευταίου emitted word στα **29,54 s**.

Η μέση διαφορά audio occupancy inference μείον training είναι **+79,34 ποσοστιαίες
μονάδες**· στη διάμεσο είναι **+90,40 μονάδες**. Η ασυμφωνία κατανομής επομένως δεν
είναι υπόθεση: είναι μεγάλη και άμεσα μετρημένη. Αυτό δεν αποδεικνύει ότι προκαλεί WER
ή deletions.

## Μπορεί να χτιστεί dense arm;

Ναι, σε επίπεδο data construction. Η υπάρχουσα greedy packer προσομοιώθηκε in-memory
πάνω στα ίδια rows, χωρίς να γραφτεί text/audio:

| | αποτέλεσμα |
|---|---:|
| packs | **3.877** |
| mean / median duration | **26,15 / 27,54 s** |
| packs στα 20–30 s | **93,27%** |
| mean encoder occupancy | **87,16%** |
| aligned source hours retained | **25,369 h** |
| inserted labelled silence | **2,788 h** |
| dropped utterances / gate failures | **0 / 0** |

Και οι τέσσερις παγωμένες feasibility gates πέρασαν. Άρα ο arm δεν χρειάζεται νέα
δεδομένα και δεν μπλοκάρεται από την κατασκευή.

## Πώς προχωράει χωρίς να ξανατρέξουμε το παλιό λάθος

Το παλιό packed P/Pn experiment **δεν απομόνωσε** το clip shape από βήματα/epochs και
timestamp supervision· η δική του αναφορά το καταγράφει ήδη. Δεν αποτελεί δίκαιη
σύγκριση του σημερινού ερωτήματος.

Το επόμενο training screen, μόνο μετά από ρητή έγκριση GPU, πρέπει να αλλάζει μία
μεταβλητή:

- A: τα σημερινά single-utterance clips,
- B: τα ίδια ακριβώς rows σε dense packs, χωρίς timestamp tokens,
- ίδιο base checkpoint, optimizer budget, sampler exposure και paired seeds,
- πρώτο screen στα 300 steps υπό τον κοινό κανόνα WER/deletion/insertion/dominance.

Οι inference segmentation arms ξαναμετρώνται **μόνο αν** ο B προαχθεί. Διαφορετικά θα
τρέχαμε ταυτόχρονα αλλαγή weights και αλλαγή serving χωρίς να ξέρουμε ποια έκανε τι.

## Όρια

- Το training speech span είναι annotation proxy, όχι VAD ανά frame.
- Το pod-built WAV cache του control δεν υπάρχει τοπικά. Το intended clip audio είναι
  ακριβές ως προς τον trainer μέχρι το τελικό clamp στο τέλος του source recording.
- Τα inference word positions είναι έξοδος του μοντέλου, όχι ανθρώπινο ground truth.
- Το audit αποδεικνύει distribution mismatch και construction feasibility, όχι βελτίωση
  μοντέλου.


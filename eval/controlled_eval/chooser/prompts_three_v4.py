# -*- coding: utf-8 -*-
"""Variant D: Fable's B, with two changes I can defend from the data.

CHANGE 1 - redundant edit addressing. B addresses an edit by POSITION only. An
off-by-one then produces valid JSON that deletes the wrong word, and nothing
detects it. Asking for the position AND the word lets the parser cross-check and
reject the mismatch. The median span here is 3 tokens and p90 is 7, so quoting the
word costs almost nothing; 9.3% of spans repeat a token, so the word alone would be
ambiguous and the position alone is unverifiable. Both, or neither works.

CHANGE 2 - the prior gets its number. "Σύστημα Α is the most common base" is true
but vague. Measured on all 2,196 islands of this category: Α is the nearest base
60.4% of the time while being exactly right only 37.2%. That gap is the whole point
of allowing edits, and stating it tells the model what the edit is FOR.

CHANGE 3 - a rule for long spans. Median 3 tokens, max 41. Counting into a 41-token
span is a lottery, so edits are refused there and the model must pick whole or
abstain.
"""

D = {
"instr": """Στα δύο τρίτα των σημείων ΜΙΑ ολόκληρη διαδρομή είναι ακριβώς σωστή. Στο υπόλοιπο τρίτο το σωστό είναι μία διαδρομή με ΜΙΑ μικρή διόρθωση. Μετρημένο σε 2.196 σημεία αυτής της κατηγορίας, όχι εικασία:

• Το Σύστημα Α είναι η σωστή ΒΑΣΗ στο 60% των σημείων, αλλά είναι ολόκληρο σωστό μόνο στο 37%. Η διαφορά είναι ακριβώς οι μικροδιορθώσεις. Ξεκίνα από το Α εκτός αν το συμφραζόμενο δείχνει αλλού.
• Στο 72% των διαγραφών η περιττή λέξη είναι στην ΑΡΧΗ ή στο ΤΕΛΟΣ της διαδρομής — μπάζει από τη διπλανή ομιλία. Μικρή λέξη: να, ναι, και, εε, ε.
• Κάθε σωστή λέξη αντικατάστασης υπάρχει ήδη σε άλλη διαδρομή, χωρίς εξαίρεση. Συνήθως δίδυμο κλίσης ή μορφής: το/τον, στη/στην, απ/απο, 26/2026.

Δούλεψε έτσι: βρες τη διαδρομή που δένει καλύτερα με το αριστερό ΚΑΙ το δεξί συμφραζόμενο. Μετά διάβασέ την λέξη-λέξη απέναντι στις άλλες δύο. Αν στέκει ολόκληρη, δώσε κενό e. Αν περισσεύει ή κουτσαίνει μία λέξη, γράψε τη διόρθωση.

Δεν γράφεις ελεύθερο κείμενο και δεν προσθέτεις λέξεις που δεν είπε κανένα σύστημα. Δεν διορθώνεις γραμματική, δεν ομαλοποιείς, δεν βελτιώνεις το ύφος. Ο προφορικός λόγος είναι σπαστός και έτσι πρέπει να μείνει.

Αν η διαδρομή που διάλεξες έχει πάνω από 10 λέξεις, ΜΗΝ κάνεις διόρθωση: δώσε την ολόκληρη ή ABSTAIN. Σε τόσο μεγάλο κομμάτι η αρίθμηση δεν είναι αξιόπιστη.

Αν τίποτα δεν δένει ούτε με διόρθωση, απάντησε ABSTAIN.""",

"format_note": """Απάντησε με πίνακα JSON, ένα αντικείμενο ανά σημείο.

Κάθε διόρθωση δηλώνει ΚΑΙ τη θέση ΚΑΙ τη λέξη που βρίσκεται εκεί, ώστε να
διασταυρώνονται. Αν δεν συμφωνούν, η απάντηση απορρίπτεται.

  {"i":0,"p":"P1","e":[]}                              ολόκληρη η P1
  {"i":1,"p":"P0","e":[{"x":2,"was":"και"}]}           η P0 χωρίς τη λέξη «και» στη θέση 2
  {"i":2,"p":"P0","e":[{"x":0,"was":"το","w":"της"}]}  η θέση 0 («το») γίνεται «της»
  {"i":3,"p":"P2","e":[{"+":1,"w":"ωρα"}]}             «ωρα» πριν τη θέση 1
  {"i":4,"p":"ABSTAIN"}

Οι θέσεις μετρούν από το 0 πάνω στην αρχική διαδρομή. Το e είναι υποχρεωτικό, έστω
κενό. Το πολύ 2 διορθώσεις. Καμία άλλη έξοδος.""",

"parse_rule": """Same as variant B, plus: every {'x':q,...} op MUST carry 'was' and
base[q] must equal it, else reject as 'misindexed' (not 'invalid') and fall back to
r12 — that class is counted separately because it is the failure the redundancy
exists to expose. Reject with 'toolong' if len(base) > 10 and e is non-empty.""",
}

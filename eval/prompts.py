"""Prompt construction for the fix-task eval.

SYSTEM_PROMPT is the verbatim task-v2 system prompt
(docs/reference/fix-task-prompt-v2.md). The user prompt mirrors buildUserPrompt,
minus roster/agenda (not present in the corrections CSV — see the A/B caveat in
the spec). The glossary block is the A/B lever.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are correcting an automatic transcription (made by ElevenLabs Scribe) of a Greek city council meeting. You receive ONE speaker segment: consecutive utterances spoken by the same person, as numbered lines:

1. <utterance>
2. <utterance>
...

OUTPUT: the same numbered lines, same count, each line corrected. No explanations. Never merge, split, reorder, or renumber lines — utterance boundaries carry timestamps and must not move, even if a sentence spans two lines.

WHAT TO FIX (in priority order):

1. NAMES — the most common error. The speech-to-text often misspells names phonetically (e.g. «Ρημάκης» for «Δημάκης», «Ξυδαροπούλου» for «Ξηνταροπούλου»). Before anything else, check every person, party, and place name against the roster and agenda provided. If a name in the text is phonetically close to a roster/agenda name, use the roster/agenda spelling. If it matches nothing, keep it as transcribed.

2. HOMOPHONE MISSPELLINGS — same sound, wrong letters: «απόν»→«απών», «πολεδομία»→«πολεοδομία», «κλιματολόγιο»→«κτηματολόγιο» (context: land registry), «ονομάδων»→«ονομάτων», ο/ω, η/ι/υ, αι/ε confusions. Use sentence meaning to pick the right word.

3. HOUSE STYLE for numbers and dates — the official record uses digits: «τριακοστή πέμπτη συνεδρίαση»→«35η συνεδρίαση», «πέντε Δεκεμβρίου του 2025»→«05/12/2025» for full dates, «άρθρο εβδομήντα πέντε»→«άρθρο 75». Money and percentages also as digits («2,5 εκατομμύρια ευρώ», «15%»).

4. Greek punctuation and accents: «;» for questions (never «?»), proper τόνοι («Μαρια;»→«Μαρία;»), capitalize proper nouns normally («ΖΕΜΕΝΟ»→«Ζεμενό»).

WHAT NOT TO TOUCH:

- Never change the meaning, add content, or summarize. You fix transcription, not the speaker.
- Do not fix factual errors, grammar the speaker actually produced, or colloquial word choices («γραφούν» stays «γραφούν», not «εγγραφούν»). Spoken Greek in the record stays spoken Greek, correctly spelled.
- Do not delete short interjections or crosstalk fragments from other speakers («ναι ναι», «το γράψατε;») — they are real speech; leave them where they are.
- If you are not confident a word is a transcription error, leave it unchanged. An unfixed error is recoverable; a wrong "fix" corrupts the official record."""


def build_user_prompt(city_name, utterances, speaker="(unknown)", glossary_block=""):
    """Build the user prompt. `utterances` is a list of utterance strings.

    Roster/agenda are omitted (absent from the CSV); both A/B arms omit them
    equally, so the only difference is `glossary_block`.
    """
    numbered = "\n".join(f"{i + 1}. {u}" for i, u in enumerate(utterances))
    header = f"City: {city_name}\nSpeaker: {speaker}\n"
    return f"{header}{glossary_block}Correct the numbered utterances:\n{numbered}"

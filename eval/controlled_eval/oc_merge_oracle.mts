// Oracle harness: runs the REAL pinned DiarizationManager over fixtures from stdin.
//
// The Python port in `oc_merge_port.py` is checked against this, so parity is
// differential rather than against hand-computed expectations.
//
// Run:  npx tsx oc_merge_oracle.ts <path-to-opencouncil-tasks> < fixtures.json
// In:   [{diarization:[{start,end,speaker}], start, end, words:[{start,end}]}, ...]
// Out:  [{speaker: string|null, drift: number|null}, ...]
//
// Speakers are passed through unmapped: the harness gives DiarizationManager a
// speakers list built from the distinct labels in the timeline, then inverts its
// numeric mapping so the output is comparable to the port's string labels.

import { readFileSync } from "node:fs";

// findBestSpeakerForUtterance logs a warning on the guess branch; stdout is the
// protocol here, so it must stay pure JSON.
console.log = () => {};

const repo = process.argv[2];
const { DiarizationManager } = await import(
    `${repo}/src/lib/DiarizationManager.ts`
);

const fixtures = JSON.parse(readFileSync(0, "utf8"));
const out = [];

for (const f of fixtures) {
    const labels: string[] = [];
    for (const d of f.diarization) if (!labels.includes(d.speaker)) labels.push(d.speaker);
    const speakers = labels.map((l) => ({ speaker: l }));
    const dm = new DiarizationManager(f.diarization, speakers as any);
    const utterance = {
        text: "",
        start: f.start,
        end: f.end,
        words: f.words.map((w: any) => ({ word: "", start: w.start, end: w.end })),
    };
    const r = dm.findBestSpeakerForUtterance(utterance as any);
    // buildSpeakerMaps numbers labels 1..n in the order they appear in `speakers`
    out.push(r === null ? { speaker: null, drift: null }
                        : { speaker: labels[r.speaker - 1], drift: r.drift });
}

process.stdout.write(JSON.stringify(out));

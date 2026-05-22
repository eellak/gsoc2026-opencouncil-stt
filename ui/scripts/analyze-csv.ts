#!/usr/bin/env bun
/**
 * Analyze the corrections CSV and produce a Markdown report.
 * Usage: bun run scripts/analyze-csv.ts [path/to/csv]
 * Output: data/csv-report.md  (plus stdout summary)
 */

import { parse } from 'csv-parse';
import { createReadStream, mkdirSync, writeFileSync } from 'fs';
import { resolve, join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { categorise, type RawCsvRow } from './lib/csv-clean';

const __dirname = dirname(fileURLToPath(import.meta.url));
const csvPath = resolve(process.argv[2] ?? join(__dirname, '..', '..', 'utterance-edits-may12-26.csv'));
const outDir = resolve(join(__dirname, '..', 'data'));
mkdirSync(outDir, { recursive: true });

console.log(`Analyzing: ${csvPath}`);

const categoryCount = new Map<string, number>();
const cleaningCount = new Map<string, number>();
const editorCount = new Map<string, number>();
const meetingCount = new Map<string, number>();
const audioCount = new Map<string, number>();
const examples = new Map<string, RawCsvRow[]>();
let total = 0;

const parser = parse({ columns: true, skip_empty_lines: true, relax_column_count: true, bom: true });

const startMs = Date.now();

parser.on('readable', () => {
	let row: RawCsvRow;
	while ((row = parser.read() as RawCsvRow) !== null) {
		total++;

		const { ingest_category, cleaning_applied } = categorise(row);

		categoryCount.set(ingest_category, (categoryCount.get(ingest_category) ?? 0) + 1);
		for (const tag of cleaning_applied.split(',').filter(Boolean)) {
			cleaningCount.set(tag, (cleaningCount.get(tag) ?? 0) + 1);
		}

		// Collect up to 5 examples per category
		const ex = examples.get(ingest_category) ?? [];
		if (ex.length < 5) ex.push(row);
		examples.set(ingest_category, ex);

		const ed = row.edited_by || 'unknown';
		editorCount.set(ed, (editorCount.get(ed) ?? 0) + 1);

		const mtg = `${row.meeting_name || '(άγνωστη)'}|${row.meeting_date || ''}`;
		meetingCount.set(mtg, (meetingCount.get(mtg) ?? 0) + 1);

		if (row.audio_url) {
			audioCount.set(row.audio_url, (audioCount.get(row.audio_url) ?? 0) + 1);
		}

		if (total % 50000 === 0) process.stdout.write(`\r  ${total.toLocaleString()} rows…`);
	}
});

parser.on('end', () => {
	const elapsed = ((Date.now() - startMs) / 1000).toFixed(1);
	console.log(`\r  Done: ${total.toLocaleString()} rows in ${elapsed}s`);

	// Build markdown report
	const lines: string[] = [];
	const h = (s: string) => { lines.push(s); lines.push(''); };
	const p = (s: string) => { lines.push(s); };

	h('# Ανάλυση CSV Corrections');
	p(`> Αρχείο: \`${csvPath}\``);
	p(`> Ημερομηνία ανάλυσης: ${new Date().toISOString().slice(0, 10)}`);
	p(`> Σύνολο γραμμών: **${total.toLocaleString()}**`);
	lines.push('');

	h('## Κατανομή ανά κατηγορία εισαγωγής');
	p('| Κατηγορία | Γραμμές | % |');
	p('|---|---|---|');
	const sortedCats = [...categoryCount.entries()].sort((a, b) => b[1] - a[1]);
	for (const [cat, n] of sortedCats) {
		p(`| \`${cat}\` | ${n.toLocaleString('el-GR')} | ${((n / total) * 100).toFixed(1)}% |`);
	}
	lines.push('');

	h('## Transformations που εφαρμόστηκαν');
	if (cleaningCount.size === 0) {
		p('Κανένα.');
	} else {
		p('| Tag | Γραμμές |');
		p('|---|---|');
		for (const [tag, n] of [...cleaningCount.entries()].sort((a, b) => b[1] - a[1])) {
			p(`| \`${tag}\` | ${n.toLocaleString('el-GR')} |`);
		}
	}
	lines.push('');

	h('## Ανά επεξεργαστή');
	p('| Επεξεργαστής | Γραμμές |');
	p('|---|---|');
	for (const [ed, n] of [...editorCount.entries()].sort((a, b) => b[1] - a[1])) {
		p(`| ${ed} | ${n.toLocaleString('el-GR')} |`);
	}
	lines.push('');

	h('## Ανά συνεδρίαση (top 20)');
	p('| Συνεδρίαση | Ημερομηνία | Γραμμές |');
	p('|---|---|---|');
	const topMeetings = [...meetingCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20);
	for (const [key, n] of topMeetings) {
		const [name, date] = key.split('|');
		p(`| ${name} | ${date} | ${n.toLocaleString('el-GR')} |`);
	}
	lines.push('');
	p(`_Σύνολο μοναδικών συνεδριάσεων: ${meetingCount.size}_`);
	lines.push('');

	h('## Ανά audio URL (top 20)');
	p('| URL | Γραμμές |');
	p('|---|---|');
	const topAudio = [...audioCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20);
	for (const [url, n] of topAudio) {
		p(`| \`${url}\` | ${n.toLocaleString('el-GR')} |`);
	}
	lines.push('');
	p(`_Σύνολο μοναδικών audio URLs: ${audioCount.size}_`);
	lines.push('');

	h('## Παραδείγματα ανά κατηγορία');
	for (const [cat, rows] of [...examples.entries()].sort()) {
		h(`### \`${cat}\``);
		for (const row of rows) {
			p(`**edit_id:** \`${row.edit_id}\``);
			p(`- **before:** ${row.before_text.slice(0, 120).replace(/\n/g, '↵')}`);
			p(`- **after:** ${row.after_text.slice(0, 120).replace(/\n/g, '↵')}`);
			p(`- ts: ${row.utterance_start}–${row.utterance_end} | editor: ${row.edited_by}`);
			lines.push('');
		}
	}

	const report = lines.join('\n');
	const outPath = join(outDir, 'csv-report.md');
	writeFileSync(outPath, report, 'utf8');
	console.log(`\nReport written to: ${outPath}`);

	// Quick stdout summary
	console.log('\nΣύνοψη κατηγοριών:');
	for (const [cat, n] of sortedCats) {
		console.log(`  ${cat.padEnd(24)} ${n.toLocaleString().padStart(8)} (${((n / total) * 100).toFixed(1)}%)`);
	}
});

parser.on('error', (err) => {
	console.error('CSV parse error:', err);
	process.exit(1);
});

createReadStream(csvPath).pipe(parser);

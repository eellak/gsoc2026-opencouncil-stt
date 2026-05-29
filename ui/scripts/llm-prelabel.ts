/**
 * Deterministic pre-labeler — assigns categories to obvious diffs so the
 * external-LLM loop doesn't waste a paste cycle on them.
 *
 * Rules (all conservative — when in doubt, do nothing):
 *   - identical after NFC normalize                    → `unusable`
 *   - identical after stripping combining marks (NFD)  → `accent_tonos`
 *   - identical after removing all whitespace          → `punctuation_capitalization`  (whitespace-only)
 *   - identical after removing all non-letter chars    → `punctuation_capitalization`
 *   - identical after lowercasing                       → `punctuation_capitalization`
 *   - identical after final-σ/ς swap                    → `final_sigma`
 *
 * Run with `bun scripts/llm-prelabel.ts --apply` to actually write events.
 * Without `--apply` it prints the counts but writes nothing.
 *
 * Writes events with `source = "auto-v2"` through the same SidecarStore
 * path so the orchestrator UI, /api/llm-batch endpoints, and stats panel
 * all pick the labels up transparently.
 */

import { promises as fs } from 'node:fs';
import { resolve } from 'node:path';
import { SidecarStore } from '../src/lib/server/state/sidecar';

const CACHE_DIR = process.env.REVIEW_CACHE_DIR ?? resolve(process.cwd(), '.cache');
const STATE_DIR = process.env.REVIEW_STATE_DIR ?? resolve(process.cwd(), '.state');

interface CachedGroup {
	utterance_id: string;
	initial_before_text: string;
	final_after_text: string;
}

type Rule = (b: string, a: string) => string[] | null;

const nfc = (s: string) => s.normalize('NFC');
const stripCombining = (s: string) => s.normalize('NFD').replace(/\p{M}/gu, '');
const stripWs = (s: string) => s.replace(/\s+/gu, '');
const lettersOnly = (s: string) => s.replace(/[^\p{L}\p{N}]+/gu, '');
const swapFinalSigma = (s: string) => s.replace(/ς/g, 'σ');

const RULES: Array<{ name: string; cats: string[]; fn: Rule }> = [
	{
		name: 'no-op (NFC equal)',
		cats: ['unusable'],
		fn: (b, a) => (nfc(b) === nfc(a) ? ['unusable'] : null)
	},
	{
		name: 'tonos-only',
		cats: ['accent_tonos'],
		fn: (b, a) => {
			if (nfc(b) === nfc(a)) return null;
			return stripCombining(nfc(b)) === stripCombining(nfc(a)) ? ['accent_tonos'] : null;
		}
	},
	{
		name: 'final-sigma-only',
		cats: ['final_sigma'],
		fn: (b, a) => {
			if (nfc(b) === nfc(a)) return null;
			return swapFinalSigma(nfc(b)) === swapFinalSigma(nfc(a)) ? ['final_sigma'] : null;
		}
	},
	{
		name: 'whitespace-only',
		cats: ['punctuation_capitalization'],
		fn: (b, a) => {
			if (nfc(b) === nfc(a)) return null;
			return stripWs(nfc(b)) === stripWs(nfc(a)) ? ['punctuation_capitalization'] : null;
		}
	},
	{
		name: 'case-only',
		cats: ['punctuation_capitalization'],
		fn: (b, a) => {
			if (nfc(b) === nfc(a)) return null;
			return nfc(b).toLowerCase() === nfc(a).toLowerCase() ? ['punctuation_capitalization'] : null;
		}
	},
	{
		name: 'punctuation-only',
		cats: ['punctuation_capitalization'],
		fn: (b, a) => {
			if (nfc(b) === nfc(a)) return null;
			return lettersOnly(nfc(b)).toLowerCase() === lettersOnly(nfc(a)).toLowerCase()
				? ['punctuation_capitalization']
				: null;
		}
	}
];

function classify(b: string, a: string): string[] | null {
	for (const r of RULES) {
		const out = r.fn(b, a);
		if (out) return out;
	}
	return null;
}

async function main() {
	const apply = process.argv.includes('--apply');
	console.log(`prelabel: cache=${CACHE_DIR} state=${STATE_DIR} apply=${apply}`);

	const groupsRaw = await fs.readFile(resolve(CACHE_DIR, 'groups.v1.json'), 'utf8');
	const groups = JSON.parse(groupsRaw) as CachedGroup[];

	const sidecar = await SidecarStore.load(STATE_DIR);
	const labels = sidecar.all();

	const counts: Record<string, number> = {};
	let touched = 0;

	for (const g of groups) {
		if (labels.has(g.utterance_id)) continue;
		const cats = classify(g.initial_before_text ?? '', g.final_after_text ?? '');
		if (!cats) continue;
		const key = cats.join('+');
		counts[key] = (counts[key] ?? 0) + 1;
		touched += 1;
		if (apply) {
			await sidecar.patchWithSource(g.utterance_id, { error_categories: cats }, 'auto-v2');
		}
	}
	await sidecar.flush();

	console.log(`prelabel: matched ${touched.toLocaleString()} / ${groups.length.toLocaleString()} groups`);
	for (const [k, v] of Object.entries(counts).sort(([, a], [, b]) => b - a)) {
		console.log(`  ${k}: ${v.toLocaleString()}`);
	}
	if (!apply) console.log('(dry run — pass --apply to write events)');
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});

/**
 * merge-state.ts — safely merge two review event logs (e.g. a local `.state`
 * into the server's `.state` that already has decisions) without losing either.
 *
 * Model: `review-events.jsonl` is an append-only log; the final label per
 * utterance is whatever the last (by id) event sets, and patches are partial
 * (one event may set a category, another the include_status). So the correct
 * merge is a chronological replay of BOTH logs:
 *
 *   1. parse both logs (tolerating a truncated final line, like the sidecar),
 *   2. drop exact-duplicate events (same ts + utterance + source + patch) so a
 *      log that was copied from the other doesn't double-count,
 *   3. sort by ts (ISO UTC sorts lexicographically), tie-break stably,
 *   4. renumber ids 1..N,
 *   5. write the merged log and remove the stale snapshot so the server
 *      rebuilds `review-labels.snapshot.json` from the merged log on next boot.
 *
 * Conflicts (same utterance decided in both) resolve to the latest ts — no
 * error, nothing deleted.
 *
 * Usage:
 *   bun ui/scripts/merge-state.ts --a <dir|jsonl> --b <dir|jsonl> --out <dir> \
 *       --stamp <suffix> [--apply]
 *
 * Dry-run by default: prints counts and writes nothing. Pass --apply to write.
 * `--stamp` is required for --apply (used for `*.bak.<stamp>` backups) since the
 * script has no clock of its own.
 */

import { promises as fs } from 'node:fs';
import { existsSync, statSync } from 'node:fs';
import { resolve, join } from 'node:path';

export interface ReviewEventLike {
	id: number;
	ts: string;
	utterance_id: string;
	source: { kind: string; username: string | null } | string;
	patch: Record<string, unknown>;
}

export interface MergeResult {
	merged: ReviewEventLike[];
	stats: {
		aCount: number;
		bCount: number;
		duplicatesDropped: number;
		mergedCount: number;
		conflicts: number; // utterances decided in BOTH inputs
	};
}

/** Parse a JSONL event log, tolerating a single truncated final line. */
export function parseEventLog(raw: string): ReviewEventLike[] {
	const out: ReviewEventLike[] = [];
	const lines = raw.split('\n');
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		if (!line) continue;
		try {
			out.push(JSON.parse(line) as ReviewEventLike);
		} catch (err) {
			const isLast = i === lines.length - 1 || lines.slice(i + 1).every((l) => !l);
			if (isLast) {
				console.warn(`[merge] dropping truncated final line at offset ${i}`);
				break;
			}
			throw new Error(`[merge] corrupt event at line ${i + 1}: ${(err as Error).message}`);
		}
	}
	return out;
}

function sourceKey(source: ReviewEventLike['source']): string {
	return typeof source === 'string' ? source : `${source.kind}:${source.username ?? ''}`;
}

/** Deterministic JSON: sort object keys recursively so two patches with the
 *  same content but different key order serialize identically. */
function stableStringify(value: unknown): string {
	if (value === null || typeof value !== 'object') return JSON.stringify(value);
	if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
	const keys = Object.keys(value as Record<string, unknown>).sort();
	const body = keys
		.map((k) => `${JSON.stringify(k)}:${stableStringify((value as Record<string, unknown>)[k])}`)
		.join(',');
	return `{${body}}`;
}

/** Content identity ignoring `id` — used to drop exact duplicates. */
function contentKey(ev: ReviewEventLike): string {
	return `${ev.ts}|${ev.utterance_id}|${sourceKey(ev.source)}|${stableStringify(ev.patch)}`;
}

/**
 * Merge two parsed logs. Pure — no IO — so it can be unit-tested.
 * `a` is treated as the lower-priority side on exact ts ties (b sorts after a,
 * so b wins a same-timestamp conflict), which keeps the order deterministic.
 */
export function mergeEventLogs(a: ReviewEventLike[], b: ReviewEventLike[]): MergeResult {
	const tagged = [
		...a.map((ev, i) => ({ ev, side: 0, origId: ev.id, ord: i })),
		...b.map((ev, i) => ({ ev, side: 1, origId: ev.id, ord: i }))
	];

	// Drop exact-duplicate events (keep first seen).
	const seen = new Set<string>();
	const deduped: typeof tagged = [];
	let duplicatesDropped = 0;
	for (const t of tagged) {
		const k = contentKey(t.ev);
		if (seen.has(k)) {
			duplicatesDropped++;
			continue;
		}
		seen.add(k);
		deduped.push(t);
	}

	// Stable chronological sort: ts, then side (a<b), then original id, then ord.
	deduped.sort((x, y) => {
		if (x.ev.ts !== y.ev.ts) return x.ev.ts < y.ev.ts ? -1 : 1;
		if (x.side !== y.side) return x.side - y.side;
		if (x.origId !== y.origId) return x.origId - y.origId;
		return x.ord - y.ord;
	});

	// Renumber ids 1..N.
	const merged = deduped.map((t, i) => ({ ...t.ev, id: i + 1 }));

	// Conflict count: utterances that received a decision (include_status) in
	// both inputs — informational only.
	const decidedInA = new Set(
		a.filter((e) => 'include_status' in e.patch).map((e) => e.utterance_id)
	);
	const decidedInB = new Set(
		b.filter((e) => 'include_status' in e.patch).map((e) => e.utterance_id)
	);
	let conflicts = 0;
	for (const u of decidedInA) if (decidedInB.has(u)) conflicts++;

	return {
		merged,
		stats: {
			aCount: a.length,
			bCount: b.length,
			duplicatesDropped,
			mergedCount: merged.length,
			conflicts
		}
	};
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function resolveEventsPath(input: string): string {
	const p = resolve(input);
	if (existsSync(p) && statSync(p).isDirectory()) return join(p, 'review-events.jsonl');
	return p; // treat as a file path
}

function parseArgs(argv: string[]): Record<string, string | boolean> {
	const out: Record<string, string | boolean> = {};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (!a.startsWith('--')) continue;
		const key = a.slice(2);
		const next = argv[i + 1];
		if (next && !next.startsWith('--')) {
			out[key] = next;
			i++;
		} else {
			out[key] = true;
		}
	}
	return out;
}

async function main() {
	const args = parseArgs(process.argv.slice(2));
	const aIn = args.a as string | undefined;
	const bIn = args.b as string | undefined;
	const outDir = args.out as string | undefined;
	const apply = args.apply === true;
	const stamp = args.stamp as string | undefined;

	if (!aIn || !bIn || !outDir) {
		console.error('Usage: bun ui/scripts/merge-state.ts --a <dir|jsonl> --b <dir|jsonl> --out <dir> --stamp <suffix> [--apply]');
		process.exit(2);
	}
	if (apply && !stamp) {
		console.error('--apply requires --stamp <suffix> (used for *.bak.<stamp> backups)');
		process.exit(2);
	}

	const aPath = resolveEventsPath(aIn);
	const bPath = resolveEventsPath(bIn);
	const aRaw = await fs.readFile(aPath, 'utf8').catch(() => '');
	const bRaw = await fs.readFile(bPath, 'utf8').catch(() => '');
	if (!aRaw) console.warn(`[merge] no events read from ${aPath}`);
	if (!bRaw) console.warn(`[merge] no events read from ${bPath}`);

	const { merged, stats } = mergeEventLogs(parseEventLog(aRaw), parseEventLog(bRaw));

	console.log('[merge] summary:');
	console.log(`  A (${aPath}): ${stats.aCount} events`);
	console.log(`  B (${bPath}): ${stats.bCount} events`);
	console.log(`  duplicates dropped: ${stats.duplicatesDropped}`);
	console.log(`  merged: ${stats.mergedCount} events (ids 1..${stats.mergedCount})`);
	console.log(`  utterances decided in BOTH (latest-ts wins): ${stats.conflicts}`);

	const outEvents = join(resolve(outDir), 'review-events.jsonl');
	const outSnapshot = join(resolve(outDir), 'review-labels.snapshot.json');

	if (!apply) {
		console.log('\n[merge] DRY RUN — nothing written. Re-run with --apply --stamp <suffix> to write.');
		console.log(`        would write: ${outEvents}`);
		console.log(`        would remove stale snapshot if present: ${outSnapshot}`);
		return;
	}

	await fs.mkdir(resolve(outDir), { recursive: true });

	// Back up anything we're about to overwrite/remove.
	if (existsSync(outEvents)) await fs.copyFile(outEvents, `${outEvents}.bak.${stamp}`);
	if (existsSync(outSnapshot)) await fs.copyFile(outSnapshot, `${outSnapshot}.bak.${stamp}`);

	const body = merged.map((e) => JSON.stringify(e)).join('\n') + (merged.length ? '\n' : '');
	const tmp = `${outEvents}.tmp`;
	await fs.writeFile(tmp, body);
	await fs.rename(tmp, outEvents);

	// Remove the stale snapshot so the sidecar rebuilds it from the merged log
	// on next boot (avoids hand-writing snapshot state that must match the
	// sidecar's canonicalization).
	if (existsSync(outSnapshot)) await fs.rm(outSnapshot);

	console.log(`\n[merge] wrote ${merged.length} events to ${outEvents}`);
	console.log('[merge] removed stale snapshot; server will rebuild it on boot.');
	console.log('[merge] after restarting the server, POST /api/stats/refresh to refresh stats.');
}

// Run as CLI only (not when imported by tests). Bun sets import.meta.main.
if ((import.meta as unknown as { main?: boolean }).main) {
	main().catch((err) => {
		console.error(err);
		process.exit(1);
	});
}

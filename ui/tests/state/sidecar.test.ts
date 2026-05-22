import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SidecarStore } from '../../src/lib/server/state/sidecar';

let dir: string;
beforeEach(async () => {
	dir = await mkdtemp(join(tmpdir(), 'sidecar-'));
});
afterEach(async () => {
	await rm(dir, { recursive: true, force: true });
});

describe('SidecarStore', () => {
	it('patch persists to JSONL and survives a reload', async () => {
		const s1 = await SidecarStore.load(dir);
		await s1.patch('u1', { include_status: 'include', error_categories: ['homophone'] });
		await s1.flush();

		const s2 = await SidecarStore.load(dir);
		const lbl = s2.get('u1');
		expect(lbl.include_status).toBe('include');
		expect(lbl.error_categories).toEqual(['homophone']);
		expect(lbl.human_updated_at).not.toBeNull();
	});

	it('omitted fields preserve previous values; null explicitly clears', async () => {
		const s = await SidecarStore.load(dir);
		await s.patch('u1', { include_status: 'include', error_categories: ['homophone'], reviewer_notes: 'a note' });
		await s.patch('u1', { reviewer_notes: null });
		await s.flush();
		const lbl = s.get('u1');
		expect(lbl.include_status).toBe('include'); // preserved
		expect(lbl.error_categories).toEqual(['homophone']); // preserved
		expect(lbl.reviewer_notes).toBeNull();      // cleared
	});

	it('accepts legacy error_category and stores as array', async () => {
		const s = await SidecarStore.load(dir);
		await s.patch('u1', { error_category: 'homophone' });
		await s.flush();
		expect(s.get('u1').error_categories).toEqual(['homophone']);
	});

	it('multi-category write replaces the whole array', async () => {
		const s = await SidecarStore.load(dir);
		await s.patch('u1', { error_categories: ['homophone', 'accent_tonos'] });
		await s.patch('u1', { error_categories: ['final_sigma'] });
		await s.flush();
		expect(s.get('u1').error_categories).toEqual(['final_sigma']);
	});

	it('rejects unknown taxonomy ids on write', async () => {
		const s = await SidecarStore.load(dir);
		await expect(s.patch('u1', { error_categories: ['totally_bogus'] })).rejects.toThrow(/unknown/i);
	});

	it('rejects invalid include_status synchronously', async () => {
		const s = await SidecarStore.load(dir);
		await expect(s.patch('u1', { include_status: 'bogus' as never })).rejects.toThrow(/include_status/);
	});

	it('rejects adjusted_start >= adjusted_end', async () => {
		const s = await SidecarStore.load(dir);
		await expect(s.patch('u1', { adjusted_start: 5, adjusted_end: 4 })).rejects.toThrow(/adjusted_start/);
	});

	it('tolerates a truncated final event line during rehydrate', async () => {
		const s = await SidecarStore.load(dir);
		await s.patch('u1', { include_status: 'include' });
		await s.flush();
		// Simulate a crash mid-write: append a partial JSON line.
		const eventsPath = join(dir, 'review-events.jsonl');
		await fs.appendFile(eventsPath, '{"id":99,"ts":"2026-');
		// Should NOT throw.
		const s2 = await SidecarStore.load(dir);
		expect(s2.get('u1').include_status).toBe('include');
	});

	it('recovers from a transient append failure without poisoning the write queue', async () => {
		const s = await SidecarStore.load(dir);
		// Land one good patch so the queue is "warm".
		await s.patch('u1', { include_status: 'include' });

		// Wrap fs.open so the next file handle's appendFile throws once.
		const eventsPath = join(dir, 'review-events.jsonl');
		const realOpen = fs.open.bind(fs);
		let failed = false;
		(fs as unknown as { open: typeof fs.open }).open = (async (
			path: Parameters<typeof fs.open>[0],
			flags?: Parameters<typeof fs.open>[1],
			mode?: Parameters<typeof fs.open>[2]
		) => {
			const handle = await realOpen(path, flags ?? 'r', mode);
			if (!failed && String(path) === eventsPath) {
				const realAppend = handle.appendFile.bind(handle);
				(handle as unknown as { appendFile: typeof handle.appendFile }).appendFile = (async () => {
					failed = true;
					throw new Error('boom');
				}) as typeof handle.appendFile;
				// Keep handle close working so we don't leak.
				void realAppend;
			}
			return handle;
		}) as typeof fs.open;

		try {
			await expect(s.patch('u2', { include_status: 'exclude' })).rejects.toThrow(/boom/);
			// Queue must not be permanently poisoned: the next patch should land.
			await s.patch('u3', { include_status: 'include' });
			await s.flush();
		} finally {
			(fs as unknown as { open: typeof fs.open }).open = realOpen;
		}

		const s2 = await SidecarStore.load(dir);
		expect(s2.get('u1').include_status).toBe('include');
		expect(s2.get('u3').include_status).toBe('include');
		// u2 must not have leaked into durable state.
		expect(s2.get('u2').include_status).toBe('unreviewed');
	});

	it('replays events past the snapshot last_event_id', async () => {
		const s1 = await SidecarStore.load(dir);
		for (let i = 0; i < 150; i++) {
			await s1.patch(`u${i}`, { include_status: 'include' });
		}
		await s1.flush();
		// Snapshot triggers every 100 events; another 50 stay only in the JSONL.
		const s2 = await SidecarStore.load(dir);
		expect(s2.get('u149').include_status).toBe('include');
		expect(s2.get('u0').include_status).toBe('include');
	});
});

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fingerprintFile } from '../../src/lib/server/cache/fingerprint';

let dir: string;
beforeEach(async () => {
	dir = await mkdtemp(join(tmpdir(), 'fp-'));
});
afterEach(async () => {
	await rm(dir, { recursive: true, force: true });
});

describe('fingerprintFile', () => {
	it('returns a stable hash for identical content', async () => {
		const a = join(dir, 'a.csv');
		await fs.writeFile(a, 'edit_id,utterance_id\n1,u1\n2,u2\n');
		const fp1 = await fingerprintFile(a);
		const fp2 = await fingerprintFile(a);
		expect(fp2.hash).toBe(fp1.hash);
	});

	it('detects content changes even when size is preserved', async () => {
		const a = join(dir, 'a.csv');
		await fs.writeFile(a, 'aaaa');
		const fp1 = await fingerprintFile(a);
		await fs.writeFile(a, 'bbbb'); // same size, different content
		const fp2 = await fingerprintFile(a);
		expect(fp2.hash).not.toBe(fp1.hash);
	});

	it('detects size changes', async () => {
		const a = join(dir, 'a.csv');
		await fs.writeFile(a, 'aaaa');
		const fp1 = await fingerprintFile(a);
		await fs.writeFile(a, 'aaaaaaaa');
		const fp2 = await fingerprintFile(a);
		expect(fp2.hash).not.toBe(fp1.hash);
		expect(fp2.size).toBe(8);
	});
});

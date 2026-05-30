/**
 * Cheap content fingerprint for the source CSV.
 *
 * mtime+size alone misses same-size rewrites and is hostage to filesystem
 * timestamp granularity. We hash the header + first/last 64 KB of the file
 * plus the byte size — enough to catch any realistic re-export of the CSV
 * without re-reading the whole 200 MB file.
 */

import { createHash } from 'node:crypto';
import { promises as fs } from 'node:fs';

const SAMPLE = 64 * 1024;

export async function fingerprintFile(path: string): Promise<{
	size: number;
	mtime_ms: number;
	hash: string;
}> {
	const stat = await fs.stat(path);
	const handle = await fs.open(path, 'r');
	try {
		const headTail = Buffer.alloc(Math.min(stat.size, SAMPLE * 2));
		const headLen = Math.min(stat.size, SAMPLE);
		await handle.read(headTail, 0, headLen, 0);
		if (stat.size > SAMPLE) {
			const tailStart = Math.max(SAMPLE, stat.size - SAMPLE);
			const tailLen = stat.size - tailStart;
			await handle.read(headTail, headLen, tailLen, tailStart);
		}
		const hash = createHash('sha256')
			.update(`${stat.size}\n`)
			.update(headTail)
			.digest('hex')
			.slice(0, 32);
		return { size: stat.size, mtime_ms: stat.mtimeMs, hash };
	} finally {
		await handle.close();
	}
}

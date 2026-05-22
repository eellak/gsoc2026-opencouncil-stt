import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Serves the GitHub-mirror url-map.json (data/audio-fix/url-map.json) to the
// client so direct audio fetches can route around broken originals without
// going through /api/audio. The map rarely changes — clients are expected to
// fetch this once on first use.

const HERE = dirname(fileURLToPath(import.meta.url));
const URL_MAP_PATH = resolve(HERE, '..', '..', '..', '..', '..', '..', 'data', 'audio-fix', 'url-map.json');

let cached: Record<string, string> | null = null;

export const GET: RequestHandler = async () => {
	if (cached === null) {
		try {
			const text = await readFile(URL_MAP_PATH, 'utf8');
			cached = JSON.parse(text) as Record<string, string>;
		} catch {
			cached = {};
		}
	}
	return json(cached, {
		headers: { 'cache-control': 'public, max-age=3600' }
	});
};

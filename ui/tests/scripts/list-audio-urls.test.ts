import { describe, expect, it } from 'vitest';
import { createClient } from '@libsql/client';
import { listAudioUrls } from '../../scripts/list-audio-urls';

describe('listAudioUrls', () => {
	it('returns distinct non-empty audio URLs sorted by URL', async () => {
		const client = createClient({ url: 'file::memory:' });
		await client.executeMultiple(`
			CREATE TABLE corrections (
				edit_id TEXT PRIMARY KEY,
				audio_url TEXT
			);
			INSERT INTO corrections (edit_id, audio_url) VALUES
				('1', 'https://data.opencouncil.gr/audio/b.mp3'),
				('2', ''),
				('3', NULL),
				('4', 'https://data.opencouncil.gr/audio/a.mp3'),
				('5', 'https://data.opencouncil.gr/audio/b.mp3');
		`);

		await expect(listAudioUrls(client)).resolves.toEqual([
			'https://data.opencouncil.gr/audio/a.mp3',
			'https://data.opencouncil.gr/audio/b.mp3'
		]);

		await client.close();
	});
});

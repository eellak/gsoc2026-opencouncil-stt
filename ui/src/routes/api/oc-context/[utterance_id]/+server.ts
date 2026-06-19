/**
 * Thin CORS bridge to the OpenCouncil per-utterance context endpoint.
 *
 * GET /api/oc-context/{utterance_id}?before=N&after=N
 *   → https://opencouncil.gr/api/utterance/{id}/context?before=N&after=N
 *
 * Replaces the old whole-meeting download (/api/oc-meeting): instead of pulling
 * the full ~600 KB transcript and slicing client-side, we ask upstream for just
 * the N utterances before/after the target. Mirrors the oc-meeting bridge:
 * abort on timeout, reject oversized bodies, stream the JSON through, and cache
 * aggressively since an utterance's neighbours are effectively immutable.
 *
 * Only successful, bounded responses are cacheable — errors throw and are not
 * cached.
 */

import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getRepo } from '$lib/server/repo';

const OPENCOUNCIL_BASE = 'https://opencouncil.gr/api/utterance';
const FETCH_TIMEOUT_MS = 8_000;
const MAX_NEIGHBOURS = 50; // upstream hard cap

function clampParam(raw: string | null, fallback: number): number {
	const n = raw == null ? fallback : Number(raw);
	if (!Number.isFinite(n)) return fallback;
	return Math.max(0, Math.min(MAX_NEIGHBOURS, Math.floor(n)));
}

export const GET: RequestHandler = async ({ params, url, fetch }) => {
	const id = params.utterance_id;
	if (!id) throw error(400, 'missing utterance_id');

	const before = clampParam(url.searchParams.get('before'), 10);
	const after = clampParam(url.searchParams.get('after'), 10);

	// Local-first: serve from the cached transcript index when this meeting is
	// indexed. Same shape as upstream. null OR any repo/DB error → fall through to
	// the live proxy, so a local-index problem can never break context entirely.
	try {
		const local = (await getRepo()).getContext(id, before, after);
		if (local) {
			return json(local, {
				headers: {
					'cache-control': 'public, max-age=3600, s-maxage=3600',
					'x-oc-source': 'local-index'
				}
			});
		}
	} catch (err) {
		console.warn('[oc-context] local lookup failed, falling back to upstream', err);
	}

	const upstream = `${OPENCOUNCIL_BASE}/${encodeURIComponent(id)}/context?before=${before}&after=${after}`;

	const ctrl = new AbortController();
	const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
	let resp: Response;
	try {
		resp = await fetch(upstream, { signal: ctrl.signal });
	} catch (err) {
		clearTimeout(timer);
		console.warn('[oc-context] fetch failed', upstream, err);
		throw error(502, 'upstream fetch failed');
	}
	clearTimeout(timer);

	if (!resp.ok) {
		console.warn('[oc-context] upstream', resp.status, upstream);
		// Pass through the auth/not-found statuses the client uses to classify a
		// "private" meeting (auto-skip) vs a transient failure (show + retry).
		// Everything else — and the network/timeout catch above — stays 502 so
		// infra hiccups never get mistaken for private data.
		const passthrough =
			resp.status === 401 || resp.status === 403 || resp.status === 404;
		throw error(passthrough ? resp.status : 502, `upstream ${resp.status}`);
	}

	// Context windows are small (≤100 short utterances); 2 MB is a generous
	// ceiling so a misbehaving upstream can't blow up memory on the VM.
	const cl = resp.headers.get('content-length');
	if (cl) {
		const n = Number.parseInt(cl, 10);
		if (Number.isFinite(n) && n > 2 * 1024 * 1024) {
			throw error(502, 'upstream body too large');
		}
	}

	return new Response(resp.body, {
		headers: {
			'content-type': 'application/json; charset=utf-8',
			'cache-control': 'public, max-age=3600, s-maxage=3600',
			'x-oc-source': 'opencouncil.gr'
		}
	});
};

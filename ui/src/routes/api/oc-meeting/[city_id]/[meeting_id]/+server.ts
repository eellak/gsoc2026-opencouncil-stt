/**
 * Thin CORS bridge: forwards the upstream OpenCouncil meeting JSON to the
 * browser. All slicing, speaker resolution, and prefetch logic happens
 * client-side — this endpoint exists only because opencouncil.gr does not
 * return `Access-Control-Allow-Origin` and a direct browser fetch would be
 * blocked.
 *
 * GET /api/oc-meeting/{city_id}/{meeting_id}
 *
 * Response is cached aggressively (1h) because the underlying transcript is
 * effectively immutable per meeting.
 */

import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const OPENCOUNCIL_BASE = 'https://opencouncil.gr/api/cities';
const FETCH_TIMEOUT_MS = 8_000;

export const GET: RequestHandler = async ({ params, fetch }) => {
	const cityId = params.city_id;
	const meetingId = params.meeting_id;
	if (!cityId || !meetingId) throw error(400, 'missing city_id or meeting_id');

	const url = `${OPENCOUNCIL_BASE}/${encodeURIComponent(cityId)}/meetings/${encodeURIComponent(meetingId)}`;

	const ctrl = new AbortController();
	const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
	let upstream: Response;
	try {
		upstream = await fetch(url, { signal: ctrl.signal });
	} catch (err) {
		clearTimeout(timer);
		console.warn('[oc-meeting] fetch failed', url, err);
		throw error(502, 'upstream fetch failed');
	}
	clearTimeout(timer);

	if (!upstream.ok) {
		console.warn('[oc-meeting] upstream', upstream.status, url);
		throw error(upstream.status === 404 ? 404 : 502, `upstream ${upstream.status}`);
	}

	// Reject obviously oversized bodies up front. The transcripts we serve are
	// typically <2MB; 10MB is a hard ceiling so a misbehaving upstream cannot
	// blow up memory on a 1GB VM.
	const cl = upstream.headers.get('content-length');
	if (cl) {
		const n = Number.parseInt(cl, 10);
		if (Number.isFinite(n) && n > 10 * 1024 * 1024) {
			throw error(502, 'upstream body too large');
		}
	}

	// Stream pass-through — the bridge does no parsing, so there is no reason
	// to buffer the entire body in memory before responding.
	return new Response(upstream.body, {
		headers: {
			'content-type': 'application/json; charset=utf-8',
			'cache-control': 'public, max-age=3600, s-maxage=3600',
			// Long browser-side cache: same meeting transcript rarely changes.
			'x-oc-source': 'opencouncil.gr'
		}
	});
};

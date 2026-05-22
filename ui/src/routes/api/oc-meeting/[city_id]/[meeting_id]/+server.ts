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

	// `text()` can reject if the upstream tears the body down mid-read
	// (TCP reset, transient TLS hiccup). Catch it explicitly so the client
	// gets a clean 502 instead of an unhandled rejection turning into 500.
	let body: string;
	try {
		body = await upstream.text();
	} catch (err) {
		console.warn('[oc-meeting] body read failed', url, err);
		throw error(502, 'upstream body read failed');
	}
	return new Response(body, {
		headers: {
			'content-type': 'application/json; charset=utf-8',
			'cache-control': 'public, max-age=3600, s-maxage=3600',
			// Long browser-side cache: same meeting transcript rarely changes.
			'x-oc-source': 'opencouncil.gr'
		}
	});
};

<script lang="ts">
	import { goto, afterNavigate } from '$app/navigation';
	import { untrack } from 'svelte';
	import Diff from '$lib/components/Diff.svelte';
	import StatusButtons from '$lib/components/StatusButtons.svelte';
	import TaxonomySelect from '$lib/components/TaxonomySelect.svelte';
	import CategoryPalette from '$lib/components/CategoryPalette.svelte';
	import MeetingContextPanel from '$lib/components/MeetingContextPanel.svelte';
	import type { Group, GroupPatchBody } from '$lib/domain/groups';
	import type { MeetingContext } from '$lib/domain/meeting-context';
	import { TAXONOMY, normalizeTaxonomyId, type TaxonomyId } from '$lib/shared/taxonomy';
	import { t, getLang } from '$lib/i18n.svelte';
	import * as queue from '$lib/client/group-queue.svelte';
	import type { IncludeStatus } from '$lib/domain/types';
	import { resolveAudioUrls } from '$lib/client/audio-source';
	import { audioPool } from '$lib/client/audio-pool.svelte';
	import * as meetingCtx from '$lib/client/meeting-context.svelte';
	import { reviewHref } from '$lib/shared/urls';
	import type { PageData } from './$types';
	import { userStore } from '$lib/client/user-store.svelte';
	import { playbackPrefs } from '$lib/client/playback-prefs.svelte';
	import UserPickerModal from '$lib/components/UserPickerModal.svelte';
	import ShortcutsModal from '$lib/components/ShortcutsModal.svelte';
	import SettingsModal from '$lib/components/SettingsModal.svelte';
	import MobileSwipeCard from '$lib/components/MobileSwipeCard.svelte';
	import { reviewPrefs } from '$lib/client/review-prefs.svelte';
	import {
		autoSkip,
		rememberDirection,
		lastNavDirection,
		noteSuccessfulLoad,
		allowSkip,
		resumeAutoSkip
	} from '$lib/client/auto-skip.svelte';

	const DIGIT_SHORTCUTS: ReadonlyMap<string, TaxonomyId> = new Map(
		TAXONOMY.filter((c) => c.shortcut).map((c) => [c.shortcut!, c.id as TaxonomyId])
	);

	const { data }: { data: PageData } = $props();

	// Active queue filter (status / category / errorCategory), resolved by the
	// load function. Drives the filter badge and keeps the filter in nav URLs.
	const filter = $derived(data.filter);
	// Status labels are i18n keys (translate them); category ids are raw.
	const STATUS_KEYS = new Set<IncludeStatus>(['include', 'exclude', 'uncertain', 'unreviewed']);
	const filterLabel = $derived.by(() => {
		const f = filter;
		if (!f) return '';
		return STATUS_KEYS.has(f.label as IncludeStatus) ? t(f.label as IncludeStatus) : f.label;
	});

	// In filter mode, (re)populate the client queue from /api/review/ids so j/k
	// and autoplay walk only the matching items. The load function does this on
	// navigation; this effect keeps it correct if the filter changes in place.
	$effect(() => {
		const f = filter;
		if (!f) return;
		let cancelled = false;
		(async () => {
			try {
				const resp = await queue.fetchFilterIds(f.query);
				if (cancelled) return;
				queue.setFilterOrder(resp.filter, resp.ids, resp.revision, resp.cache_hash);
			} catch (e) {
				console.warn('[review] fetchFilterIds failed', e);
			}
		})();
		return () => {
			cancelled = true;
		};
	});

	const item = $derived<Group>(queue.get(data.item.utterance_id) ?? data.item);
	const prev = $derived(queue.prevOf(data.item.utterance_id));
	const next = $derived(queue.nextOf(data.item.utterance_id));
	const prevId = $derived(queue.prevIdOf(data.item.utterance_id));
	const nextId = $derived(queue.nextIdOf(data.item.utterance_id));

	let showFullChain = $state(false);

	const beforeText = $derived(showFullChain
		? item.edits.map((e, i) => `[edit ${i + 1} by ${e.edited_by ?? '—'}]\n${e.before_text}`).join('\n\n')
		: item.initial_before_text);
	const afterText = $derived(showFullChain
		? item.edits.map((e, i) => `[edit ${i + 1} by ${e.edited_by ?? '—'}]\n${e.after_text}`).join('\n\n')
		: item.final_after_text);

	let saveStatus = $state<'idle' | 'saving' | 'saved' | 'error'>('idle');
	let saveTimer: ReturnType<typeof setTimeout>;
	let shareCopied = $state(false);
	let shareCopiedTimer: ReturnType<typeof setTimeout>;

	async function patch(updates: GroupPatchBody) {
		clearTimeout(saveTimer);
		saveStatus = 'saving';
		queue.patchLocalLabel(item.utterance_id, updates);
		const body = userStore.value ? { ...updates, username: userStore.value } : updates;
		try {
			const res = await fetch(`/api/review/group/${encodeURIComponent(item.utterance_id)}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body)
			});
			if (!res.ok) throw new Error(await res.text());
			saveStatus = 'saved';
		} catch {
			saveStatus = 'error';
		}
		saveTimer = setTimeout(() => (saveStatus = 'idle'), 2000);
	}

	function patchStatus(status: IncludeStatus, opts: { advance?: boolean } = {}) {
		patch({ include_status: status });
		// Only auto-advance for terminal decisions, not 'unreviewed' reverts.
		if (status === 'unreviewed') return;
		const shouldAdvance = opts.advance ?? reviewPrefs.autoAdvance;
		if (shouldAdvance && canNext) {
			// Tiny delay so the status flash is visible before navigation.
			setTimeout(() => goNext(), 120);
		}
	}

	function swipeInclude() { patchStatus('include', { advance: true }); }
	function swipeExclude() { patchStatus('exclude', { advance: true }); }

	let paletteOpen = $state(false);
	let shortcutsOpen = $state(false);
	let settingsOpen = $state(false);
	let showUserModal = $state(userStore.value === '');
	// The visible <audio> element is owned by the pool — pool.setActive moves
	// the matching pool element into our `audioSlot` div. We read it back via
	// the pool's reactive state to call play/pause/seek and attach listeners.
	// See `audio-pool.svelte.ts` for the element-swap rationale.
	let audioSlot: HTMLDivElement | null = $state(null);
	const audioEl = $derived(audioPool.state.activeEl);
	let audioReady = $state(false);
	let audioReadyFlash = $state(false);
	let audioAutoplayReady = $state(false);
	let autoplayFallbackTimer: ReturnType<typeof setTimeout> | null = null;
	let isPlaying = $state(false);
	function clearAutoplayFallback() {
		if (autoplayFallbackTimer) { clearTimeout(autoplayFallbackTimer); autoplayFallbackTimer = null; }
	}
	function onAudioCanPlay() {
		audioReady = true;
		// Fallback: if canplaythrough never fires (slow/unstable network, Safari
		// quirks), unblock autoplay 1.5s after canplay so we don't stall forever.
		if (!audioAutoplayReady && !autoplayFallbackTimer) {
			autoplayFallbackTimer = setTimeout(() => {
				audioAutoplayReady = true;
				autoplayFallbackTimer = null;
			}, 1500);
		}
	}
	function onAudioCanPlayThrough() {
		audioReady = true;
		audioAutoplayReady = true;
		clearAutoplayFallback();
	}
	function onAudioWaiting() { audioReady = false; audioReadyFlash = false; audioAutoplayReady = false; clearAutoplayFallback(); }
	function onAudioLoadStart() { audioReady = false; audioReadyFlash = false; audioAutoplayReady = false; isPlaying = false; clearAutoplayFallback(); }
	function onAudioPlay() {
		isPlaying = true;
		// Flash sweeps once per play-start regardless of how it started
		// (manual click vs autoplay). The :playing throb takes over after.
		audioReadyFlash = true;
	}
	function onAudioPause() { isPlaying = false; }
	function onAudioEnded() { isPlaying = false; }
	const audioSources = $derived(resolveAudioUrls(item.audio_url));
	const primaryAudioUrl = $derived(audioSources.direct);
	let usingFallback = $state(false);
	const activeAudioUrl = $derived(usingFallback ? audioSources.proxied : primaryAudioUrl);

	function onAudioError() {
		if (!usingFallback && audioSources.proxied !== primaryAudioUrl) {
			console.warn('[review] direct audio failed, falling back to proxy:', item.audio_url);
			usingFallback = true;
		}
	}

	// Identity-stable primitives so that label patches (which spawn a new
	// `item` object reference via SvelteMap) don't refire the prefetch
	// effects below. Each derived only emits when its string value really
	// changes — i.e. when the user navigates to a new utterance.
	const currentId = $derived(item.utterance_id);

	// Wire the pool to our visible slot. The pool moves elements in/out as
	// the active id changes. Cleanup on unmount so a navigation away from
	// /review doesn't strand the audio element in a destroyed slot.
	$effect(() => {
		audioPool.attachVisibleHost(audioSlot);
		return () => audioPool.attachVisibleHost(null);
	});

	// Drive pool.setActive on id (or URL — fallback flip) change. The URL
	// is part of the trigger because flipping `usingFallback` after a direct
	// load fails needs to swap src on the active element. Everything else
	// (neighbours snapshot, fallback reset) runs untracked so a label patch
	// or timestamp edit on a neighbour doesn't churn the pool.
	$effect(() => {
		const id = currentId;
		const url = activeAudioUrl;
		untrack(() => {
			if (!usingFallback) usingFallback = false; // no-op; reset happens when id changes below
			const audioNeighbours = queue.neighborsAround(id, 10).map((g) => ({
				utterance_id: g.utterance_id,
				url: resolveAudioUrls(g.audio_url).direct,
				start: g.label.adjusted_start ?? g.start
			}));
			audioPool.setActive(
				{ utterance_id: id, url, start: regionStart },
				audioNeighbours
			);
		});
	});

	// Reset the fallback flag when the utterance id changes — each clip
	// starts on the direct CDN URL; only failure flips it to proxied.
	$effect(() => {
		currentId; // dep
		untrack(() => { usingFallback = false; });
	});

	// Remember the last item viewed per seed so re-entering the same seed can
	// resume here. Seeded mode only — filter queues don't carry a seed.
	$effect(() => {
		if (filter) return;
		const seed = data.seed;
		const id = currentId;
		if (typeof localStorage === 'undefined') return;
		try { localStorage.setItem(`oc:resume:${seed}`, id); } catch { /* quota — fine */ }
	});

	// Listener wiring: when the pool hands us a new active <audio>, attach
	// our state-syncing handlers and seed audioReady/isPlaying from the
	// element's current state (it may already have buffered + canplayed
	// while it sat in the pool — those events fired before we listened).
	$effect(() => {
		const el = audioEl;
		if (!el) return;
		// Sync from element's actual condition.
		audioReady = el.readyState >= 3;
		audioAutoplayReady = el.readyState >= 4;
		audioReadyFlash = false;
		isPlaying = !el.paused;
		clearAutoplayFallback();
		// If element already canplay but not canplaythrough, arm the fallback
		// so autoplay doesn't wait forever for an event that already fired.
		if (audioReady && !audioAutoplayReady) {
			autoplayFallbackTimer = setTimeout(() => {
				audioAutoplayReady = true;
				autoplayFallbackTimer = null;
			}, 1500);
		}
		// On (re-)activation, snap playhead to the segment start if we're
		// not already inside the region. Element survives across nav, so
		// `currentTime` might be wherever the previous viewing left it.
		if (el.paused && el.readyState >= 1) {
			if (el.currentTime < regionStart || el.currentTime >= regionEnd) {
				try { el.currentTime = regionStart; } catch { /* fine */ }
			}
		}
		el.addEventListener('loadedmetadata', onAudioMeta);
		el.addEventListener('timeupdate', onAudioTimeUpdate);
		el.addEventListener('error', onAudioError);
		el.addEventListener('canplay', onAudioCanPlay);
		el.addEventListener('canplaythrough', onAudioCanPlayThrough);
		el.addEventListener('waiting', onAudioWaiting);
		el.addEventListener('loadstart', onAudioLoadStart);
		el.addEventListener('play', onAudioPlay);
		el.addEventListener('pause', onAudioPause);
		el.addEventListener('ended', onAudioEnded);
		return () => {
			el.removeEventListener('loadedmetadata', onAudioMeta);
			el.removeEventListener('timeupdate', onAudioTimeUpdate);
			el.removeEventListener('error', onAudioError);
			el.removeEventListener('canplay', onAudioCanPlay);
			el.removeEventListener('canplaythrough', onAudioCanPlayThrough);
			el.removeEventListener('waiting', onAudioWaiting);
			el.removeEventListener('loadstart', onAudioLoadStart);
			el.removeEventListener('play', onAudioPlay);
			el.removeEventListener('pause', onAudioPause);
			el.removeEventListener('ended', onAudioEnded);
		};
	});

	let utteranceAnchor: HTMLElement | null = $state(null);
	let topBarEl: HTMLElement | null = $state(null);
	// After every navigation (including j/k and swipe-commit), scroll so the
	// utterance edit sits near the top of the viewport instead of getting
	// buried under the (slowly-loading) transcript panel above. Offset is
	// measured from the actual sticky-header so wrapped rows on mobile
	// don't hide the card behind it.
	afterNavigate(() => {
		if (typeof window === 'undefined') return;
		// Only force-scroll in mobile mode; on desktop the user keeps
		// their own scroll position across navigations.
		if (!reviewPrefs.mobileMode) return;
		requestAnimationFrame(() => {
			if (!utteranceAnchor) return;
			const headerH = topBarEl?.getBoundingClientRect().height ?? 70;
			const rect = utteranceAnchor.getBoundingClientRect();
			const offset = headerH + 12;
			const target = Math.max(0, window.scrollY + rect.top - offset);
			// Instant jump, not smooth: the per-navigation scroll was distracting
			// on mobile (everything visibly sliding on each "next"). Snap so the
			// eye doesn't track it. Skip entirely when we're already close to
			// avoid a pointless 1-2px nudge.
			if (Math.abs(window.scrollY - target) < 4) return;
			window.scrollTo({ top: target, behavior: 'auto' });
		});
	});

	// Keep scroll-margin-top in sync with the sticky header so anchor jumps
	// (e.g. browser back/forward) also clear it.
	$effect(() => {
		if (typeof document === 'undefined') return;
		const update = () => {
			const h = topBarEl?.getBoundingClientRect().height ?? 70;
			document.documentElement.style.setProperty('--top-bar-h', `${Math.ceil(h)}px`);
		};
		update();
		const ro = topBarEl ? new ResizeObserver(update) : null;
		if (topBarEl && ro) ro.observe(topBarEl);
		window.addEventListener('resize', update);
		return () => {
			ro?.disconnect();
			window.removeEventListener('resize', update);
		};
	});

	// Toggle a body class so global CSS can react to mobile-mode (e.g. hide
	// keyboard shortcut hints that aren't useful on touch).
	$effect(() => {
		if (typeof document === 'undefined') return;
		const on = reviewPrefs.mobileMode;
		document.body.classList.toggle('mobile-mode', on);
		return () => document.body.classList.remove('mobile-mode');
	});

	// Autoplay: wait for canplaythrough (or 1.5s fallback after canplay) to
	// avoid the "start → stall → resume" stutter that comes from playing on
	// canplay alone. See onAudioCanPlay/onAudioCanPlayThrough.
	$effect(() => {
		if (audioAutoplayReady && playbackPrefs.autoplay && audioEl && audioEl.paused) {
			if (audioEl.currentTime < regionStart || audioEl.currentTime >= regionEnd) {
				try { audioEl.currentTime = regionStart; } catch { /* fine */ }
			}
			void audioEl.play();
		}
	});

	// Transcript prefetch: same sliding window, separate effect so its
	// dependencies stay independent of the audio one. Triggers on id change
	// only; the neighbour scan runs untracked.
	$effect(() => {
		const id = currentId;
		untrack(() => {
			// Prefetch context for upcoming utterances so the panel is warm on
			// arrival. Each context payload is a few KB (per-utterance endpoint),
			// so prefetching individual neighbours is cheap.
			for (const g of queue.neighborsAround(id, 10)) {
				meetingCtx.prefetch(g.utterance_id);
			}
		});
	});

	// Surrounding-utterance context, fetched client-side via
	// `meeting-context.svelte`, which calls the per-utterance OpenCouncil
	// endpoint through the /api/oc-context/{id} CORS bridge (before=prevRadius,
	// after=nextRadius). Only the needed neighbours come over the wire — no
	// whole-meeting download. Speaker names aren't available from this endpoint.
	let contextState = $state<'loading' | 'ready' | 'error' | 'empty'>('loading');
	let contextData = $state<MeetingContext | null>(null);
	let prevRadius = $state(5);
	let nextRadius = $state(5);

	// Reset context radii when navigating to a new utterance
	$effect(() => {
		currentId;
		prevRadius = 5;
		nextRadius = 5;
	});

	$effect(() => {
		const id = currentId;
		const pr = prevRadius;
		const nr = nextRadius;
		let cancelled = false;

		// Avoid the "loading…" flicker when this exact context window is already
		// resolved in the cache. We still call getContext (the .then resolves
		// synchronously from cache) and keep the old contextData visible until
		// the new data lands.
		const wasCached = meetingCtx.hasContext(id, pr, nr);
		if (!wasCached) {
			contextState = 'loading';
			contextData = null;
		}

		meetingCtx
			.getContext(id, pr, nr)
			.then((data) => {
				if (cancelled) return;
				contextData = data;
				if (data.error) {
					// Unavailable (401/403/404 — private meeting OR a removed
					// utterance) → auto-skip just this utterance, showing 'loading'
					// (not a flashed error) while we navigate; fall back to 'error'
					// if the skip can't fire (cap/queue edge). Transient errors
					// (5xx/timeout/network) stay on screen for the reviewer to retry.
					if (data.error_kind === 'private') {
						contextState = autoSkipPrivate() ? 'loading' : 'error';
					} else {
						contextState = 'error';
					}
				} else if (data.prev.length === 0 && data.next.length === 0) {
					contextState = 'empty';
					noteSuccessfulLoad();
				} else {
					contextState = 'ready';
					noteSuccessfulLoad();
				}
			})
			.catch(() => {
				// Network/abort — transient, never auto-skip.
				if (!cancelled) contextState = 'error';
			});
		return () => {
			cancelled = true;
		};
	});

	// Advance past a private meeting in the last navigation direction. Terminal
	// rule: no candidate that way (queue edge) → stop, don't wrap. Returns true
	// when it actually navigates (cap not hit and a candidate exists).
	function autoSkipPrivate(): boolean {
		// Skipping a 404/private item is a low-level "step off this broken item"
		// move and must always make progress. Prefer the skip-aware target (so we
		// land on the next unreviewed item), but fall back to the immediate
		// neighbour — otherwise an all-classified forward window would leave the
		// skip-aware href null and strand the reviewer on the unavailable item.
		const targetId =
			lastNavDirection() === 'prev'
				? (prevTargetId ?? prevId ?? null)
				: (nextTargetId ?? nextId ?? null);
		const href = navHref(targetId);
		if (!href) return false;
		if (!allowSkip()) return false;
		goto(href);
		return true;
	}

	// Speaker for the current utterance. The per-utterance context endpoint
	// doesn't return the anchor or any speaker names, so this is null and the
	// Diff header shows its neutral placeholder. Kept as a derived so it lights
	// up automatically if a name source is wired in later.
	const currentSpeakerName = $derived(
		contextData?.current?.speaker_name ?? contextData?.current?.speaker_label ?? null
	);

	const regionStart = $derived(item.label.adjusted_start ?? item.start);
	const regionEnd = $derived(item.label.adjusted_end ?? item.end);

	function togglePlay() {
		if (!audioEl || !audioReady) return;
		if (audioEl.paused) {
			if (audioEl.currentTime < regionStart || audioEl.currentTime >= regionEnd) {
				audioEl.currentTime = regionStart;
			}
			void audioEl.play();
		} else {
			audioEl.pause();
		}
	}

	function onAudioMeta() {
		if (audioEl) audioEl.currentTime = regionStart;
	}

	let loopTimer: ReturnType<typeof setTimeout>;
	function onAudioTimeUpdate() {
		if (!audioEl) return;
		if (!audioEl.paused && audioEl.currentTime >= regionEnd) {
			audioEl.pause();
			audioEl.currentTime = regionStart;
			if (playbackPrefs.loop) {
				clearTimeout(loopTimer);
				loopTimer = setTimeout(() => { void audioEl?.play(); }, playbackPrefs.loopGapMs);
			}
		}
	}

	function commitTimestamps(start: number, end: number) {
		if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end) return;
		patch({ adjusted_start: start, adjusted_end: end });
	}

	// Manual fine-sync of the segment boundaries. Each press moves one
	// boundary by the configured step (the "delay", in ms). Persists through
	// the same adjusted_start/adjusted_end fields commitTimestamps writes.
	const NUDGE_PREVIEW_SEC = 0.8;
	function roundMs(sec: number): number {
		return Math.round(sec * 1000) / 1000;
	}
	const stepSec = $derived(playbackPrefs.nudgeStepMs / 1000);
	// Disable controls that would invert/collapse the region or push start
	// below zero, so a precise click never silently no-ops.
	const canStartBack = $derived(regionStart > 0);
	const canStartFwd = $derived(roundMs(regionStart + stepSec) < regionEnd);
	const canEndBack = $derived(roundMs(regionEnd - stepSec) > regionStart);
	const canEndFwd = $derived(true);
	// Whole-utterance move can always go forward (clamped to duration); going
	// back is only possible while the start is above zero.
	const canMoveBack = $derived(regionStart > 0);
	const canMoveFwd = $derived(true);

	function nudge(which: 'start' | 'end', dir: -1 | 1) {
		const deltaSec = (dir * playbackPrefs.nudgeStepMs) / 1000;
		let s = regionStart;
		let e = regionEnd;
		if (which === 'start') {
			s = roundMs(Math.max(0, regionStart + deltaSec));
		} else {
			e = roundMs(regionEnd + deltaSec);
			const dur = audioEl?.duration;
			if (typeof dur === 'number' && Number.isFinite(dur)) e = Math.min(roundMs(dur), e);
		}
		if (s >= e) return; // boundary would invert — handled visually by disabled buttons
		commitTimestamps(s, e);
		// Audible/visual feedback: drop the playhead onto the boundary being
		// tuned (a hair before the new end so its tail is what plays back).
		const el = audioEl;
		if (el) {
			const target = which === 'start' ? s : Math.max(s, e - NUDGE_PREVIEW_SEC);
			try { el.currentTime = target; } catch { /* element may be mid-swap */ }
		}
	}

	// Shift the whole utterance window (start AND end) by one step, keeping its
	// duration fixed. The applied delta is clamped so start never goes below 0
	// and end never passes the audio duration — the window slides, never squashes.
	function shiftWhole(dir: -1 | 1) {
		let delta = (dir * playbackPrefs.nudgeStepMs) / 1000;
		if (regionStart + delta < 0) delta = -regionStart;
		const dur = audioEl?.duration;
		if (typeof dur === 'number' && Number.isFinite(dur) && regionEnd + delta > dur) {
			delta = dur - regionEnd;
		}
		if (delta === 0) return;
		const s = roundMs(regionStart + delta);
		const e = roundMs(regionEnd + delta);
		commitTimestamps(s, e);
		const el = audioEl;
		if (el) {
			try { el.currentTime = s; } catch { /* element may be mid-swap */ }
		}
	}

	function navHref(targetId: string | undefined | null): string | null {
		if (!targetId) return null;
		if (filter) {
			// Preserve the filter in the URL so the next page stays in filter mode.
			return `/review/${encodeURIComponent(targetId)}?${filter.query}`;
		}
		return reviewHref({ utterance_id: targetId, seed: data.seed });
	}
	// Skip-aware navigation: when the pref is on (seeded mode only), next/prev
	// jump over already-classified items so re-entering with the same seed
	// resumes past finished work instead of re-showing it.
	const skipNav = $derived(reviewPrefs.skipClassified && !filter);
	const nextTargetId = $derived(
		skipNav
			? (queue.nextUnreviewedIdLoaded(data.item.utterance_id) ?? null)
			: (next?.utterance_id ?? nextId ?? null)
	);
	const prevTargetId = $derived(
		skipNav
			? (queue.prevUnreviewedId(data.item.utterance_id) ?? null)
			: (prev?.utterance_id ?? prevId ?? null)
	);
	const prevHref = $derived(navHref(prevTargetId));
	const nextHref = $derived(navHref(nextTargetId));
	// The real next target can sit past the warm window, so allow "next" while
	// more pages exist even if the loaded target is null — goNext pages to it.
	const canNext = $derived(
		skipNav ? nextTargetId !== null || queue.hasMoreSeeded() : nextTargetId !== null
	);
	const canPrev = $derived(prevTargetId !== null);

	// Bumped on every nav request so a slower skip-walk can't navigate after a
	// newer click/key press has already moved on.
	let navSeq = 0;
	async function goNext() {
		rememberDirection('next');
		const seq = ++navSeq;
		let target = nextTargetId;
		if (skipNav) {
			const found = await queue.nextUnreviewedId(data.item.utterance_id);
			if (seq !== navSeq) return; // superseded by a newer navigation
			target = found ?? null; // exhausted → no nav, never a classified neighbour
		}
		const href = navHref(target);
		if (href) goto(href);
	}
	function goPrev() {
		rememberDirection('prev');
		if (prevHref) goto(prevHref);
	}

	// Touch swipe nav — touch only (no mouse), so Mac trackpad swipe-back
	// keeps working. Threshold 60px horizontal, must be mostly horizontal.
	let touchStartX = 0;
	let touchStartY = 0;
	let touchStartT = 0;
	function onTouchStart(e: TouchEvent) {
		if (e.touches.length !== 1) return;
		touchStartX = e.touches[0].clientX;
		touchStartY = e.touches[0].clientY;
		touchStartT = performance.now();
	}
	function onTouchEnd(e: TouchEvent) {
		if (e.changedTouches.length !== 1) return;
		const dx = e.changedTouches[0].clientX - touchStartX;
		const dy = e.changedTouches[0].clientY - touchStartY;
		const dt = performance.now() - touchStartT;
		if (Math.abs(dx) < 60) return;
		if (Math.abs(dy) > Math.abs(dx) * 0.6) return;
		if (dt > 600) return;
		if (dx < 0 && canNext) goNext();
		else if (dx > 0 && canPrev) goPrev();
	}

	async function copyShareLink() {
		const url = new URL(window.location.href);
		url.searchParams.set('seed', String(data.seed));
		try {
			await navigator.clipboard.writeText(url.toString());
		} catch {
			// ignore — clipboard blocked
		}
		clearTimeout(shareCopiedTimer);
		shareCopied = true;
		shareCopiedTimer = setTimeout(() => (shareCopied = false), 1500);
	}

	// Greek QWERTY physical-key aliases. When the user has the Greek layout
	// active and presses the same physical key as English `i`, `e.key` is `ι`,
	// not `i`. Map both so shortcuts work in either layout.
	function normalizeShortcut(k: string): string {
		const m: Record<string, string> = {
			ι: 'i', χ: 'x', θ: 'u', ψ: 'c', ξ: 'j', κ: 'k', ν: 'n', α: 'a', λ: 'l', ' ': ' ',
			Ι: 'i', Χ: 'x', Θ: 'u', Ψ: 'c', Ξ: 'j', Κ: 'k', Ν: 'n', Α: 'a', Λ: 'l'
		};
		return m[k] ?? k;
	}

	function onKeydown(e: KeyboardEvent) {
		if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) return;
		if (paletteOpen || showUserModal || settingsOpen) return;
		if (e.ctrlKey || e.metaKey || e.altKey) return;
		// ? toggles shortcuts modal; allow it through even when shortcutsOpen is true
		if (e.key === '?') { e.preventDefault(); shortcutsOpen = !shortcutsOpen; return; }
		if (shortcutsOpen) return;
		const k = normalizeShortcut(e.key);
		// Spatial mapping: left key (j) → previous, right key (k) → next.
		if ((e.key === 'ArrowLeft' || k === 'j') && canPrev) { e.preventDefault(); goPrev(); }
		if ((e.key === 'ArrowRight' || k === 'k') && canNext) { e.preventDefault(); goNext(); }
		if (k === 'i') patchStatus('include');
		if (k === 'x') patchStatus('exclude');
		if (k === 'u') patchStatus('uncertain');
		if (k === ' ') { e.preventDefault(); togglePlay(); }
		if (k === 'a') { e.preventDefault(); playbackPrefs.toggleAutoplay(); }
		if (k === 'l') { e.preventDefault(); playbackPrefs.toggleLoop(); }
		if (e.key === '/') { e.preventDefault(); paletteOpen = true; return; }
		// Segment fine-sync. Use physical key codes so the brackets work on the
		// Greek layout too (where e.key would differ). Shift targets the end.
		if (e.code === 'BracketLeft') { e.preventDefault(); nudge(e.shiftKey ? 'end' : 'start', -1); return; }
		if (e.code === 'BracketRight') { e.preventDefault(); nudge(e.shiftKey ? 'end' : 'start', 1); return; }
		// Shift + < / > (physical comma/period) slides the whole utterance window.
		if (e.code === 'Comma' && e.shiftKey) { e.preventDefault(); shiftWhole(-1); return; }
		if (e.code === 'Period' && e.shiftKey) { e.preventDefault(); shiftWhole(1); return; }
		// ↑ / ↓ adjust the nudge step itself (one 10ms grid notch each press).
		if (e.key === 'ArrowUp') { e.preventDefault(); playbackPrefs.setNudgeStepMs(playbackPrefs.nudgeStepMs + 10); return; }
		if (e.key === 'ArrowDown') { e.preventDefault(); playbackPrefs.setNudgeStepMs(playbackPrefs.nudgeStepMs - 10); return; }
		if (k === 'c' && item.edits.length > 1) { e.preventDefault(); showFullChain = !showFullChain; }
		const catId = DIGIT_SHORTCUTS.get(e.key);
		if (catId) { e.preventDefault(); toggleCategory(catId); }
	}

	function toggleCategory(id: TaxonomyId) {
		const current = new Set<TaxonomyId>(
			item.label.error_categories
				.map((c) => normalizeTaxonomyId(c))
				.filter((v): v is TaxonomyId => v !== null)
		);
		if (current.has(id)) current.delete(id);
		else current.add(id);
		const ordered = TAXONOMY.filter((c) => current.has(c.id as TaxonomyId)).map((c) => c.id as TaxonomyId);
		patch({ error_categories: ordered });
	}
</script>

<svelte:window onkeydown={onKeydown} />

<div class="review-page">
	{#if autoSkip.paused}
		<div class="auto-skip-banner" role="status">
			<span>{t('autoSkipPaused', { n: autoSkip.skipped })}</span>
			<button type="button" onclick={resumeAutoSkip}>{t('autoSkipResume')}</button>
		</div>
	{/if}
	<header class="top-bar" bind:this={topBarEl}>
		<div class="top-row">
			<div class="meeting-info">
				{#if item.meeting_name}<span class="meeting-title">{item.meeting_name}</span>{/if}
				{#if item.city_id}<span class="badge city">{item.city_id}</span>{/if}
				{#if item.meeting_date}<span class="badge date">{item.meeting_date}</span>{/if}
			</div>
			<div class="top-row-actions">
				{#if userStore.value}
					<button type="button" class="user-chip" onclick={() => (showUserModal = true)} title="Αλλαγή χρήστη">
						{userStore.value} ✎
					</button>
				{:else}
					<button type="button" class="user-chip missing" onclick={() => (showUserModal = true)}>
						+ όνομα
					</button>
				{/if}
				<button
					type="button"
					class="settings-cog-btn"
					onclick={() => (settingsOpen = true)}
					aria-label={t('settingsAria')}
					title={t('settingsTitle')}
				>
					<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 11-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 110-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 114 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 110 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
				</button>
				<button
					type="button"
					class="share-icon-btn"
					class:copied={shareCopied}
					onclick={copyShareLink}
					title={`${t('shareSeedTitle')} (seed ${data.seed})`}
					aria-label={t('shareSeed')}
				>
					{#if shareCopied}
						<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 8.5l3.5 3.5L13 5"/></svg>
					{:else}
						<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 11V2"/><path d="M5 5l3-3 3 3"/><path d="M3 9v4a1 1 0 001 1h8a1 1 0 001-1V9"/></svg>
					{/if}
				</button>
			</div>
		</div>
		<div class="bottom-row">
			<div class="meta">
				{#if filter}
					<span class="badge filter">{t('filteredQueueLabel', { status: filterLabel })}</span>
				{/if}
				<span class="badge edits" title="Number of edits in this utterance group">
					{item.edits.length} edit{item.edits.length === 1 ? '' : 's'}
				</span>
				{#if !item.chain_consistent}
					<span class="badge warn" title="Some edit's before_text does not match the previous after_text">chain break</span>
				{/if}
			</div>
			<div class="nav-links">
				{#if canPrev}<a href={prevHref} class="nav-btn" title="j / ←" aria-label={t('prevAria')} onclick={(e) => { e.preventDefault(); goPrev(); }}>{t('prev')} <kbd>j</kbd></a>{/if}
				{#if canNext}<a href={nextHref} class="nav-btn" title="k / →" aria-label={t('nextAria')} onclick={(e) => { e.preventDefault(); goNext(); }}>{t('next')} <kbd>k</kbd></a>{/if}
			</div>
		</div>
	</header>

	<main class="content">
		<MeetingContextPanel
			utterances={contextData?.prev.slice(-prevRadius) ?? []}
			label={t('contextBefore')}
			state={contextState}
			hasMore={!!contextData && contextData.prev.length >= prevRadius}
			onLoadMore={() => (prevRadius += 5)}
			loadMoreAtTop
		/>

		<div id="utterance-edit" bind:this={utteranceAnchor} class="utt-anchor">
			{#if reviewPrefs.mobileMode}
				<MobileSwipeCard
					onInclude={swipeInclude}
					onExclude={swipeExclude}
					onTap={reviewPrefs.tapAdvances ? goNext : undefined}
					labelInclude={t('swipeInclude')}
					labelExclude={t('swipeExclude')}
				>
					{@render diffCardBody()}
				</MobileSwipeCard>
			{:else}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<section
					class="diff-section"
					ontouchstart={onTouchStart}
					ontouchend={onTouchEnd}
				>
					{@render diffCardBody()}
				</section>
			{/if}
		</div>

		<!-- Decision controls live in their own bar directly below the card so
		     they never widen the diff grid (which broke the mobile layout). On
		     phones the three buttons stack so they fit the viewport. -->
		<div class="decision-bar">
			<StatusButtons
				status={item.label.include_status}
				saving={saveStatus === 'saving'}
				onchange={(s) => patchStatus(s)}
			/>
		</div>

		{#snippet diffCardBody()}
			{#if canPrev}
				<a href={prevHref} class="utt-chevron left" title={t('prev') + ' (j)'} aria-label={t('prevAria')} onclick={(e) => { e.preventDefault(); goPrev(); }}>
					<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="15 6 9 12 15 18"/></svg>
				</a>
			{/if}
			{#if canNext}
				<a href={nextHref} class="utt-chevron right" title={t('next') + ' (k)'} aria-label={t('nextAria')} onclick={(e) => { e.preventDefault(); goNext(); }}>
					<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 6 15 12 9 18"/></svg>
				</a>
			{/if}
			{#if item.edits.length > 1}
				<label class="chain-toggle">
					<input type="checkbox" bind:checked={showFullChain} />
					{t('chainToggle', { n: item.edits.length })} <kbd>c</kbd>
				</label>
			{/if}
			{#snippet playButton()}
				<div class="play-controls">
					<div
						class="play-btn-wrap"
						class:loading-audio={!audioReady}
						class:ready-flash={audioReadyFlash}
						class:playing={isPlaying}
						onanimationend={(e) => { if (e.animationName === 'sweep-border') audioReadyFlash = false; }}
					>
						<button
							type="button"
							class="play-btn"
							class:playing={isPlaying}
							disabled={!audioReady}
							onclick={togglePlay}
							aria-label={!audioReady ? 'Φόρτωση...' : isPlaying ? 'Pause' : 'Play'}
						>
							{#if !audioReady}
								<span class="spinner" aria-hidden="true"></span>
							{:else}
								{isPlaying ? '⏸' : '▶'}
							{/if}
						</button>
					</div>
					<label class="pref-toggle" title="Autoplay (a)">
						<input type="checkbox" checked={playbackPrefs.autoplay} onchange={() => playbackPrefs.toggleAutoplay()} />
						<span>auto</span> <kbd>a</kbd>
					</label>
					<label class="pref-toggle" title="Loop (l)">
						<input type="checkbox" checked={playbackPrefs.loop} onchange={() => playbackPrefs.toggleLoop()} />
						<span>loop</span> <kbd>l</kbd>
					</label>
				</div>
			{/snippet}
			<Diff
				before={beforeText}
				after={afterText}
				speakerName={currentSpeakerName}
				speakerLoading={contextState === 'loading'}
				errorCategoryIds={item.label.error_categories}
				lang={getLang()}
				playSlot={playButton}
			/>
		{/snippet}

		<MeetingContextPanel
			utterances={contextData?.next.slice(0, nextRadius) ?? []}
			label={t('contextAfter')}
			state={contextState}
			hasMore={!!contextData && contextData.next.length >= nextRadius}
			onLoadMore={() => (nextRadius += 5)}
		/>

		<section class="audio-section">
			<!--
				TODO(segment-waveform): once we have a segment-render lib + a
				/api/audio/segment?u=&start=&end= endpoint, replace this native
				player with the waveform component again. Until then we use ONLY
				a native <audio> element so playback
				streams range chunks (~tens of KB) instead of pulling the whole
				3-hour meeting MP3.
			-->
			<div class="audio-toolbar">
				<div class="boundary">
					<span class="boundary-label">start</span>
					<button type="button" class="nudge" disabled={!canStartBack} onclick={() => nudge('start', -1)} title={t('nudgeStartBack')} aria-label={t('nudgeStartBack')}>−<kbd>[</kbd></button>
					<input
						type="number"
						step="0.1"
						min="0"
						value={regionStart}
						onchange={(e) => commitTimestamps(Number((e.target as HTMLInputElement).value), regionEnd)}
					/>
					<button type="button" class="nudge" disabled={!canStartFwd} onclick={() => nudge('start', 1)} title={t('nudgeStartFwd')} aria-label={t('nudgeStartFwd')}>+<kbd>]</kbd></button>
				</div>
				<div class="boundary">
					<span class="boundary-label">end</span>
					<button type="button" class="nudge" disabled={!canEndBack} onclick={() => nudge('end', -1)} title={t('nudgeEndBack')} aria-label={t('nudgeEndBack')}>−<kbd>{'{'}</kbd></button>
					<input
						type="number"
						step="0.1"
						min="0"
						value={regionEnd}
						onchange={(e) => commitTimestamps(regionStart, Number((e.target as HTMLInputElement).value))}
					/>
					<button type="button" class="nudge" disabled={!canEndFwd} onclick={() => nudge('end', 1)} title={t('nudgeEndFwd')} aria-label={t('nudgeEndFwd')}>+<kbd>{'}'}</kbd></button>
				</div>
				<div class="boundary">
					<span class="boundary-label">{t('nudgeMoveLabel')}</span>
					<button type="button" class="nudge" disabled={!canMoveBack} onclick={() => shiftWhole(-1)} title={t('nudgeMoveBack')} aria-label={t('nudgeMoveBack')}>◀<kbd>{'<'}</kbd></button>
					<button type="button" class="nudge" disabled={!canMoveFwd} onclick={() => shiftWhole(1)} title={t('nudgeMoveFwd')} aria-label={t('nudgeMoveFwd')}>▶<kbd>{'>'}</kbd></button>
				</div>
				<label class="step-ctl" title={t('nudgeStepTitle')}>
					<span>{t('nudgeStepLabel')}</span>
					<input
						type="number"
						step="10"
						min="10"
						max="1000"
						value={playbackPrefs.nudgeStepMs}
						onchange={(e) => playbackPrefs.setNudgeStepMs(Number((e.target as HTMLInputElement).value))}
					/>
					<span class="unit">ms</span>
					<kbd>↑</kbd><kbd>↓</kbd>
				</label>
				<span class="hint">segment: {(regionEnd - regionStart).toFixed(2)}s of {item.end.toFixed(1)}s</span>
			</div>
			<div class="audio-wrap" class:loading={!audioReady}>
				<!--
					The visible <audio> is owned by audioPool. On every
					setActive, the pool moves the corresponding element
					(which may already be fully buffered from prefetch)
					into this slot — no duplicate fetch.
				-->
				<div class="audio-slot" bind:this={audioSlot}></div>
				{#if !audioReady}
					<div class="audio-skeleton" aria-hidden="true" role="presentation">
						<span class="audio-skeleton-bar"></span>
						<span class="audio-skeleton-text">{t('audioLoading')}</span>
					</div>
				{/if}
			</div>
		</section>

		<section class="label-section">
			<TaxonomySelect
				values={item.label.error_categories}
				disabled={saveStatus === 'saving'}
				onchange={(v) => patch({ error_categories: v })}
			/>
			<label class="notes-label">
				<span>{t('notes')}</span>
				<textarea
					rows="2"
					placeholder={t('notesPlaceholder')}
					value={item.label.reviewer_notes ?? ''}
					oninput={(e) => {
						const val = (e.target as HTMLTextAreaElement).value;
						clearTimeout(saveTimer);
						saveTimer = setTimeout(() => patch({ reviewer_notes: val || null }), 600);
					}}
				></textarea>
			</label>
			{#if saveStatus !== 'idle'}
				<span class="save-status {saveStatus}">
					{saveStatus === 'saving' ? t('saving') : saveStatus === 'saved' ? t('savedOk') : t('saveError')}
				</span>
			{/if}
			<div class="shortcuts">
				<kbd>Space</kbd> play/pause
				<kbd>j</kbd><kbd>k</kbd> prev/next
				<kbd>i</kbd><kbd>x</kbd><kbd>u</kbd> include/exclude/uncertain
				<kbd>1</kbd>…<kbd>0</kbd> category
				<kbd>/</kbd> palette
				<kbd>a</kbd> autoplay
				<kbd>l</kbd> loop
				<kbd>[</kbd><kbd>]</kbd> {t('shortcutNudgeStart')}
				<kbd>{'{'}</kbd><kbd>{'}'}</kbd> {t('shortcutNudgeEnd')}
				<kbd>{'<'}</kbd><kbd>{'>'}</kbd> {t('shortcutNudgeMove')}
				<kbd>↑</kbd><kbd>↓</kbd> {t('shortcutStep')}
				<kbd>?</kbd> {t('shortcutsModalTitle')}
				{#if item.edits.length > 1}<kbd>c</kbd> {t('chainToggleHint')}{/if}
			</div>
		</section>
	</main>

	<CategoryPalette
		open={paletteOpen}
		onclose={() => (paletteOpen = false)}
		values={item.label.error_categories}
		onchange={(cats) => patch({ error_categories: cats })}
	/>
</div>

{#if showUserModal}
	<UserPickerModal onclose={() => (showUserModal = false)} />
{/if}

<ShortcutsModal open={shortcutsOpen} onclose={() => (shortcutsOpen = false)} showChain={item.edits.length > 1} />
<SettingsModal open={settingsOpen} onclose={() => (settingsOpen = false)} />

<style>
	.review-page { max-width: 860px; margin: 0 auto; padding: 1rem 1rem 5rem; }
	.auto-skip-banner {
		display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;
		background: #fffbeb; border: 1px solid #fcd34d; color: #92400e;
		border-radius: 8px; padding: 0.5rem 0.75rem; margin-bottom: 0.75rem;
		font-size: 0.85rem;
	}
	.auto-skip-banner button {
		flex-shrink: 0; font-family: inherit; font-size: 0.8rem; cursor: pointer;
		background: #92400e; color: #fff; border: none; border-radius: 6px;
		padding: 0.3rem 0.7rem;
	}
	.auto-skip-banner button:hover { background: #78350f; }
	.top-bar {
		position: sticky; top: 0; z-index: 10;
		background: #ffffff;
		border-bottom: 1px solid var(--border, #e2e8f0);
		box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
		padding: 0.5rem 1rem 0.4rem;
		margin: -1rem -1rem 1.2rem;
		display: flex; flex-direction: column; gap: 0.3rem;
	}
	.top-row {
		display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
	}
	.meeting-info {
		display: flex; align-items: baseline; gap: 0.4rem; flex-wrap: wrap; min-width: 0;
	}
	.meeting-title {
		font-size: 0.95rem; font-weight: 600; color: #0f172a;
		white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 380px;
	}
	.top-row-actions { display: flex; align-items: center; gap: 0.35rem; flex-shrink: 0; }
	.user-chip {
		font-size: 0.72rem; padding: 0.18rem 0.55rem; border-radius: 999px;
		background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0;
		cursor: pointer; font-family: inherit; white-space: nowrap;
	}
	.user-chip:hover { background: #e2e8f0; }
	.user-chip.missing { color: #2563eb; border-color: #93c5fd; background: #eff6ff; }
	.share-icon-btn {
		font-size: 0.85rem; padding: 0.2rem 0.45rem; border-radius: 6px;
		background: transparent; border: 1px solid var(--border, #e2e8f0);
		cursor: pointer; transition: background 0.15s; line-height: 1;
	}
	.share-icon-btn:hover { background: #f1f5f9; }
	.settings-cog-btn {
		display: inline-flex; align-items: center; justify-content: center;
		padding: 0.2rem 0.35rem; border-radius: 6px;
		background: transparent; border: 1px solid var(--border, #e2e8f0);
		cursor: pointer; color: #475569; transition: background 0.15s, color 0.15s;
	}
	.settings-cog-btn:hover { background: #f1f5f9; color: #0f172a; }
	.share-icon-btn.copied { background: #dcfce7; border-color: #86efac; }
	.bottom-row {
		display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
	}
	.meta { display: flex; gap: 0.35rem; flex-wrap: wrap; align-items: center; }
	.badge {
		font-size: 0.72rem; padding: 0.18rem 0.55rem; border-radius: 999px;
		font-weight: 500; letter-spacing: 0.01em;
	}
	.badge.meeting { background: var(--accent-light, #dbeafe); color: #1e3a8a; }
	.badge.date { background: var(--surface-3, #f1f5f9); color: var(--text-2, #475569); }
	.badge.city { background: #ecfdf5; color: #047857; }
	.badge.edits { background: #fef9c3; color: #713f12; }
	.badge.warn { background: #fee2e2; color: #b91c1c; }
	.nav-links { display: flex; gap: 0.4rem; flex-wrap: wrap; }
	.nav-btn {
		font-size: 0.8rem; padding: 0.25rem 0.65rem;
		border-radius: var(--radius-sm, 6px);
		background: var(--surface, #fff);
		border: 1px solid var(--border, #e2e8f0);
		text-decoration: none; color: var(--text-2, #475569);
		display: inline-flex; align-items: center; gap: 0.3rem;
	}
	.nav-btn:hover { background: var(--surface-3, #f1f5f9); }
	.badge.filter { font-weight: 600; }
	.badge.filter.include { background: #dcfce7; color: #14532d; }
	.badge.filter.exclude { background: #fee2e2; color: #7f1d1d; }
	.badge.filter.uncertain { background: #fef3c7; color: #78350f; }
	.content { display: flex; flex-direction: column; gap: 1.2rem; }
	.audio-toolbar {
		display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
		margin-bottom: 0.6rem; font-size: 0.85rem;
	}
	.audio-toolbar label { display: flex; align-items: center; gap: 0.3rem; }
	.audio-toolbar input[type="number"] {
		width: 5rem; padding: 0.2rem 0.4rem;
		border: 1px solid var(--border, #e2e8f0);
		border-radius: var(--radius-sm, 6px);
		font-size: 0.85rem; background: var(--surface, #fff);
	}
	.boundary { display: inline-flex; align-items: center; gap: 0.3rem; }
	.boundary-label { color: var(--text-2, #475569); }
	.boundary input[type="number"] { width: 4.6rem; }
	.nudge {
		display: inline-flex; align-items: center; gap: 0.18rem;
		padding: 0.2rem 0.4rem; line-height: 1;
		border: 1px solid var(--border, #e2e8f0);
		border-radius: var(--radius-sm, 6px);
		background: var(--surface, #fff); color: var(--text-2, #475569);
		cursor: pointer; font-size: 0.85rem; font-family: inherit;
	}
	.nudge:hover:not(:disabled) { background: var(--surface-3, #f1f5f9); }
	.nudge:active:not(:disabled) { transform: scale(0.95); }
	.nudge:disabled { opacity: 0.4; cursor: default; }
	.nudge kbd { font-size: 0.62rem; padding: 0 0.22rem; }
	.step-ctl { display: inline-flex; align-items: center; gap: 0.3rem; }
	.step-ctl .unit { color: var(--text-3, #94a3b8); font-size: 0.78rem; }
	.step-ctl input[type="number"] { width: 4rem; }
	.step-ctl kbd { font-size: 0.62rem; padding: 0 0.22rem; }
	:global(body.mobile-mode) .step-ctl kbd { display: none; }
	@media (max-width: 540px) { .step-ctl kbd { display: none; } }
	:global(body.mobile-mode) .nudge kbd { display: none; }
	@media (max-width: 540px) { .nudge kbd { display: none; } }
	.play-controls {
		display: inline-flex; align-items: center; gap: 0.5rem;
		flex-wrap: nowrap; min-width: 0;
	}
	:global(body.mobile-mode) .play-controls kbd { display: none; }
	:global(body.mobile-mode) .pref-toggle { font-size: 0.7rem; }
	@media (max-width: 540px) {
		.play-controls kbd { display: none; }
	}
	.pref-toggle {
		display: inline-flex; align-items: center; gap: 0.2rem;
		font-size: 0.72rem; color: var(--text-2, #475569); cursor: pointer;
		user-select: none;
	}
	.pref-toggle input { margin: 0; cursor: pointer; }
	.spinner {
		display: inline-block; width: 0.9em; height: 0.9em;
		border: 2px solid currentColor; border-top-color: transparent;
		border-radius: 50%;
		animation: spin 0.7s linear infinite;
	}
	@keyframes spin { to { transform: rotate(360deg); } }
	kbd {
		display: inline-block; padding: 0 4px; font: 11px ui-monospace, monospace;
		border: 1px solid #cbd5e1; border-bottom-width: 2px; border-radius: 3px;
		background: #f8fafc; color: #475569; line-height: 1.6;
	}
	.play-btn-wrap { position: relative; display: inline-flex; }
	.play-btn {
		min-width: 2.4rem;
		padding: 0.3rem 0.9rem;
		border: 1px solid var(--border-accent, #93c5fd);
		background: var(--surface, #fff);
		border-radius: 999px; cursor: pointer; font-size: 0.95rem;
		line-height: 1.2;
		transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.05s;
	}
	.play-btn:hover:not(:disabled) { background: var(--accent-light, #dbeafe); }
	.play-btn:active:not(:disabled) { transform: scale(0.96); }
	.play-btn:disabled { cursor: default; }
	.play-btn.playing {
		background: #16a34a;
		border-color: #15803d;
		color: #fff;
	}
	.play-btn-wrap.playing .play-btn {
		animation: play-btn-throb 1.5s ease-in-out infinite;
	}
	@keyframes play-btn-throb {
		0%, 100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.55); }
		50%      { box-shadow: 0 0 0 6px rgba(22, 163, 74, 0); }
	}

	/* Pulse while audio is buffering (but not while playing — the throb takes over) */
	.play-btn-wrap.loading-audio:not(.playing) .play-btn {
		animation: play-btn-pulse 1.4s ease-in-out infinite;
	}
	@keyframes play-btn-pulse {
		0%, 100% { opacity: 0.45; }
		50% { opacity: 0.85; }
	}

	/* One-shot sweep animation when audio becomes ready */
	@property --sweep-angle {
		syntax: '<angle>';
		inherits: false;
		initial-value: 0deg;
	}
	.play-btn-wrap.ready-flash::before {
		content: '';
		position: absolute;
		inset: -2px;
		border-radius: 999px;
		background: conic-gradient(#60a5fa var(--sweep-angle), transparent var(--sweep-angle));
		mask: radial-gradient(farthest-side, transparent calc(100% - 2px), black calc(100% - 2px));
		-webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 2px), black calc(100% - 2px));
		animation: sweep-border 0.3s linear forwards;
		pointer-events: none;
	}
	@keyframes sweep-border {
		from { --sweep-angle: 0deg; }
		to { --sweep-angle: 360deg; }
	}
	.audio-toolbar .hint { color: var(--text-3, #94a3b8); font-size: 0.78rem; margin-left: auto; }
	.native-player { width: 100%; height: 36px; display: block; }
	.audio-wrap { position: relative; }
	.audio-skeleton {
		position: absolute; inset: 0;
		display: flex; align-items: center; gap: 0.6rem;
		padding: 0 0.6rem;
		border-radius: var(--radius-sm, 6px);
		background: linear-gradient(90deg, rgba(226,232,240,0) 0%, rgba(226,232,240,0.65) 50%, rgba(226,232,240,0) 100%);
		background-size: 200% 100%;
		animation: audio-skeleton-shimmer 1.4s linear infinite;
		pointer-events: none;
	}
	.audio-skeleton-bar {
		flex: 1; height: 8px;
		border-radius: 4px;
		background: var(--border, #e2e8f0);
	}
	.audio-skeleton-text {
		font-size: 0.72rem;
		color: var(--text-3, #94a3b8);
		font-weight: 500;
		white-space: nowrap;
	}
	@keyframes audio-skeleton-shimmer {
		0% { background-position: 200% 0; }
		100% { background-position: -200% 0; }
	}

	.diff-section, .audio-section, .label-section {
		background: var(--surface, #fff);
		border: 1px solid var(--border, #e2e8f0);
		border-radius: var(--radius, 10px);
		padding: 1rem 1.1rem;
		box-shadow: var(--shadow, 0 1px 3px rgba(0,0,0,.08));
	}
	.diff-section { position: relative; min-width: 0; overflow: hidden; }
	.utt-anchor { scroll-margin-top: calc(var(--top-bar-h, 80px) + 12px); min-width: 0; }
	.decision-bar { margin-top: 0.6rem; min-width: 0; }
	.decision-bar :global(.status-buttons) { max-width: 460px; margin: 0 auto; }
	.utt-chevron {
		position: absolute; top: 50%; transform: translateY(-50%);
		width: 30px; height: 30px;
		display: flex; align-items: center; justify-content: center;
		border-radius: 50%;
		background: rgba(248, 250, 252, 0.85);
		border: 1px solid var(--border, #e2e8f0);
		color: #64748b; text-decoration: none;
		opacity: 0.55; transition: opacity 0.15s, color 0.15s, background 0.15s;
	}
	.utt-chevron:hover { opacity: 1; color: #0f172a; background: #f1f5f9; }
	.utt-chevron.left { left: -42px; }
	.utt-chevron.right { right: -42px; }
	@media (max-width: 960px) {
		.utt-chevron.left { left: -8px; }
		.utt-chevron.right { right: -8px; }
		.utt-chevron { background: rgba(255, 255, 255, 0.92); }
	}
	/* Chevrons stay visible on every breakpoint (including mobile) — the
	   reviewer asked for always-available prev/next arrows. On phones they sit
	   just inside the card edges via the 960px rule above. */
	.chain-toggle {
		display: flex; align-items: center; gap: 0.4rem;
		font-size: 0.8rem; color: var(--text-2, #475569);
		margin-bottom: 0.6rem;
	}
	.notes-label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.875rem; color: var(--text-2, #475569); }
	.notes-label textarea {
		padding: 0.45rem 0.55rem; border: 1px solid var(--border, #e2e8f0);
		border-radius: var(--radius-sm, 6px); font-size: 0.875rem;
		resize: vertical; background: var(--surface-2, #f8fafc); font-family: inherit;
	}
	.save-status { font-size: 0.8rem; }
	.save-status.saving { color: var(--text-3, #94a3b8); }
	.save-status.saved { color: #16a34a; }
	.save-status.error { color: #dc2626; }
	.shortcuts {
		font-size: 0.7rem; color: var(--text-3, #94a3b8);
		display: flex; flex-wrap: wrap; gap: 0.25rem 0.55rem; align-items: center;
	}
	kbd {
		font-size: 0.68rem; background: var(--surface-3, #f1f5f9);
		border: 1px solid var(--border, #e2e8f0); border-radius: 4px;
		padding: 0.05rem 0.32rem; font-family: monospace;
		color: var(--text-2, #475569);
	}
</style>

<script lang="ts">
	import { goto } from '$app/navigation';
	import { untrack } from 'svelte';
	import Diff from '$lib/components/Diff.svelte';
	import StatusButtons from '$lib/components/StatusButtons.svelte';
	import TaxonomySelect from '$lib/components/TaxonomySelect.svelte';
	import CategoryPalette from '$lib/components/CategoryPalette.svelte';
	import MeetingContextPanel from '$lib/components/MeetingContextPanel.svelte';
	import type { Group, GroupPatchBody } from '$lib/domain/groups';
	import type { MeetingContext } from '$lib/domain/meeting-context';
	import { TAXONOMY, normalizeTaxonomyId, type TaxonomyId } from '$lib/shared/taxonomy';
	import { t } from '$lib/i18n.svelte';
	import * as queue from '$lib/client/group-queue.svelte';
	import { resolveAudioUrls } from '$lib/client/audio-source';
	import { audioPool } from '$lib/client/audio-pool.svelte';
	import * as meetingCtx from '$lib/client/meeting-context.svelte';
	import { reviewHref } from '$lib/shared/urls';
	import { page } from '$app/state';
	import type { PageData } from './$types';

	const DIGIT_SHORTCUTS: ReadonlyMap<string, TaxonomyId> = new Map(
		TAXONOMY.filter((c) => c.shortcut).map((c) => [c.shortcut!, c.id as TaxonomyId])
	);

	const { data }: { data: PageData } = $props();

	const item = $derived<Group>(queue.get(data.item.utterance_id) ?? data.item);
	const prev = $derived(queue.prevOf(data.item.utterance_id));
	const next = $derived(queue.nextOf(data.item.utterance_id));

	let showFullChain = $state(false);

	const beforeText = $derived(showFullChain
		? item.edits.map((e, i) => `[edit ${i + 1} by ${e.edited_by ?? '—'}]\n${e.before_text}`).join('\n\n')
		: item.initial_before_text);
	const afterText = $derived(showFullChain
		? item.edits.map((e, i) => `[edit ${i + 1} by ${e.edited_by ?? '—'}]\n${e.after_text}`).join('\n\n')
		: item.final_after_text);

	let saveStatus = $state<'idle' | 'saving' | 'saved' | 'error'>('idle');
	let saveTimer: ReturnType<typeof setTimeout>;

	async function patch(updates: GroupPatchBody) {
		clearTimeout(saveTimer);
		saveStatus = 'saving';
		queue.patchLocalLabel(item.utterance_id, updates);
		try {
			const res = await fetch(`/api/review/group/${encodeURIComponent(item.utterance_id)}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(updates)
			});
			if (!res.ok) throw new Error(await res.text());
			saveStatus = 'saved';
		} catch {
			saveStatus = 'error';
		}
		saveTimer = setTimeout(() => (saveStatus = 'idle'), 2000);
	}

	let paletteOpen = $state(false);
	let audioEl: HTMLAudioElement | null = $state(null);
	// Track whether the active <audio> has buffered enough data to start
	// playing. The native control still renders, but we overlay a skeleton +
	// disable the keyboard shortcut so a Space press doesn't silently fail.
	let audioReady = $state(false);
	function onAudioCanPlay() { audioReady = true; }
	function onAudioWaiting() { audioReady = false; }
	function onAudioLoadStart() { audioReady = false; }
	// Native <audio> playback does NOT need CORS. Direct CDN is preferred so
	// Vercel proxy bandwidth stays near zero. `audio-source.ts` keeps the proxied
	// URL as a fallback for onerror. See decisions/audio.md.
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
	const currentCityId = $derived(item.city_id);
	const currentMeetingId = $derived(item.meeting_id);

	// Audio prefetch: only re-run when the *id* changes. The neighbours
	// snapshot is read inside untrack() so SvelteMap mutations on neighbour
	// groups (a label patch on a +3 ahead, for example) don't drag this
	// effect with them. Same for the usingFallback reset.
	$effect(() => {
		const id = currentId;
		untrack(() => {
			usingFallback = false;
			const audioNeighbours = queue.neighborsAround(id, 5).map((g) => ({
				utterance_id: g.utterance_id,
				url: resolveAudioUrls(g.audio_url).direct,
				// Use the adjusted_start if the reviewer already set one for
				// this neighbour; otherwise the original start. This is the
				// time the visible player will seek to when the user
				// navigates to it.
				start: g.label.adjusted_start ?? g.start
			}));
			audioPool.warm(id, audioNeighbours);
		});
	});

	// Transcript prefetch: same sliding window, separate effect so its
	// dependencies stay independent of the audio one. Triggers on id change
	// only; the neighbour scan runs untracked.
	$effect(() => {
		const id = currentId;
		untrack(() => {
			const seenMeetings = new Set<string>();
			for (const g of queue.neighborsAround(id, 5)) {
				if (!g.city_id || !g.meeting_id) continue;
				const k = `${g.city_id}|${g.meeting_id}`;
				if (seenMeetings.has(k)) continue;
				seenMeetings.add(k);
				meetingCtx.prefetch(g.city_id, g.meeting_id);
			}
		});
	});

	// Surrounding-utterance context, fetched client-side from the LRU cache
	// in `meeting-context.svelte`. The cache transparently relays the meeting
	// JSON through /api/oc-meeting/... (a CORS bridge — no slicing happens
	// server-side) and slices ±5 around the current utterance here.
	let contextState = $state<'loading' | 'ready' | 'error' | 'empty'>('loading');
	let contextData = $state<MeetingContext | null>(null);

	$effect(() => {
		const id = currentId;
		const cityId = currentCityId;
		const meetingId = currentMeetingId;
		let cancelled = false;

		// Avoid the "loading…" flicker when the meeting is already resolved
		// in the LRU. We still call getContext so the slice can change
		// (different utterance in the same meeting), but we keep the old
		// contextData visible until the new slice lands — synchronously, in
		// the .then handler that resolves the same tick from the LRU.
		const wasCached = meetingCtx.hasMeeting(cityId, meetingId);
		if (!wasCached) {
			contextState = 'loading';
			contextData = null;
		}

		meetingCtx
			.getContext(cityId, meetingId, id, 5)
			.then((data) => {
				if (cancelled) return;
				contextData = data;
				if (data.error) {
					contextState = 'error';
				} else if (data.prev.length === 0 && data.next.length === 0) {
					contextState = 'empty';
				} else {
					contextState = 'ready';
				}
			})
			.catch(() => {
				if (!cancelled) contextState = 'error';
			});
		return () => {
			cancelled = true;
		};
	});

	// Speaker for the current utterance — pulled from the context payload
	// once it lands. Falls back to the SPEAKER_N label when the meeting JSON
	// has no person record for the tag.
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

	function onAudioTimeUpdate() {
		if (!audioEl) return;
		if (!audioEl.paused && audioEl.currentTime >= regionEnd) {
			audioEl.pause();
			audioEl.currentTime = regionStart;
		}
	}

	function commitTimestamps(start: number, end: number) {
		if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end) return;
		patch({ adjusted_start: start, adjusted_end: end });
	}

	const prevHref = $derived(prev ? reviewHref({ utterance_id: prev.utterance_id, seed: data.seed }) : null);
	const nextHref = $derived(next ? reviewHref({ utterance_id: next.utterance_id, seed: data.seed }) : null);
	const highlightEditId = $derived(page.url.searchParams.get('highlight'));

	function goNext() { if (nextHref) goto(nextHref); }
	function goPrev() { if (prevHref) goto(prevHref); }

	async function copyShareLink() {
		const url = new URL(window.location.href);
		url.searchParams.set('seed', String(data.seed));
		try {
			await navigator.clipboard.writeText(url.toString());
		} catch {
			// ignore — clipboard blocked
		}
	}

	function onKeydown(e: KeyboardEvent) {
		if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) return;
		if (paletteOpen) return;
		if (e.ctrlKey || e.metaKey || e.altKey) return;
		if ((e.key === 'ArrowLeft' || e.key === 'k') && prevHref) { e.preventDefault(); goPrev(); }
		if ((e.key === 'ArrowRight' || e.key === 'j') && nextHref) { e.preventDefault(); goNext(); }
		if (e.key === 'i') patch({ include_status: 'include' });
		if (e.key === 'x') patch({ include_status: 'exclude' });
		if (e.key === 'u') patch({ include_status: 'uncertain' });
		if (e.key === ' ') { e.preventDefault(); togglePlay(); }
		if (e.key === '/') { e.preventDefault(); paletteOpen = true; return; }
		if (e.key === 'c' && item.edits.length > 1) { e.preventDefault(); showFullChain = !showFullChain; }
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
	<header class="top-bar">
		<div class="meta">
			<button
				type="button"
				class="badge mode share-btn"
				onclick={copyShareLink}
				title={t('shareSeedTitle')}
			>seed {data.seed} · {t('shareSeed')}</button>
			{#if item.meeting_name}
				<span class="badge meeting">{item.meeting_name}</span>
			{/if}
			{#if item.city_id}
				<span class="badge city">{item.city_id}</span>
			{/if}
			{#if item.meeting_date}
				<span class="badge date">{item.meeting_date}</span>
			{/if}
			<span class="badge edits" title="Number of edits in this utterance group">
				{item.edits.length} edit{item.edits.length === 1 ? '' : 's'}
			</span>
			{#if !item.chain_consistent}
				<span class="badge warn" title="Some edit's before_text does not match the previous after_text">chain break</span>
			{/if}
		</div>
		<div class="nav-links">
			{#if prevHref}<a href={prevHref} class="nav-btn" title="k / ←">{t('prev')}</a>{/if}
			{#if nextHref}<a href={nextHref} class="nav-btn" title="j / →" onclick={(e) => { e.preventDefault(); goNext(); }}>{t('next')}</a>{/if}
			<a href="/stats" class="nav-btn stats-link">{t('statsLink')}</a>
		</div>
	</header>

	<main class="content">
		<MeetingContextPanel
			utterances={contextData?.prev ?? []}
			label={t('contextBefore')}
			state={contextState}
		/>

		<section class="diff-section">
			{#if highlightEditId}
				{@const hl = item.edits.find((e) => e.edit_id === highlightEditId)}
				<div class="highlight-banner">
					{#if hl}
						{t('highlightingEdit')} <code>{highlightEditId}</code>
						{#if item.edits.length > 1}
							· {item.edits.length} {t('editsInThisUtterance')}
						{/if}
					{:else}
						{t('highlightEditNotFound')} <code>{highlightEditId}</code>
					{/if}
				</div>
			{/if}
			{#if item.edits.length > 1}
				<label class="chain-toggle">
					<input type="checkbox" bind:checked={showFullChain} />
					{t('chainToggle', { n: item.edits.length })} <kbd>c</kbd>
				</label>
			{/if}
			<Diff before={beforeText} after={afterText} speakerName={currentSpeakerName} />
		</section>

		<MeetingContextPanel
			utterances={contextData?.next ?? []}
			label={t('contextAfter')}
			state={contextState}
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
				<button type="button" class="play-btn" onclick={togglePlay} title="Space">▶/⏸</button>
				<label>
					<span>start</span>
					<input
						type="number"
						step="0.1"
						min="0"
						value={regionStart}
						onchange={(e) => commitTimestamps(Number((e.target as HTMLInputElement).value), regionEnd)}
					/>
				</label>
				<label>
					<span>end</span>
					<input
						type="number"
						step="0.1"
						min="0"
						value={regionEnd}
						onchange={(e) => commitTimestamps(regionStart, Number((e.target as HTMLInputElement).value))}
					/>
				</label>
				<span class="hint">segment: {(regionEnd - regionStart).toFixed(2)}s of {item.end.toFixed(1)}s</span>
			</div>
			<div class="audio-wrap" class:loading={!audioReady}>
				<audio
					bind:this={audioEl}
					class="native-player"
					controls
					preload="auto"
					src={activeAudioUrl}
					onloadedmetadata={onAudioMeta}
					ontimeupdate={onAudioTimeUpdate}
					onerror={onAudioError}
					oncanplay={onAudioCanPlay}
					onwaiting={onAudioWaiting}
					onloadstart={onAudioLoadStart}
				></audio>
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
				<kbd>j</kbd><kbd>k</kbd> next/prev
				<kbd>i</kbd><kbd>x</kbd><kbd>u</kbd> include/exclude/uncertain
				<kbd>1</kbd>…<kbd>0</kbd> category
				<kbd>/</kbd> palette
				{#if item.edits.length > 1}<kbd>c</kbd> {t('chainToggleHint')}{/if}
			</div>
		</section>
	</main>

	<footer class="action-footer">
		<StatusButtons
			status={item.label.include_status}
			saving={saveStatus === 'saving'}
			onchange={(s) => patch({ include_status: s })}
		/>
		<div class="footer-nav">
			{#if prevHref}<a href={prevHref} class="footer-nav-btn">{t('prev')}</a>{/if}
			{#if nextHref}<a href={nextHref} class="footer-nav-btn primary" onclick={(e) => { e.preventDefault(); goNext(); }}>{t('next')}</a>{/if}
		</div>
	</footer>

	<CategoryPalette
		open={paletteOpen}
		onclose={() => (paletteOpen = false)}
		values={item.label.error_categories}
		onchange={(cats) => patch({ error_categories: cats })}
	/>
</div>

<style>
	.review-page { max-width: 860px; margin: 0 auto; padding: 1rem 1rem 5rem; }
	.top-bar {
		position: sticky; top: 0; z-index: 10;
		background: var(--surface-2, #f8fafc);
		border-bottom: 1px solid var(--border, #e2e8f0);
		display: flex; justify-content: space-between; align-items: center;
		padding: 0.5rem 0; margin-bottom: 1.2rem; gap: 0.5rem; flex-wrap: wrap;
	}
	.meta { display: flex; gap: 0.35rem; flex-wrap: wrap; }
	.badge {
		font-size: 0.72rem; padding: 0.18rem 0.55rem; border-radius: 999px;
		font-weight: 500; letter-spacing: 0.01em;
	}
	.badge.mode { background: #0f172a; color: #f8fafc; }
	.badge.share-btn { cursor: pointer; font-family: inherit; border: none; }
	.badge.share-btn:hover { background: #1e293b; }
	.highlight-banner {
		background: #fef3c7; color: #92400e; padding: 0.45rem 0.7rem;
		border: 1px solid #fbbf24; border-radius: 6px; font-size: 0.85rem;
		margin-bottom: 0.6rem;
	}
	.highlight-banner code { background: #fde68a; padding: 0 0.3rem; border-radius: 3px; }
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
	}
	.nav-btn:hover { background: var(--surface-3, #f1f5f9); }
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
	.audio-toolbar .play-btn {
		padding: 0.2rem 0.7rem;
		border: 1px solid var(--border-accent, #93c5fd);
		background: var(--surface, #fff);
		border-radius: 999px; cursor: pointer; font-size: 0.85rem;
	}
	.audio-toolbar .play-btn:hover { background: var(--accent-light, #dbeafe); }
	.audio-toolbar .hint { color: var(--text-3, #94a3b8); font-size: 0.78rem; margin-left: auto; }
	.native-player { width: 100%; height: 36px; display: block; }
	.audio-wrap { position: relative; }
	/* While the audio buffers, dim the native control and overlay a skeleton
	   that visibly signals the play button isn't ready yet. Pointer events
	   stay enabled on the overlay so clicks land on it (not the disabled
	   play button), and the keyboard toggle is guarded server-side too. */
	.audio-wrap.loading .native-player { opacity: 0.45; pointer-events: none; filter: grayscale(0.4); }
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
	.action-footer {
		position: fixed; bottom: 0; left: 0; right: 0; z-index: 20;
		background: var(--surface, #fff); border-top: 1px solid var(--border, #e2e8f0);
		box-shadow: 0 -2px 12px rgba(0,0,0,.08);
		display: flex; align-items: center; justify-content: space-between;
		padding: 0.6rem max(1rem, calc(50vw - 430px)); gap: 1rem;
	}
	.footer-nav { display: flex; gap: 0.4rem; }
	.footer-nav-btn {
		font-size: 0.82rem; padding: 0.35rem 0.8rem;
		border-radius: var(--radius-sm, 6px);
		background: var(--surface-3, #f1f5f9);
		border: 1px solid var(--border, #e2e8f0);
		text-decoration: none; color: var(--text-2, #475569);
	}
	.footer-nav-btn.primary { background: var(--accent, #2563eb); color: #fff; border-color: transparent; }
</style>

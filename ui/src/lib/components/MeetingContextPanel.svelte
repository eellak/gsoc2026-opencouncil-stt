<script lang="ts">
	import { t } from '$lib/i18n.svelte';
	import type { ContextUtterance } from '$lib/domain/meeting-context';
	import { mergeBySpeaker } from '$lib/client/meeting-context.svelte';
	import { toGreekUpper } from '$lib/shared/text';

	interface Props {
		utterances: ContextUtterance[];
		label: string;
		state: 'loading' | 'ready' | 'error' | 'empty';
		onLoadMore?: () => void;
		hasMore?: boolean;
		loadMoreAtTop?: boolean;
	}
	const { utterances, label, state, onLoadMore, hasMore, loadMoreAtTop }: Props = $props();

	const runs = $derived(mergeBySpeaker(utterances));

	function fmt(t: number): string {
		if (!Number.isFinite(t)) return '';
		const m = Math.floor(t / 60);
		const s = Math.floor(t % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}
</script>

<aside class="ctx-panel" aria-label={label}>
	{#if state === 'loading'}
		<div class="status loading">{t('loadingContext')}</div>
	{:else if state === 'error'}
		<div class="status error">{t('contextUnavailable')}</div>
	{:else if state === 'empty' || runs.length === 0}
		<div class="status muted">{t('noContext')}</div>
	{:else}
		{#if hasMore && onLoadMore && loadMoreAtTop}
			<button type="button" class="load-more-btn top" onclick={onLoadMore} title={t('loadMoreContext')}>
				↑ {t('loadMoreContext')}
			</button>
		{/if}
		<ol>
			{#each runs as run (run.parts[0].utterance_id)}
				<li class:same-speaker={run.same_speaker_as_current}>
					<div class="meta">
						<span class="speaker" title={run.speaker_label ?? ''}>
							{run.speaker_name ?? run.speaker_label ?? '—'}
						</span>
						<span class="time">{fmt(run.start)}</span>
					</div>
					<div class="text">{run.text}</div>
				</li>
			{/each}
		</ol>
		{#if hasMore && onLoadMore && !loadMoreAtTop}
			<button type="button" class="load-more-btn bottom" onclick={onLoadMore} title={t('loadMoreContext')}>
				↓ {t('loadMoreContext')}
			</button>
		{/if}
	{/if}
</aside>

<style>
	.ctx-panel {
		border: 1px solid var(--border, #e2e8f0);
		border-radius: var(--radius, 10px);
		background: var(--surface-2, #f8fafc);
		padding: 0.5rem 0.75rem;
		font-size: 0.82rem;
		line-height: 1.45;
	}
	.status {
		color: var(--text-3, #94a3b8);
		font-style: italic;
		padding: 0.2rem 0;
	}
	.status.error { color: #b91c1c; font-style: normal; }
	.status.muted { font-size: 0.78rem; }

	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}
	li {
		padding: 0.35rem 0.5rem;
		border-left: 2px solid transparent;
		border-radius: 4px;
		background: var(--surface, #fff);
	}
	li.same-speaker { border-left-color: var(--border-accent, #93c5fd); }
	.meta {
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
		color: var(--text-3, #94a3b8);
		font-size: 0.72rem;
		margin-bottom: 0.1rem;
	}
	.speaker { font-weight: 600; color: var(--text-2, #475569); }
	.time { font-variant-numeric: tabular-nums; }
	.text { color: var(--text, #0f172a); white-space: pre-wrap; word-break: break-word; }

	.load-more-btn {
		display: block;
		width: 100%;
		padding: 0.18rem 0.4rem;
		font-size: 0.68rem;
		color: var(--text-3, #94a3b8);
		background: none;
		border: 1px dashed var(--border, #e2e8f0);
		border-radius: 4px;
		cursor: pointer;
		text-align: center;
		font-family: inherit;
		transition: color 0.15s, border-color 0.15s;
	}
	.load-more-btn:hover { color: var(--text-2, #475569); border-color: var(--text-3, #94a3b8); }
	.load-more-btn.top { margin-bottom: 0.35rem; }
	.load-more-btn.bottom { margin-top: 0.35rem; }
</style>

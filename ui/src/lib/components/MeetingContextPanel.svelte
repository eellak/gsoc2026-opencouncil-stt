<script lang="ts">
	import { t } from '$lib/i18n.svelte';
	import type { ContextUtterance } from '$lib/domain/meeting-context';
	import { mergeBySpeaker } from '$lib/client/meeting-context.svelte';

	interface Props {
		utterances: ContextUtterance[];
		label: string;
		state: 'loading' | 'ready' | 'error' | 'empty';
	}
	const { utterances, label, state }: Props = $props();

	const runs = $derived(mergeBySpeaker(utterances));

	function fmt(t: number): string {
		if (!Number.isFinite(t)) return '';
		const m = Math.floor(t / 60);
		const s = Math.floor(t % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}
</script>

<aside class="ctx-panel" aria-label={label}>
	<header>{label}</header>
	{#if state === 'loading'}
		<div class="status loading">{t('loadingContext')}</div>
	{:else if state === 'error'}
		<div class="status error">{t('contextUnavailable')}</div>
	{:else if state === 'empty' || runs.length === 0}
		<div class="status muted">{t('noContext')}</div>
	{:else}
		<ol>
			{#each runs as run (run.parts[0].utterance_id)}
				<li class:same-speaker={run.same_speaker_as_current}>
					<div class="meta">
						<span class="speaker" title={run.speaker_label ?? ''}>
							{run.speaker_name ?? run.speaker_label ?? '—'}
						</span>
						<span class="time">{fmt(run.start)}</span>
						{#if run.parts.length > 1}
							<span class="merged-badge" title="{run.parts.length} consecutive utterances merged">×{run.parts.length}</span>
						{/if}
					</div>
					<div class="text">{run.text}</div>
				</li>
			{/each}
		</ol>
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
	header {
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-3, #94a3b8);
		margin-bottom: 0.3rem;
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
	.merged-badge {
		font-size: 0.65rem;
		padding: 0.05rem 0.3rem;
		border-radius: 999px;
		background: var(--surface-3, #f1f5f9);
		color: var(--text-3, #94a3b8);
		font-variant-numeric: tabular-nums;
	}
	.text { color: var(--text, #0f172a); white-space: pre-wrap; word-break: break-word; }
</style>

<script lang="ts">
	import DiffMatchPatch from 'diff-match-patch';
	import type { Snippet } from 'svelte';
	import { TAXONOMY_MAP, normalizeTaxonomyId, type TaxonomyEntry } from '$lib/shared/taxonomy';
	import type { Lang } from '$lib/shared/taxonomy';

	interface Props {
		before?: string;
		after?: string;
		speakerName?: string | null;
		/** Whether the meeting-context transcript fetch is still in flight. */
		speakerLoading?: boolean;
		errorCategoryIds?: readonly string[];
		lang?: Lang;
		/** Snippet rendered in the centre of the header (play button). */
		playSlot?: Snippet;
	}
	const {
		before = '',
		after = '',
		speakerName = null,
		speakerLoading = false,
		errorCategoryIds = [],
		lang = 'el',
		playSlot
	}: Props = $props();

	const dmp = new DiffMatchPatch();

	// [op, text]: op -1=delete, 0=equal, 1=insert
	const diffs = $derived.by(() => {
		const d = dmp.diff_main(before, after);
		dmp.diff_cleanupSemantic(d);
		return d;
	});

	const resolvedCategories = $derived(
		errorCategoryIds
			.map((id) => normalizeTaxonomyId(id))
			.filter((id): id is NonNullable<typeof id> => id !== null)
			.map((id) => TAXONOMY_MAP[id])
			.filter((e): e is TaxonomyEntry => !!e)
	);
</script>

<div class="diff has-header">
	<div class="diff-header">
		<div
			class="diff-speaker"
			aria-label="Speaker"
			class:placeholder={!speakerName}
			title={speakerName ?? ''}
		>
			{#if speakerName}
				<span class="mic" aria-hidden="true">🎙</span>
				<span class="speaker-name">{speakerName}</span>
			{:else if speakerLoading}
				<span class="dot-shimmer" aria-hidden="true">🎙 …</span>
			{:else}
				🎙 —
			{/if}
		</div>
		{#if playSlot}
			<div class="diff-play">{@render playSlot()}</div>
		{/if}
		<div class="diff-categories">
			{#each resolvedCategories as entry (entry.id)}
				<button type="button" class="cat-chip group-{entry.group}">
					<span class="cat-label">{entry[lang]}</span>
					<span class="cat-popover" role="tooltip">
						<strong>{entry[lang]}</strong>
						<span class="cat-example">
							<mark class="del">{entry.example_before}</mark>
							<span class="arrow">→</span>
							<mark class="ins">{entry.example_after}</mark>
						</span>
					</span>
				</button>
			{/each}
		</div>
	</div>
	<div class="diff-block before">
		{#each diffs as [op, text]}
			{#if op === -1}
				<mark class="del">{text}</mark>
			{:else if op === 0}
				<span>{text}</span>
			{/if}
		{/each}
	</div>
	<div class="diff-block after">
		{#each diffs as [op, text]}
			{#if op === 1}
				<mark class="ins">{text}</mark>
			{:else if op === 0}
				<span>{text}</span>
			{/if}
		{/each}
	</div>
</div>

<style>
	.diff {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		gap: 0.5rem;
		font-size: 1rem;
		line-height: 1.6;
		min-width: 0;
	}
	.diff.has-header {
		grid-template-areas: 'header header' 'before after';
	}
	.diff.has-header .diff-header { grid-area: header; }
	.diff.has-header .diff-block.before { grid-area: before; }
	.diff.has-header .diff-block.after { grid-area: after; }

	.diff-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
		/* Reserve a stable strip so the layout doesn't jump when the speaker
		   resolves or categories arrive. */
		min-height: 2.1rem;
	}

	.diff-speaker {
		font-size: 0.78rem;
		font-weight: 600;
		color: var(--text-2, #475569);
		padding: 0.25rem 0.5rem;
		background: #ede9fe;
		border-radius: 6px;
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		max-width: min(220px, 50%);
		min-width: 0;
		cursor: help;
	}
	.diff-speaker .mic { flex-shrink: 0; }
	.diff-speaker .speaker-name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}
	.diff-speaker.placeholder {
		background: #f1f5f9;
		color: var(--text-3, #94a3b8);
	}
	.dot-shimmer {
		display: inline-block;
		animation: speaker-shimmer 1.2s ease-in-out infinite;
	}
	@keyframes speaker-shimmer {
		0%, 100% { opacity: 0.55; }
		50% { opacity: 0.95; }
	}

	.diff-play {
		display: inline-flex;
		align-items: center;
		/* Sits between speaker (left) and categories (right). */
	}

	.diff-categories {
		display: flex;
		gap: 0.3rem;
		flex-wrap: wrap;
		margin-left: auto;
	}

	/* Category chip */
	.cat-chip {
		position: relative;
		display: inline-flex;
		align-items: center;
		font-size: 0.7rem;
		font-weight: 500;
		padding: 0.18rem 0.48rem;
		border-radius: 999px;
		cursor: default;
		outline-offset: 2px;
		max-width: 11rem;
		border: none;
	}
	.cat-chip:focus-visible { outline: 2px solid #3b82f6; }

	.cat-label {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	/* Group tints */
	.cat-chip.group-phonetic        { background: #dbeafe; color: #1e40af; }
	.cat-chip.group-morphological   { background: #ede9fe; color: #4c1d95; }
	.cat-chip.group-named_entity    { background: #fef3c7; color: #78350f; }
	.cat-chip.group-formatting      { background: #f1f5f9; color: #334155; }
	.cat-chip.group-meta            { background: #fee2e2; color: #7f1d1d; }

	/* Popover */
	.cat-popover {
		position: absolute;
		top: calc(100% + 4px);
		right: 0;
		z-index: 30;
		min-width: 13rem;
		max-width: 18rem;
		background: #fff;
		border: 1px solid var(--border, #e2e8f0);
		border-radius: 8px;
		box-shadow: 0 4px 16px rgba(0,0,0,0.12);
		padding: 0.5rem 0.65rem;
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		opacity: 0;
		pointer-events: none;
		transition: opacity 120ms ease;
		font-size: 0.78rem;
		color: var(--text, #0f172a);
		white-space: normal;
	}
	.cat-chip:hover .cat-popover,
	.cat-chip:focus-within .cat-popover {
		opacity: 1;
		pointer-events: auto;
	}

	.cat-popover strong { font-weight: 600; display: block; }

	.cat-example {
		display: flex;
		align-items: baseline;
		gap: 0.3rem;
		flex-wrap: wrap;
	}
	.cat-example .arrow { color: var(--text-3, #94a3b8); font-size: 0.72rem; }

	/* Diff blocks */
	.diff-block {
		padding: 0.75rem 1rem;
		border-radius: 6px;
		background: var(--surface2, #f5f5f5);
		word-break: break-word;
		overflow-wrap: anywhere;
		white-space: pre-wrap;
		min-width: 0;
		max-width: 100%;
	}

	mark {
		border-radius: 3px;
		padding: 0 2px;
	}

	mark.del {
		background: #ffd7d7;
		color: #900;
		text-decoration: line-through;
	}

	mark.ins {
		background: #d7ffd7;
		color: #060;
	}

	@media (max-width: 640px) {
		.diff {
			grid-template-columns: 1fr;
		}
		.diff.has-header {
			grid-template-areas: 'header' 'before' 'after';
		}
		.diff-categories { margin-left: 0; width: 100%; }
	}
</style>

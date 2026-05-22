<script lang="ts">
	import DiffMatchPatch from 'diff-match-patch';

	interface Props {
		before?: string;
		after?: string;
		/**
		 * Optional resolved speaker name (or SPEAKER_N label fallback) for the
		 * utterance these diff blocks belong to. When provided, renders a
		 * compact header above both blocks so the reviewer sees who was
		 * speaking next to the correction instead of having to read the top
		 * meta bar.
		 */
		speakerName?: string | null;
	}
	const { before = '', after = '', speakerName = null }: Props = $props();

	const dmp = new DiffMatchPatch();

	// [op, text]: op -1=delete, 0=equal, 1=insert
	const diffs = $derived.by(() => {
		const d = dmp.diff_main(before, after);
		dmp.diff_cleanupSemantic(d);
		return d;
	});
</script>

<div class="diff" class:has-speaker={!!speakerName}>
	{#if speakerName}
		<div class="diff-speaker" aria-label="Speaker">🎙 {speakerName}</div>
	{/if}
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
		grid-template-columns: 1fr 1fr;
		gap: 0.5rem;
		font-size: 1rem;
		line-height: 1.6;
	}
	.diff.has-speaker {
		grid-template-areas: 'speaker speaker' 'before after';
	}
	.diff.has-speaker .diff-speaker { grid-area: speaker; }
	.diff.has-speaker .diff-block.before { grid-area: before; }
	.diff.has-speaker .diff-block.after { grid-area: after; }

	.diff-speaker {
		font-size: 0.78rem;
		font-weight: 600;
		color: var(--text-2, #475569);
		padding: 0.25rem 0.5rem;
		background: #ede9fe;
		border-radius: 6px;
		display: inline-flex;
		align-self: start;
		justify-self: start;
		max-width: 100%;
	}

	.diff-block {
		padding: 0.75rem 1rem;
		border-radius: 6px;
		background: var(--surface2, #f5f5f5);
		word-break: break-word;
		white-space: pre-wrap;
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
</style>

<script lang="ts">
	import { t } from '$lib/i18n.svelte';
	import type { IncludeStatus } from '$lib/domain/types';

	interface Props {
		counts: Record<IncludeStatus, number>;
		/** When set, each clickable pill includes this seed in its review href. */
		seed?: number | null;
		/** Layout density. `compact` is for the home page row. */
		variant?: 'compact' | 'full';
	}

	const { counts, seed = null, variant = 'compact' }: Props = $props();

	const total = $derived(
		counts.include + counts.exclude + counts.uncertain + counts.unreviewed
	);

	function pct(n: number): string {
		if (!total) return '0%';
		const v = (n / total) * 100;
		return v >= 10 ? `${v.toFixed(0)}%` : `${v.toFixed(1)}%`;
	}

	const order: IncludeStatus[] = ['include', 'exclude', 'uncertain', 'unreviewed'];
	const clickable: ReadonlySet<IncludeStatus> = new Set([
		'include',
		'exclude',
		'uncertain'
	]);

	function listHref(status: IncludeStatus): string {
		return `/stats/by-status/${status}`;
	}
	function queueHref(status: IncludeStatus): string {
		const params = new URLSearchParams({ status });
		if (seed != null) params.set('seed', String(seed));
		return `/?${params.toString()}`;
	}
</script>

<div class="dist {variant}">
	{#each order as status (status)}
		{@const n = counts[status] ?? 0}
		{@const isClickable = clickable.has(status)}
		{#if isClickable}
			<div class="pill {status}">
				<a class="pill-link" href={listHref(status)} title={t('openListForStatus')}>
					<span class="label">{t(status)}</span>
					<span class="count">{n.toLocaleString('el-GR')}</span>
					<span class="pct">{pct(n)}</span>
				</a>
				<a
					class="queue-arrow"
					href={queueHref(status)}
					title={t('openFilteredQueue')}
					aria-label={t('openFilteredQueue')}>▸</a
				>
			</div>
		{:else}
			<div class="pill {status} muted">
				<span class="label">{t(status)}</span>
				<span class="count">{n.toLocaleString('el-GR')}</span>
				<span class="pct">{pct(n)}</span>
			</div>
		{/if}
	{/each}
</div>

<style>
	.dist {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}
	.pill {
		display: inline-flex;
		align-items: center;
		gap: 0.45rem;
		padding: 0.4rem 0.75rem;
		border-radius: 999px;
		font-size: 0.85rem;
		text-decoration: none;
		border: 1px solid transparent;
		color: inherit;
		background: var(--surface-2, #f8fafc);
		transition: filter 0.15s ease, transform 0.05s ease;
	}
	.pill-link {
		display: inline-flex;
		align-items: center;
		gap: 0.45rem;
		color: inherit;
		text-decoration: none;
	}
	.pill-link:hover {
		filter: brightness(0.95);
	}
	.pill-link:active {
		transform: translateY(1px);
	}
	.pill.muted {
		opacity: 0.75;
		cursor: default;
	}
	.label {
		font-weight: 600;
	}
	.count {
		font-variant-numeric: tabular-nums;
	}
	.pct {
		font-size: 0.72rem;
		opacity: 0.75;
		font-variant-numeric: tabular-nums;
	}
	.queue-arrow {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-width: 1.4rem;
		height: 1.4rem;
		border-radius: 999px;
		font-size: 0.85rem;
		background: rgba(255, 255, 255, 0.55);
		color: inherit;
		text-decoration: none;
		margin-left: 0.1rem;
	}
	.queue-arrow:hover {
		background: rgba(255, 255, 255, 0.9);
	}

	.pill.include {
		background: #dcfce7;
		color: #14532d;
	}
	.pill.exclude {
		background: #fee2e2;
		color: #7f1d1d;
	}
	.pill.uncertain {
		background: #fef3c7;
		color: #78350f;
	}
	.pill.unreviewed {
		background: #f1f5f9;
		color: #475569;
	}

	.dist.full .pill {
		padding: 0.55rem 1rem;
		font-size: 0.95rem;
	}
</style>

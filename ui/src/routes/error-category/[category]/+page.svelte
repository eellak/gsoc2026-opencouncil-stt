<script lang="ts">
	import type { PageData } from './$types';
	import { getLang, t } from '$lib/i18n.svelte';
	import { reviewHref, randomSeed } from '$lib/shared/urls';

	const { data }: { data: PageData } = $props();
	const lang = $derived(getLang());

	// If the user arrived from /stats without a seed, generate one once so the
	// Open links land on the deterministic queue from the very first click.
	// Must be `$state`, not `$derived`: a $derived would recompute on every
	// reactive tick and hand back a different random seed each time.
	const generatedSeed = randomSeed();
	const activeSeed = $derived(data.seed ?? generatedSeed);

	const totalPages = $derived(Math.ceil(data.total / data.page_size));

	function pageUrl(p: number) {
		const params = new URLSearchParams();
		params.set('page', String(p));
		if (data.seed !== null) params.set('seed', String(data.seed));
		return `?${params.toString()}`;
	}
</script>

<div class="page">
	<header>
		<a href="/stats" class="back">← {t('statsLink')}</a>
		<div class="title-row">
			<h1>
				<span class="badge">{data.taxonomy ? data.taxonomy[lang] : data.category}</span>
				<code class="key">{data.category}</code>
			</h1>
			<span class="count">{t('totalItems', { n: data.total.toLocaleString(lang === 'el' ? 'el-GR' : 'en-US') })}</span>
			{#if data.items.length > 0}
				<a
					class="play-through"
					href="/review/{encodeURIComponent(data.items[0].utterance_id)}?errorCategory={encodeURIComponent(data.category)}"
					title={t('playThroughTitle')}
				>▶ {t('playThroughLabel')}</a>
			{/if}
		</div>
		{#if data.taxonomy}
			<p class="example">
				<span class="ex-before">{data.taxonomy.example_before}</span>
				→
				<span class="ex-after">{data.taxonomy.example_after}</span>
			</p>
		{/if}
	</header>

	{#if data.items.length === 0}
		<p class="empty">{t('noItems')}</p>
	{:else}
		<ul class="rows">
			{#each data.items as item}
				<li>
					<a
						class="row"
						href={reviewHref({ utterance_id: item.utterance_id, seed: activeSeed })}
					>
						<div class="meta">
							{#if item.meeting_name}<span class="meeting" title={item.meeting_name}>{item.meeting_name}</span>{/if}
							{#if item.meeting_date}<span class="date">{item.meeting_date}</span>{/if}
							{#if item.all_categories.length > 1}
								<span class="extra-cats" title={item.all_categories.join(', ')}>+{item.all_categories.length - 1}</span>
							{/if}
						</div>
						<div class="diffs">
							<div class="before">{item.before}</div>
							<div class="after">{item.after}</div>
						</div>
					</a>
				</li>
			{/each}
		</ul>

		{#if totalPages > 1}
			<nav class="pagination">
				{#if data.page > 1}
					<a href={pageUrl(data.page - 1)} class="pg-btn">{t('prev')}</a>
				{/if}
				<span class="pg-info">{data.page} / {totalPages}</span>
				{#if data.page < totalPages}
					<a href={pageUrl(data.page + 1)} class="pg-btn">{t('next')}</a>
				{/if}
			</nav>
		{/if}
	{/if}
</div>

<style>
	.page { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
	header { margin-bottom: 1.5rem; }
	.back { font-size: 0.85rem; color: #6b7280; text-decoration: none; }
	.back:hover { color: #1e40af; }
	.title-row { display: flex; align-items: baseline; gap: 1rem; margin: 0.5rem 0; flex-wrap: wrap; }
	h1 { margin: 0; font-size: 1.4rem; display: flex; align-items: center; gap: 0.5rem; }
	.badge { padding: 0.2rem 0.6rem; border-radius: 999px; background: #dbeafe; color: #1e40af; font-size: 0.9rem; font-weight: 600; }
	.key { font-size: 0.8rem; color: #6b7280; background: #f3f4f6; padding: 0.1rem 0.4rem; border-radius: 4px; }
	.count { color: #6b7280; font-size: 0.9rem; margin-left: auto; }
	.play-through {
		font-size: 0.85rem; font-weight: 600; padding: 0.3rem 0.7rem;
		border-radius: 6px; background: #dcfce7; color: #166534; text-decoration: none;
	}
	.play-through:hover { background: #bbf7d0; }
	.example { margin: 0.5rem 0 0; font-size: 0.85rem; color: #374151; font-family: serif; }
	.ex-before { color: #7f1d1d; }
	.ex-after { color: #14532d; }
	.rows { list-style: none; padding: 0; margin: 1rem 0 0; display: flex; flex-direction: column; gap: 0.5rem; }
	.row {
		display: flex; flex-direction: column; gap: 0.5rem;
		padding: 0.75rem 0.9rem;
		background: #fff;
		border: 1px solid #e5e7eb;
		border-radius: 8px;
		text-decoration: none; color: inherit;
		transition: border-color 0.15s, box-shadow 0.15s;
	}
	.row:hover { border-color: #93c5fd; box-shadow: 0 2px 8px rgba(37,99,235,.08); }
	.meta {
		display: flex; flex-wrap: wrap; gap: 0.3rem 0.6rem;
		font-size: 0.74rem; color: #475569;
	}
	.meta .meeting {
		max-width: 22rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
		color: #1e3a8a; font-weight: 500;
	}
	.meta .date { color: #64748b; font-variant-numeric: tabular-nums; }
	.meta .extra-cats { background: #f1f5f9; color: #475569; padding: 0.05rem 0.4rem; border-radius: 4px; }
	.diffs {
		display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;
		font-size: 0.88rem; line-height: 1.45;
	}
	.before { background: #fef2f2; padding: 0.4rem 0.55rem; border-radius: 5px; color: #7f1d1d; white-space: pre-wrap; word-break: break-word; }
	.after  { background: #f0fdf4; padding: 0.4rem 0.55rem; border-radius: 5px; color: #14532d; white-space: pre-wrap; word-break: break-word; }
	.empty { color: #9ca3af; padding: 2rem; text-align: center; }
	.pagination { display: flex; gap: 0.75rem; align-items: center; margin-top: 1.5rem; justify-content: center; }
	.pg-btn { font-size: 0.85rem; padding: 0.3rem 0.8rem; background: #f0f0f0; border-radius: 4px; text-decoration: none; color: #374151; }
	.pg-btn:hover { background: #dbeafe; color: #1e40af; }
	.pg-info { font-size: 0.85rem; color: #6b7280; }

	@media (max-width: 640px) {
		.page { padding: 1rem 0.75rem; }
		h1 { font-size: 1.1rem; }
		.diffs { grid-template-columns: 1fr; }
		.meta .meeting { max-width: 100%; }
	}
</style>

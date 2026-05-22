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
		<table>
			<thead>
				<tr>
					<th>{t('meeting')}</th>
					<th>{t('before')}</th>
					<th>{t('after')}</th>
					<th>{t('byCategory')}</th>
					<th></th>
				</tr>
			</thead>
			<tbody>
				{#each data.items as item}
					<tr>
						<td class="meta">
							{item.meeting_name ?? '—'}
							{#if item.meeting_date}<small>{item.meeting_date}</small>{/if}
						</td>
						<td class="text before">{item.before}</td>
						<td class="text after">{item.after}</td>
						<td class="meta small">{item.all_categories.join(', ')}</td>
						<td class="action">
							<a
								href={reviewHref({ utterance_id: item.utterance_id, seed: activeSeed })}
								class="view-btn"
							>{t('openInReview')}</a>
						</td>
					</tr>
				{/each}
			</tbody>
		</table>

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
	.example { margin: 0.5rem 0 0; font-size: 0.85rem; color: #374151; font-family: serif; }
	.ex-before { color: #7f1d1d; }
	.ex-after { color: #14532d; }
	table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 1rem; }
	thead th { text-align: left; padding: 0.4rem 0.5rem; border-bottom: 2px solid #e5e7eb; color: #6b7280; font-weight: 600; }
	tbody tr:hover { background: #f9fafb; }
	td { padding: 0.35rem 0.5rem; border-bottom: 1px solid #f3f4f6; vertical-align: top; }
	td.text { max-width: 300px; white-space: pre-wrap; word-break: break-word; }
	td.before { color: #7f1d1d; }
	td.after { color: #14532d; }
	td.meta { color: #6b7280; }
	td.meta small { display: block; font-size: 0.72rem; color: #94a3b8; }
	td.small { font-size: 0.72rem; }
	.view-btn { font-size: 0.78rem; padding: 0.15rem 0.5rem; background: #f0f0f0; border-radius: 4px; text-decoration: none; color: #374151; white-space: nowrap; }
	.view-btn:hover { background: #dbeafe; color: #1e40af; }
	.empty { color: #9ca3af; padding: 2rem; text-align: center; }
	.pagination { display: flex; gap: 0.75rem; align-items: center; margin-top: 1.5rem; justify-content: center; }
	.pg-btn { font-size: 0.85rem; padding: 0.3rem 0.8rem; background: #f0f0f0; border-radius: 4px; text-decoration: none; color: #374151; }
	.pg-btn:hover { background: #dbeafe; color: #1e40af; }
	.pg-info { font-size: 0.85rem; color: #6b7280; }
</style>

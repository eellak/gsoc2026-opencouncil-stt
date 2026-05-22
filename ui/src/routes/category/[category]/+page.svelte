<script lang="ts">
	import type { PageData } from './$types';
	import { page as pageState } from '$app/state';
	import { editHref } from '$lib/shared/urls';
	const { data }: { data: PageData } = $props();

	const totalPages = $derived(Math.ceil(data.total / data.page_size));
	const seedQs = $derived(pageState.url.searchParams.get('seed') ?? '');

	function pageUrl(p: number) {
		const params = new URLSearchParams();
		params.set('page', String(p));
		if (seedQs) params.set('seed', seedQs);
		return `?${params.toString()}`;
	}
</script>

<div class="page">
	<header>
		<a href="/stats" class="back">← Στατιστικά</a>
		<div class="title-row">
			<h1>
				<span class="badge" class:rejected={data.meta.is_rejected}>{data.meta.label_el}</span>
				<code class="key">{data.meta.key}</code>
			</h1>
			<span class="count">{data.total.toLocaleString('el-GR')} εγγραφές</span>
		</div>
		<p class="reason">{data.meta.reason_el}</p>
	</header>

	{#if data.items.length === 0}
		<p class="empty">Καμία εγγραφή σε αυτή την κατηγορία.</p>
	{:else}
		<table>
			<thead>
				<tr>
					<th>before_text</th>
					<th>after_text</th>
					<th>editor</th>
					<th>cleaning</th>
					<th></th>
				</tr>
			</thead>
			<tbody>
				{#each data.items as item}
					<tr>
						<td class="text before">{item.before_text?.slice(0, 120) ?? ''}</td>
						<td class="text after">{item.after_text?.slice(0, 120) ?? ''}</td>
						<td class="meta">{item.edited_by ?? '—'}</td>
						<td class="meta small">{item.cleaning_applied || '—'}</td>
						<td class="action">
							<a href={editHref(item.edit_id, seedQs ? Number(seedQs) : undefined)} class="view-btn">Άνοιγμα</a>
						</td>
					</tr>
				{/each}
			</tbody>
		</table>

		{#if totalPages > 1}
			<nav class="pagination">
				{#if data.page > 1}
					<a href={pageUrl(data.page - 1)} class="pg-btn">← Προηγ.</a>
				{/if}
				<span class="pg-info">Σελ. {data.page} / {totalPages}</span>
				{#if data.page < totalPages}
					<a href={pageUrl(data.page + 1)} class="pg-btn">Επόμ. →</a>
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
	.badge.rejected { background: #fee2e2; color: #b91c1c; }
	.key { font-size: 0.8rem; color: #6b7280; background: #f3f4f6; padding: 0.1rem 0.4rem; border-radius: 4px; }
	.count { color: #6b7280; font-size: 0.9rem; margin-left: auto; }
	.reason { margin: 0.5rem 0 0; color: #374151; font-size: 0.9rem; line-height: 1.5; max-width: 700px; }

	table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 1rem; }
	thead th { text-align: left; padding: 0.4rem 0.5rem; border-bottom: 2px solid #e5e7eb; color: #6b7280; font-weight: 600; }
	tbody tr:hover { background: #f9fafb; }
	td { padding: 0.35rem 0.5rem; border-bottom: 1px solid #f3f4f6; vertical-align: top; }
	td.text { max-width: 350px; white-space: pre-wrap; word-break: break-word; }
	td.before { color: #7f1d1d; }
	td.after { color: #14532d; }
	td.meta { color: #6b7280; white-space: nowrap; }
	td.small { font-size: 0.75rem; }
	.view-btn { font-size: 0.78rem; padding: 0.15rem 0.5rem; background: #f0f0f0; border-radius: 4px; text-decoration: none; color: #374151; white-space: nowrap; }
	.view-btn:hover { background: #dbeafe; color: #1e40af; }

	.empty { color: #9ca3af; padding: 2rem; text-align: center; }
	.pagination { display: flex; gap: 0.75rem; align-items: center; margin-top: 1.5rem; justify-content: center; }
	.pg-btn { font-size: 0.85rem; padding: 0.3rem 0.8rem; background: #f0f0f0; border-radius: 4px; text-decoration: none; color: #374151; }
	.pg-btn:hover { background: #dbeafe; color: #1e40af; }
	.pg-info { font-size: 0.85rem; color: #6b7280; }
</style>

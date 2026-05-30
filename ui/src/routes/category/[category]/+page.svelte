<script lang="ts">
	import type { PageData } from './$types';
	import { page as pageState } from '$app/state';
	import { editHref } from '$lib/shared/urls';
	const { data }: { data: PageData } = $props();

	const totalPages = $derived(Math.ceil(data.total / data.page_size));
	const seedQs = $derived(pageState.url.searchParams.get('seed') ?? '');
	// Only treat seedQs as a real seed when it's a non-negative integer string.
	// Number("abc") would otherwise hand NaN to editHref.
	const validatedSeed = $derived(/^\d+$/.test(seedQs) ? Number(seedQs) : undefined);

	function pageUrl(p: number) {
		const params = new URLSearchParams();
		params.set('page', String(p));
		if (validatedSeed !== undefined) params.set('seed', String(validatedSeed));
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
			{#if data.items.length > 0}
				<a
					class="play-through"
					href="/review/{encodeURIComponent(data.items[0].utterance_id)}?category={encodeURIComponent(data.category)}"
					title="Άνοιγμα στη ροή αξιολόγησης — με autoplay τα ακούς το ένα μετά το άλλο"
				>▶ Αξιολόγηση αυτών</a>
			{/if}
		</div>
		<p class="reason">{data.meta.reason_el}</p>
	</header>

	{#if data.items.length === 0}
		<p class="empty">Καμία εγγραφή σε αυτή την κατηγορία.</p>
	{:else}
		<ul class="rows">
			{#each data.items as item}
				<li>
					<a class="row" href={editHref(item.edit_id, validatedSeed)}>
						<div class="meta">
							{#if item.edited_by}<span class="editor">{item.edited_by}</span>{/if}
							{#if item.cleaning_applied}<span class="cleaning">{item.cleaning_applied}</span>{/if}
						</div>
						<div class="diffs">
							<div class="before">{item.before_text?.slice(0, 200) ?? ''}</div>
							<div class="after">{item.after_text?.slice(0, 200) ?? ''}</div>
						</div>
					</a>
				</li>
			{/each}
		</ul>

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
	.play-through {
		font-size: 0.85rem; font-weight: 600; padding: 0.3rem 0.7rem;
		border-radius: 6px; background: #dcfce7; color: #166534; text-decoration: none;
	}
	.play-through:hover { background: #bbf7d0; }
	.reason { margin: 0.5rem 0 0; color: #374151; font-size: 0.9rem; line-height: 1.5; max-width: 700px; }

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
	.meta { display: flex; flex-wrap: wrap; gap: 0.3rem 0.6rem; font-size: 0.74rem; color: #475569; }
	.meta .editor { color: #1e3a8a; font-weight: 500; }
	.meta .cleaning { background: #f1f5f9; padding: 0.05rem 0.4rem; border-radius: 4px; color: #475569; font-size: 0.7rem; }
	.diffs { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.88rem; line-height: 1.45; }
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
	}
</style>

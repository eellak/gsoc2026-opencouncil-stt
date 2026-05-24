<script lang="ts">
	import { t } from '$lib/i18n.svelte';
	import type { PageData } from './$types';

	const { data }: { data: PageData } = $props();
</script>

<div class="page">
	<header>
		<a class="back" href="/stats">{t('backToStats')}</a>
		<h1>
			<span class="badge {data.status}">{t(data.status)}</span>
			<span class="count">{data.total.toLocaleString('el-GR')}</span>
		</h1>
	</header>

	{#if data.rows.length === 0}
		<p class="empty">{t('noItems')}</p>
	{:else}
		<ul class="rows">
			{#each data.rows as row (row.utterance_id)}
				<li>
					<a class="row" href={`/review/${encodeURIComponent(row.utterance_id)}`}>
						<div class="meta">
							{#if row.meeting_name}
								<span class="meeting" title={row.meeting_name}>{row.meeting_name}</span>
							{/if}
							{#if row.city_id}<span class="city">{row.city_id}</span>{/if}
							{#if row.meeting_date}<span class="date">{row.meeting_date}</span>{/if}
							{#if row.edited_by}<span class="editor">{row.edited_by}</span>{/if}
						</div>
						<div class="diffs">
							<div class="before">{row.before_preview}</div>
							<div class="after">{row.after_preview}</div>
						</div>
					</a>
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.page { max-width: 960px; margin: 0 auto; padding: 1.25rem; }
	header { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1rem; }
	.back { font-size: 0.85rem; color: #2563eb; text-decoration: none; }
	.back:hover { text-decoration: underline; }
	h1 { font-size: 1.2rem; display: flex; align-items: center; gap: 0.6rem; margin: 0; }
	.badge { padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
	.badge.include { background: #dcfce7; color: #14532d; }
	.badge.exclude { background: #fee2e2; color: #7f1d1d; }
	.badge.uncertain { background: #fef3c7; color: #78350f; }
	.count { color: #6b7280; font-variant-numeric: tabular-nums; font-weight: 500; }
	.empty { color: #6b7280; padding: 2rem; text-align: center; }
	.rows { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.5rem; }
	.row {
		display: flex; flex-direction: column; gap: 0.5rem;
		padding: 0.75rem 0.9rem;
		background: #fff;
		border: 1px solid #e5e7eb;
		border-radius: 8px;
		text-decoration: none;
		color: inherit;
		transition: border-color 0.15s, box-shadow 0.15s;
	}
	.row:hover { border-color: #93c5fd; box-shadow: 0 2px 8px rgba(37,99,235,.08); }
	.meta {
		display: flex; flex-wrap: wrap; gap: 0.3rem 0.6rem;
		font-size: 0.74rem; color: #475569;
	}
	.meta .meeting {
		max-width: 22rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		color: #1e3a8a;
		font-weight: 500;
	}
	.meta .city { background: #ecfdf5; color: #047857; padding: 0.05rem 0.4rem; border-radius: 4px; }
	.meta .date { color: #64748b; font-variant-numeric: tabular-nums; }
	.meta .editor { color: #64748b; }
	.diffs {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.5rem;
		font-size: 0.88rem;
		line-height: 1.45;
	}
	.before { background: #fef2f2; padding: 0.4rem 0.55rem; border-radius: 5px; color: #7f1d1d; }
	.after { background: #f0fdf4; padding: 0.4rem 0.55rem; border-radius: 5px; color: #14532d; }

	@media (max-width: 640px) {
		.diffs { grid-template-columns: 1fr; }
		.meta .meeting { max-width: 100%; }
	}
</style>

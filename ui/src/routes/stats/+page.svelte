<script lang="ts">
	import type { PageData } from './$types';
	import { getLang, t } from '$lib/i18n.svelte';
	import { taxonomyLabel } from '$lib/shared/taxonomy';
	import { errorCategoryHref } from '$lib/shared/urls';
	import StatusDistribution from '$lib/components/StatusDistribution.svelte';
	const { data }: { data: PageData } = $props();
	const stats = $derived(data.stats);
	const lang = $derived(getLang());

	function categoryDisplay(id: string | null): string {
		if (!id) return t('uncategorized');
		const lbl = taxonomyLabel(id, lang);
		return lbl || id;
	}
</script>

<div class="stats-page">
	<header>
		<h1>Στατιστικά</h1>
		<div class="header-actions">
			<a href="/" class="btn">Αρχική</a>
			<a href="/api/export" class="btn export">Εξαγωγή included</a>
		</div>
	</header>

	<div class="grid">
		<section class="card">
			<h2>Σύνολο</h2>
			<p class="big-number">{stats.total.toLocaleString('el-GR')}</p>
		</section>

		<section class="card wide">
			<h2>Ανά κατάσταση</h2>
			<StatusDistribution counts={stats.by_status} variant="full" />
		</section>

		<section class="card">
			<h2>{t('statsCategoryAssignments')}</h2>
			<p class="card-note">{t('statsCategoryAssignmentsHint')}</p>
			<table>
				<tbody>
					{#each stats.by_category as row}
						<tr>
							<td>
								{#if row.category}
									<a href={errorCategoryHref(row.category)} class="cat-link">{categoryDisplay(row.category)}</a>
								{:else}
									<span class="cat-none">{categoryDisplay(null)}</span>
								{/if}
							</td>
							<td class="count">{row.count.toLocaleString('el-GR')}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>

		<section class="card">
			<h2>Ανά επεξεργαστή</h2>
			<table>
				<tbody>
					{#each stats.by_editor as row}
						<tr>
							<td>{row.edited_by ?? '(άγνωστος)'}</td>
							<td class="count">{row.count.toLocaleString('el-GR')}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>

		<section class="card">
			<h2>Ανά διάρκεια</h2>
			<table>
				<tbody>
					{#each stats.by_duration_bucket as row}
						<tr>
							<td>{row.bucket}</td>
							<td class="count">{row.count.toLocaleString('el-GR')}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>

		<section class="card">
			<h2>Ανά συνεδρίαση (top 20)</h2>
			<table>
				<tbody>
					{#each stats.by_meeting as row}
						<tr>
							<td class="ellipsis" title={row.meeting_name ?? ''}>{row.meeting_name ?? '(άγνωστη)'}</td>
							<td class="count">{row.count.toLocaleString('el-GR')}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>

		{#if stats.by_ingest_category?.length}
			<section class="card wide">
				<h2>Κατηγοριοποίηση εισαγωγής CSV</h2>
				<p class="card-note">Αυτόματη ταξινόμηση κατά την εισαγωγή. Κάντε κλικ για να δείτε παραδείγματα.</p>
				<table>
					<tbody>
						{#each stats.by_ingest_category as row}
							<tr class="cat-row" class:rejected={row.is_rejected === 1}>
								<td>
									<a href="/category/{row.ingest_category}" class="cat-link">
										{row.label_el ?? row.ingest_category ?? '(χωρίς κατηγορία)'}
									</a>
									{#if row.is_rejected}
										<span class="rejected-badge">αποκλείεται</span>
									{/if}
								</td>
								<td class="reason-cell">{row.reason_el ?? ''}</td>
								<td class="count">{row.count.toLocaleString('el-GR')}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</section>
		{/if}
	</div>
</div>

<style>
	.stats-page { max-width: 1000px; margin: 0 auto; padding: 1.5rem; }
	header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 0.5rem; }
	h1 { margin: 0; font-size: 1.5rem; }
	.header-actions { display: flex; gap: 0.5rem; }
	.btn { font-size: 0.85rem; padding: 0.3rem 0.7rem; border-radius: 4px; background: #f0f0f0; text-decoration: none; color: #333; }
	.btn.export { background: #dbeafe; color: #1e40af; }
	.btn:hover { filter: brightness(0.93); }
	.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
	.wide { grid-column: 1 / -1; }
	.card-note { margin: -0.25rem 0 0.75rem; font-size: 0.8rem; color: #6b7280; }
	.cat-row.rejected td:first-child { color: #b91c1c; }
	.cat-link { color: #1e40af; text-decoration: none; font-weight: 500; }
	.cat-link:hover { text-decoration: underline; }
	.cat-none { color: #94a3b8; font-style: italic; }
	.rejected-badge { margin-left: 0.4rem; font-size: 0.7rem; padding: 0.1rem 0.4rem; background: #fee2e2; color: #b91c1c; border-radius: 999px; vertical-align: middle; }
	.reason-cell { font-size: 0.78rem; color: #6b7280; max-width: 400px; white-space: normal; line-height: 1.4; }
	.ellipsis { max-width: 18rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; }
	h2 { margin: 0 0 0.75rem; font-size: 1rem; color: #374151; }
	.big-number { font-size: 2.5rem; font-weight: 700; margin: 0; color: #1e40af; }
	table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
	tr:not(:last-child) td { border-bottom: 1px solid #f3f4f6; }
	td { padding: 0.3rem 0; }
	td.count { text-align: right; color: #374151; font-variant-numeric: tabular-nums; }
	@media (max-width: 640px) {
		.stats-page { padding: 1rem 0.75rem; }
		.grid { grid-template-columns: 1fr; gap: 0.75rem; }
		.card { padding: 0.85rem; }
		h1 { font-size: 1.25rem; }
		.big-number { font-size: 2rem; }
		.ellipsis { max-width: 100%; }
		.reason-cell {
			max-width: 100%;
			display: -webkit-box;
			-webkit-line-clamp: 2;
			line-clamp: 2;
			-webkit-box-orient: vertical;
			overflow: hidden;
		}
		.cat-row td { padding: 0.45rem 0; }
		table { font-size: 0.82rem; }
	}
</style>

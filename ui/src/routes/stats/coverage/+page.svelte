<script lang="ts">
	import type { PageData } from './$types';
	const { data }: { data: PageData } = $props();
	const cov = $derived(data.coverage);
	const s = $derived(cov.summary as Record<string, number | number[]>);

	const num = (n: number | null | undefined, d = 0) =>
		n === null || n === undefined ? '—' : n.toLocaleString('el-GR', { maximumFractionDigits: d });
	const pct = (x: number | null | undefined, d = 1) =>
		x === null || x === undefined ? '—' : (x * 100).toLocaleString('el-GR', { maximumFractionDigits: d }) + '%';

	const hirMicro = $derived((s.hir_micro as number) ?? null);
	const ci = $derived((s.hir_ci95 as number[]) ?? null);

	type Filter = 'all' | 'reviewed' | 'not_reviewed' | 'private';
	let filter = $state<Filter>('reviewed');
	let cityFilter = $state<string>('');

	const meetings = $derived(
		cov.meetings.filter((m) => {
			if (cityFilter && m.city !== cityFilter) return false;
			if (filter === 'reviewed') return m.public && m.reviewed === true;
			if (filter === 'not_reviewed') return m.public && m.reviewed === false;
			if (filter === 'private') return !m.public;
			return true;
		})
	);
	const cityNames = $derived([...new Set(cov.cities.map((c) => c.city))].sort());
</script>

<div class="stats-page">
	<header>
		<h1>Κάλυψη δεδομένων &amp; HIR</h1>
		<div class="header-actions">
			<a href="/stats" class="btn">← Στατιστικά</a>
			<a href="/" class="btn">Αρχική</a>
		</div>
	</header>

	<div class="grid">
		<section class="card">
			<h2>Human Intervention Rate</h2>
			<p class="big-number">{pct(hirMicro)}</p>
			<p class="card-note">
				{#if ci}95% CI {pct(ci[0])}–{pct(ci[1])} · {/if}First-Pass Yield ≈ {pct(hirMicro === null ? null : 1 - hirMicro)}
			</p>
		</section>

		<section class="card">
			<h2>Συνεδριάσεις</h2>
			<table>
				<tbody>
					<tr><td>Public</td><td class="count">{num(s.meetings_public as number)}</td></tr>
					<tr><td>Private (εκτός)</td><td class="count">{num(s.meetings_private as number)}</td></tr>
					<tr><td class="st-include">Reviewed ✓</td><td class="count st-include">{num(s.meetings_reviewed as number)}</td></tr>
					<tr><td class="st-exclude">Μη-reviewed ✗</td><td class="count st-exclude">{num(s.meetings_not_reviewed as number)}</td></tr>
					<tr><td>Πόλεις</td><td class="count">{num(s.cities_total as number)}</td></tr>
					<tr><td>Speakers</td><td class="count">{num(s.speakers_identified as number)}</td></tr>
				</tbody>
			</table>
		</section>

		<section class="card">
			<h2>Ώρες ήχου (public)</h2>
			<table>
				<tbody>
					<tr><td>Σύνολο public</td><td class="count">{num(s.total_hours_public as number, 1)}h</td></tr>
					<tr><td class="st-include">Reviewed</td><td class="count st-include">{num(s.hours_reviewed as number, 1)}h</td></tr>
					<tr><td>no-edit backbone</td><td class="count">{num(s.backbone_noedit_reviewed_h as number, 1)}h</td></tr>
					<tr><td>human-verified</td><td class="count">{num(s.human_verified_reviewed_h as number, 1)}h</td></tr>
					<tr><td>task-final</td><td class="count">{num(s.task_final_reviewed_h as number, 1)}h</td></tr>
					<tr><td class="st-exclude">μη-reviewed (εκτός)</td><td class="count st-exclude">{num(s.untrusted_notreviewed_h as number, 1)}h</td></tr>
				</tbody>
			</table>
		</section>

		<section class="card wide">
			<h2>Ανά πόλη / δημοτικό συμβούλιο</h2>
			<p class="card-note">HIR υπολογίζεται μόνο σε reviewed συνεδριάσεις (humanReview=true).</p>
			<table>
				<thead>
					<tr>
						<th>Πόλη</th>
						<th class="count">Public</th>
						<th class="count">Private</th>
						<th class="count">Reviewed</th>
						<th class="count">Ώρες</th>
						<th class="count">Speakers</th>
						<th class="count">HIR</th>
					</tr>
				</thead>
				<tbody>
					{#each cov.cities as c}
						<tr>
							<td><button type="button" class="cat-link link-btn" onclick={() => (cityFilter = cityFilter === c.city ? '' : c.city)}>{c.city}</button></td>
							<td class="count">{num(c.n_meetings_public)}</td>
							<td class="count">{c.n_meetings_private ? num(c.n_meetings_private) : '—'}</td>
							<td class="count">{num(c.n_reviewed)}</td>
							<td class="count">{num(c.hours, 1)}</td>
							<td class="count">{num(c.n_speakers)}</td>
							<td class="count">{pct(c.hir)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>

		<section class="card wide">
			<h2>Συνεδριάσεις <span class="muted">({meetings.length})</span></h2>
			<div class="filters">
				<button type="button" class="chip" class:on={filter === 'reviewed'} onclick={() => (filter = 'reviewed')}>Reviewed</button>
				<button type="button" class="chip" class:on={filter === 'not_reviewed'} onclick={() => (filter = 'not_reviewed')}>Μη-reviewed</button>
				<button type="button" class="chip" class:on={filter === 'private'} onclick={() => (filter = 'private')}>Private</button>
				<button type="button" class="chip" class:on={filter === 'all'} onclick={() => (filter = 'all')}>Όλες</button>
				<select bind:value={cityFilter} class="city-select">
					<option value="">Όλες οι πόλεις</option>
					{#each cityNames as cn}<option value={cn}>{cn}</option>{/each}
				</select>
			</div>
			<table>
				<thead>
					<tr>
						<th>Πόλη</th>
						<th>Συνεδρίαση</th>
						<th>Ημ/νία</th>
						<th class="count">Utts</th>
						<th class="count">Ώρες</th>
						<th class="count">Spk</th>
						<th class="count">HIR</th>
						<th>Κατάσταση</th>
					</tr>
				</thead>
				<tbody>
					{#each meetings as m}
						<tr>
							<td>{m.city}</td>
							<td class="ellipsis" title={m.meeting}>{m.meeting}</td>
							<td class="date">{m.date ? m.date.slice(0, 10) : '—'}</td>
							<td class="count">{num(m.n_utts)}</td>
							<td class="count">{num(m.hours, 1)}</td>
							<td class="count">{num(m.n_speakers)}</td>
							<td class="count">{pct(m.hir)}</td>
							<td>
								{#if !m.public}<span class="tag priv">private</span>
								{:else if m.reviewed}<span class="tag rev">reviewed</span>
								{:else}<span class="tag notrev">μη-reviewed</span>{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	</div>
</div>

<style>
	.stats-page { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
	header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 0.5rem; }
	h1 { margin: 0; font-size: 1.5rem; }
	.header-actions { display: flex; gap: 0.5rem; }
	.btn { font-size: 0.85rem; padding: 0.3rem 0.7rem; border-radius: 4px; background: #f0f0f0; text-decoration: none; color: #333; }
	.btn:hover { filter: brightness(0.93); }
	.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
	.wide { grid-column: 1 / -1; }
	.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; }
	.card-note { margin: -0.25rem 0 0.75rem; font-size: 0.8rem; color: #6b7280; }
	h2 { margin: 0 0 0.75rem; font-size: 1rem; color: #374151; }
	.muted { color: #94a3b8; font-weight: 400; }
	.big-number { font-size: 2.5rem; font-weight: 700; margin: 0; color: #1e40af; }
	table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
	thead th { font-size: 0.75rem; font-weight: 600; color: #6b7280; padding: 0.2rem 0; border-bottom: 1px solid #e5e7eb; text-align: left; }
	thead th.count { text-align: right; }
	tr:not(:last-child) td { border-bottom: 1px solid #f3f4f6; }
	td { padding: 0.3rem 0; }
	td.count { text-align: right; color: #374151; font-variant-numeric: tabular-nums; }
	td.date { color: #6b7280; font-variant-numeric: tabular-nums; }
	.st-include { color: #166534; }
	.st-exclude { color: #b91c1c; }
	.ellipsis { max-width: 14rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.cat-link { color: #1e40af; }
	.link-btn { background: none; border: none; padding: 0; font: inherit; cursor: pointer; text-decoration: none; }
	.link-btn:hover { text-decoration: underline; }
	.filters { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.6rem; }
	.chip { font-size: 0.78rem; padding: 0.25rem 0.65rem; border-radius: 999px; border: 1px solid #d1d5db; background: #fff; color: #374151; cursor: pointer; }
	.chip.on { background: #1e40af; color: #fff; border-color: #1e40af; }
	.city-select { font-size: 0.8rem; padding: 0.25rem 0.4rem; border-radius: 6px; border: 1px solid #d1d5db; margin-left: auto; }
	.tag { font-size: 0.7rem; padding: 0.1rem 0.45rem; border-radius: 999px; }
	.tag.rev { background: #dcfce7; color: #166534; }
	.tag.notrev { background: #fee2e2; color: #b91c1c; }
	.tag.priv { background: #e5e7eb; color: #4b5563; }
	@media (max-width: 640px) {
		.stats-page { padding: 1rem 0.75rem; }
		.grid { grid-template-columns: 1fr; }
		.ellipsis { max-width: 8rem; }
	}
</style>

<script lang="ts">
	import {
		TAXONOMY,
		TAXONOMY_GROUP_ORDER,
		TAXONOMY_GROUP_LABELS,
		normalizeTaxonomyId
	} from '$lib/shared/taxonomy';
	import { getLang, t } from '$lib/i18n.svelte';
	import type { TaxonomyId, TaxonomyEntry } from '$lib/shared/taxonomy';
	import TaxonomyExamplesModal from './TaxonomyExamplesModal.svelte';

	interface Props {
		/** Multi-value: the full set of currently-assigned category ids. */
		values: readonly string[];
		/** Called with the next full set after a toggle. */
		onchange: (next: TaxonomyId[]) => void;
		disabled?: boolean;
	}

	const { values, onchange, disabled = false }: Props = $props();
	const lang = $derived(getLang());

	/**
	 * Normalize incoming values once: drop legacy ids that don't map and bring
	 * known ones to their canonical id. Order preserved so the user's choice
	 * order is visible.
	 */
	const activeSet = $derived(new Set<TaxonomyId>(
		values
			.map((v) => normalizeTaxonomyId(v))
			.filter((v): v is TaxonomyId => v !== null)
	));

	let examplesOpen = $state(false);

	const grouped = $derived(
		TAXONOMY_GROUP_ORDER.map((g) => ({
			group: g,
			label: TAXONOMY_GROUP_LABELS[g][lang],
			items: TAXONOMY.filter((cat) => cat.group === g)
		}))
	);

	function chipTitle(cat: TaxonomyEntry) {
		return `${cat[lang]} · ${cat.example_before} → ${cat.example_after}${cat.shortcut ? ` · [${cat.shortcut}]` : ''}`;
	}

	function toggle(id: TaxonomyId) {
		const next = new Set(activeSet);
		if (next.has(id)) next.delete(id);
		else next.add(id);
		// Preserve canonical TAXONOMY order for determinism.
		const ordered = TAXONOMY.filter((c) => next.has(c.id as TaxonomyId)).map((c) => c.id as TaxonomyId);
		onchange(ordered);
	}

	function clearAll() {
		if (activeSet.size === 0) return;
		onchange([]);
	}
</script>

<div class="taxonomy-select">
	<div class="header-row">
		<label for="cat-select">{t('errorCategory')}</label>
		<button
			type="button"
			class="examples-btn"
			onclick={() => (examplesOpen = true)}
			title={t('examplesModalTitle')}
		>
			{t('examplesButton')}
		</button>
	</div>

	<div class="groups">
		<button
			type="button"
			class="chip none"
			class:active={activeSet.size === 0}
			{disabled}
			onclick={clearAll}
			aria-pressed={activeSet.size === 0}
		>
			{t('noneCategory')}
		</button>

		{#each grouped as g (g.group)}
			<div class="group">
				<span class="group-label">{g.label}</span>
				<div class="chips">
					{#each g.items as cat (cat.id)}
						{@const active = activeSet.has(cat.id as TaxonomyId)}
						<button
							type="button"
							class="chip"
							class:active
							{disabled}
							onclick={() => toggle(cat.id as TaxonomyId)}
							title={chipTitle(cat)}
							aria-pressed={active}
						>
							{#if cat.shortcut}
								<span class="shortcut">{cat.shortcut}</span>
							{/if}
							{cat[lang]}
						</button>
					{/each}
				</div>
			</div>
		{/each}
	</div>
</div>

<TaxonomyExamplesModal open={examplesOpen} onclose={() => (examplesOpen = false)} />

<style>
	.taxonomy-select { display: flex; flex-direction: column; gap: 0.4rem; }
	.header-row { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
	label { font-size: 0.8rem; font-weight: 600; color: var(--muted, #6b7280); text-transform: uppercase; letter-spacing: 0.04em; }
	.examples-btn { background: transparent; border: 1px solid var(--border, #d1d5db); border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.75rem; cursor: pointer; color: var(--accent, #2563eb); }
	.examples-btn:hover { background: var(--accent-light, #eff6ff); border-color: var(--accent, #2563eb); }
	.groups { display: flex; flex-direction: column; gap: 0.35rem; }
	.group { display: flex; flex-direction: column; gap: 0.2rem; }
	.group-label { font-size: 0.65rem; color: var(--muted, #6b7280); text-transform: uppercase; letter-spacing: 0.04em; }
	.chips { display: flex; flex-wrap: wrap; gap: 0.3rem; }
	.chip { display: flex; align-items: center; gap: 0.25rem; padding: 0.2rem 0.6rem; border: 1px solid var(--border, #d1d5db); border-radius: 99px; background: white; font-size: 0.8rem; cursor: pointer; transition: background 0.1s, border-color 0.1s, color 0.1s; }
	.chip.none { align-self: flex-start; }
	.chip:not(:disabled):hover { background: var(--accent-light, #eff6ff); border-color: var(--accent, #2563eb); }
	.chip.active:not(:disabled) { background: var(--accent, #2563eb); color: white; border-color: var(--accent, #2563eb); }
	.chip:disabled { opacity: 0.5; cursor: not-allowed; pointer-events: none; background: var(--surface2, #f5f5f5); color: var(--muted, #6b7280); }
	.shortcut { font-size: 0.65rem; opacity: 0.7; font-family: monospace; }
	.chip.active .shortcut { opacity: 0.9; }
</style>

<script lang="ts">
	import {
		TAXONOMY,
		TAXONOMY_GROUP_LABELS,
		normalizeTaxonomyId,
		type TaxonomyId
	} from '$lib/shared/taxonomy';
	import { getLang, t } from '$lib/i18n.svelte';
	import { foldAscii, greekToLatin } from '$lib/shared/greeklish';
	import { tick } from 'svelte';

	// Precompute one accent-folded, multi-form haystack per taxonomy entry so
	// search matches against: Greek label (folded), English label (folded),
	// id, shortcut, AND a Greeklish transliteration of the Greek label.
	// Typing "akronymio" → matches Ακρωνύμιο; typing "punctuation" → matches
	// the English label; typing the Greek label still works.
	const HAYSTACKS = new Map<string, string>(
		TAXONOMY.map((c) => [
			c.id,
			[
				foldAscii(c.el),
				foldAscii(c.en),
				foldAscii(c.id),
				c.shortcut ?? '',
				greekToLatin(c.el),
				// Also transliterate the English label so typing partial
				// English in a Greek layout still resolves.
				greekToLatin(c.en),
				foldAscii(TAXONOMY_GROUP_LABELS[c.group].el),
				foldAscii(TAXONOMY_GROUP_LABELS[c.group].en)
			].join(' ')
		])
	);

	interface Props {
		open: boolean;
		onclose: () => void;
		/** Current selection. Toggles in/out of this set as the user picks. */
		values: readonly string[];
		/** Called with the next full set whenever a category is toggled. */
		onchange: (next: TaxonomyId[]) => void;
	}

	const { open, onclose, values, onchange }: Props = $props();
	const lang = $derived(getLang());

	let query = $state('');
	let highlighted = $state(0);
	let inputRef: HTMLInputElement | null = $state(null);

	const activeSet = $derived(new Set<TaxonomyId>(
		values
			.map((v) => normalizeTaxonomyId(v))
			.filter((v): v is TaxonomyId => v !== null)
	));

	const filtered = $derived.by(() => {
		const raw = query.trim();
		if (!raw) return TAXONOMY.slice();
		// Each input token is tested in two forms (accent-folded and
		// transliterated) against a multi-form haystack. The query matches
		// if every token matches at least one form.
		const queryForms = raw
			.split(/\s+/)
			.filter(Boolean)
			.map((tok) => [foldAscii(tok), greekToLatin(tok)] as const);
		return TAXONOMY.filter((cat) => {
			const hay = HAYSTACKS.get(cat.id) ?? '';
			return queryForms.every(([f, l]) => hay.includes(f) || hay.includes(l));
		});
	});

	$effect(() => {
		if (!open) return;
		query = '';
		highlighted = 0;
		tick().then(() => inputRef?.focus());
	});

	$effect(() => {
		void filtered;
		if (highlighted >= filtered.length) highlighted = Math.max(0, filtered.length - 1);
	});

	function toggle(id: TaxonomyId) {
		const next = new Set(activeSet);
		if (next.has(id)) next.delete(id);
		else next.add(id);
		const ordered = TAXONOMY.filter((c) => next.has(c.id as TaxonomyId)).map((c) => c.id as TaxonomyId);
		onchange(ordered);
	}

	function onKey(e: KeyboardEvent) {
		if (!open) return;
		if (e.key === 'Escape') {
			e.preventDefault();
			onclose();
			return;
		}
		if (e.key === 'ArrowDown') {
			e.preventDefault();
			if (filtered.length === 0) return;
			highlighted = Math.min(filtered.length - 1, highlighted + 1);
			return;
		}
		if (e.key === 'ArrowUp') {
			e.preventDefault();
			if (filtered.length === 0) return;
			highlighted = Math.max(0, highlighted - 1);
			return;
		}
		if (e.key === 'Enter') {
			e.preventDefault();
			const pick = filtered[highlighted];
			if (pick) toggle(pick.id as TaxonomyId);
			return;
		}
	}
</script>

{#if open}
	<div class="palette-backdrop" onclick={onclose} role="presentation"></div>
	<div class="palette" role="dialog" tabindex="-1" aria-label={t('categoryPaletteTitle')} onkeydown={onKey}>
		<div class="palette-header">
			<strong>{t('categoryPaletteTitle')}</strong>
			<button class="close-btn" onclick={onclose} aria-label={t('closeModal')}>✕</button>
		</div>
		<input
			bind:this={inputRef}
			bind:value={query}
			class="palette-input"
			type="text"
			placeholder={t('categoryPalettePlaceholder')}
			autocomplete="off"
			spellcheck="false"
		/>
		<ul class="results" role="listbox" aria-multiselectable="true">
			{#if filtered.length === 0}
				<li class="empty">{t('categoryPaletteNoResults')}</li>
			{:else}
				{#each filtered as cat, i (cat.id)}
					{@const active = activeSet.has(cat.id as TaxonomyId)}
					<li
						class="result"
						class:active={i === highlighted}
						class:selected={active}
						role="option"
						aria-selected={active}
						onmousemove={() => (highlighted = i)}
						onclick={() => toggle(cat.id as TaxonomyId)}
						onkeydown={(e) => {
							if (e.key === 'Enter' || e.key === ' ') {
								e.preventDefault();
								toggle(cat.id as TaxonomyId);
							}
						}}
					>
						<span class="checkbox" aria-hidden="true">{active ? '✓' : ''}</span>
						<span class="group-tag">{TAXONOMY_GROUP_LABELS[cat.group][lang]}</span>
						<span class="label">{cat[lang]}</span>
						{#if cat.shortcut}
							<kbd class="shortcut">{cat.shortcut}</kbd>
						{/if}
					</li>
				{/each}
			{/if}
		</ul>
		<div class="palette-hint">{t('categoryPaletteHint')}</div>
	</div>
{/if}

<style>
	.palette-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(15, 23, 42, 0.35);
		z-index: 60;
	}
	.palette {
		position: fixed;
		top: 12vh;
		left: 50%;
		transform: translateX(-50%);
		width: min(520px, 92vw);
		max-height: 70vh;
		display: flex;
		flex-direction: column;
		background: var(--surface, #fff);
		border: 1px solid var(--border, #e2e8f0);
		border-radius: 12px;
		box-shadow: 0 12px 40px rgba(0, 0, 0, 0.18);
		z-index: 70;
		overflow: hidden;
	}
	.palette-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.6rem 0.85rem 0.4rem;
		font-size: 0.9rem;
	}
	.close-btn {
		background: none;
		border: none;
		font-size: 0.95rem;
		color: var(--text-3, #94a3b8);
		cursor: pointer;
		padding: 0;
	}
	.close-btn:hover { color: var(--text, #0f172a); }
	.palette-input {
		margin: 0 0.85rem 0.5rem;
		padding: 0.5rem 0.7rem;
		border: 1px solid var(--border, #e2e8f0);
		border-radius: 8px;
		font-size: 0.95rem;
		font-family: inherit;
		background: var(--surface-2, #f8fafc);
	}
	.palette-input:focus {
		outline: 2px solid var(--accent, #2563eb);
		outline-offset: 1px;
		border-color: transparent;
	}
	.results {
		list-style: none;
		margin: 0;
		padding: 0.2rem 0.4rem 0.4rem;
		overflow-y: auto;
		flex: 1;
	}
	.result {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		padding: 0.45rem 0.55rem;
		border-radius: 6px;
		cursor: pointer;
		font-size: 0.88rem;
	}
	.result.active {
		background: var(--accent-light, #dbeafe);
		color: var(--accent-dark, #1d4ed8);
	}
	.result.selected {
		background: var(--accent-light, #dbeafe);
	}
	.result.active.selected {
		background: var(--accent, #2563eb);
		color: white;
	}
	.checkbox {
		width: 1rem;
		display: inline-flex;
		justify-content: center;
		align-items: center;
		font-weight: 700;
		color: var(--accent, #2563eb);
	}
	.result.active.selected .checkbox { color: white; }
	.result .group-tag {
		font-size: 0.65rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--text-3, #94a3b8);
		min-width: 7.5rem;
	}
	.result.active .group-tag { color: var(--accent-dark, #1d4ed8); }
	.result .label { flex: 1; }
	.result .shortcut {
		font-family: monospace;
		font-size: 0.72rem;
		padding: 0.05rem 0.35rem;
		border: 1px solid var(--border, #e2e8f0);
		border-radius: 4px;
		background: var(--surface-2, #f8fafc);
	}
	.empty {
		padding: 1rem;
		text-align: center;
		color: var(--text-3, #94a3b8);
		font-size: 0.85rem;
	}
	.palette-hint {
		padding: 0.45rem 0.85rem 0.6rem;
		font-size: 0.72rem;
		color: var(--text-3, #94a3b8);
		border-top: 1px solid var(--border, #e2e8f0);
		background: var(--surface-2, #f8fafc);
	}
</style>

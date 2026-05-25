<script lang="ts">
	import { goto } from '$app/navigation';
	import { parseSeedParam, randomSeed, hashSeedString, UINT32_MAX } from '$lib/shared/urls';
	import { t } from '$lib/i18n.svelte';
	import StatusDistribution from '$lib/components/StatusDistribution.svelte';
	import type { PageData } from './$types';

	const { data }: { data: PageData } = $props();

	let seedInput = $state<string>('');
	let skipClassified = $state(true);
	let error = $state<string | null>(null);

	const INT_RE = /^\d+$/;
	const seedPreview = $derived.by(() => {
		const raw = seedInput.trim();
		if (raw === '' || INT_RE.test(raw)) return null;
		return hashSeedString(raw);
	});

	function start() {
		error = null;
		const raw = seedInput.trim();
		let seed: number;
		if (raw === '') {
			seed = data.suggestedSeed;
		} else {
			const parsed = parseSeedParam(raw);
			if (parsed === null) {
				error = t('seedInvalid');
				return;
			}
			seed = parsed;
		}
		const params = new URLSearchParams({ seed: String(seed) });
		if (skipClassified) params.set('skip', '1');
		goto(`/?${params.toString()}`);
	}

	function newRandom() {
		seedInput = String(randomSeed());
	}
</script>

<div class="landing">
	<h1>{t('landingTitle')}</h1>
	<p class="lead">{t('landingLead')}</p>

	{#if data.distribution}
		<section class="distribution">
			<h2>{t('distributionTitle')}</h2>
			<StatusDistribution counts={data.distribution} seed={data.suggestedSeed} variant="compact" />
		</section>
	{/if}

	<form
		class="seed-form"
		onsubmit={(e) => {
			e.preventDefault();
			start();
		}}
	>
		<label>
			<span>{t('seedLabel')}</span>
			<div class="row">
				<input
					type="text"
					placeholder={t('seedPlaceholder', { default: String(data.suggestedSeed) })}
					bind:value={seedInput}
					aria-invalid={error !== null}
				/>
				<button type="button" class="ghost" onclick={newRandom}>{t('seedRandomize')}</button>
			</div>
			{#if seedPreview !== null}
				<small class="preview">→ {seedPreview}</small>
			{/if}
			{#if error}<span class="error">{error}</span>{/if}
			<small class="hint">{t('seedHint')}</small>
		</label>
		<label class="skip-toggle">
			<input type="checkbox" bind:checked={skipClassified} />
			<span>{t('skipClassifiedLabel')}</span>
		</label>
		<button type="submit" class="primary">{t('startReview')}</button>
	</form>

	<nav class="links">
		<a href="/stats">{t('statsLink')}</a>
	</nav>
</div>

<style>
	.landing { max-width: 620px; margin: 3rem auto; padding: 1.5rem; }
	h1 { font-size: 1.6rem; margin: 0 0 0.4rem; }
	h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-3, #94a3b8); margin: 0 0 0.55rem; font-weight: 600; }
	.lead { color: var(--text-2, #475569); margin: 0 0 1.4rem; }
	.distribution { margin: 0 0 1.8rem; }
	.seed-form { display: flex; flex-direction: column; gap: 1rem; }
	label { display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.85rem; color: var(--text-2, #475569); }
	.row { display: flex; gap: 0.5rem; }
	input {
		flex: 1; padding: 0.55rem 0.75rem; font-size: 1rem;
		border: 1px solid var(--border, #e2e8f0); border-radius: 8px;
		font-family: monospace;
	}
	input[aria-invalid='true'] { border-color: #dc2626; }
	.hint { font-size: 0.72rem; color: var(--text-3, #94a3b8); }
	.preview { font-size: 0.72rem; color: var(--accent, #2563eb); font-family: monospace; }
	.skip-toggle {
		display: flex; flex-direction: row; align-items: center; gap: 0.45rem;
		font-size: 0.82rem; color: var(--text-2, #475569); margin-top: 0.2rem;
	}
	.skip-toggle input { accent-color: var(--accent, #2563eb); }
	.error { font-size: 0.78rem; color: #dc2626; }
	button {
		padding: 0.5rem 0.95rem; font-size: 0.9rem;
		border-radius: 8px; cursor: pointer; font-family: inherit;
	}
	.primary { background: var(--accent, #2563eb); color: white; border: none; }
	.primary:hover { background: var(--accent-dark, #1d4ed8); }
	.ghost {
		background: var(--surface-2, #f8fafc); color: var(--text-2, #475569);
		border: 1px solid var(--border, #e2e8f0);
	}
	.ghost:hover { background: var(--surface-3, #f1f5f9); }
	.links { margin-top: 2rem; display: flex; gap: 1rem; }
	.links a { color: var(--accent, #2563eb); font-size: 0.9rem; }
</style>

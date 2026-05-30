<script lang="ts">
	import { getLang, toggleLang } from '$lib/i18n.svelte';
	const lang = $derived(getLang());
</script>

<button onclick={toggleLang} class="lang-btn" aria-label="Toggle language">
	{lang === 'el' ? 'EN' : 'EL'}
</button>

<style>
	.lang-btn {
		padding: 0.2rem 0.6rem;
		border: 1px solid var(--border, #d1d5db);
		border-radius: 4px;
		background: white;
		font-size: 0.8rem;
		font-weight: 700;
		cursor: pointer;
		letter-spacing: 0.05em;
	}

	.lang-btn:hover {
		background: var(--accent-light, #eff6ff);
	}
</style>

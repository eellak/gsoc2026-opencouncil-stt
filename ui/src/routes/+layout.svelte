<script lang="ts">
	import LangToggle from '$lib/components/LangToggle.svelte';
	import { t } from '$lib/i18n.svelte';

	let { children } = $props();
</script>

<svelte:head>
	<title>OpenCouncil Correction Explorer</title>
</svelte:head>

<div class="app">
	<nav class="topnav">
		<a href="/" class="brand">OpenCouncil</a>
		<a href="/" class="nav-link">{t('review')}</a>
		<a href="/stats" class="nav-link">{t('stats')}</a>
		<span class="spacer"></span>
		<LangToggle />
	</nav>
	<main>
		{@render children()}
	</main>
</div>

<style>
	:global(*, *::before, *::after) {
		box-sizing: border-box;
	}

	:global(body) {
		margin: 0;
		font-family: system-ui, sans-serif;
		background: #f9fafb;
		color: #111827;
		--border: #d1d5db;
		--surface: #f3f4f6;
		--surface2: #f9fafb;
		--muted: #6b7280;
		--accent: #2563eb;
		--accent-light: #eff6ff;
		--text: #111827;
	}

	.app {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
	}

	.topnav {
		display: flex;
		align-items: center;
		gap: 1rem;
		padding: 0 1.25rem;
		height: 48px;
		background: white;
		border-bottom: 1px solid var(--border);
		position: sticky;
		top: 0;
		z-index: 10;
	}

	.brand {
		font-weight: 700;
		font-size: 0.95rem;
		color: var(--accent);
		text-decoration: none;
		margin-right: 0.5rem;
	}

	.nav-link {
		font-size: 0.85rem;
		color: var(--muted);
		text-decoration: none;
		padding: 0.25rem 0.5rem;
		border-radius: 4px;
	}

	.nav-link:hover { color: var(--accent); background: var(--accent-light); }

	.spacer { flex: 1; }

	main {
		flex: 1;
		padding: 1rem 1.25rem;
		max-width: 1400px;
		width: 100%;
		margin: 0 auto;
	}
</style>

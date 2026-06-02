<script lang="ts">
	import { t } from '$lib/i18n.svelte';

	interface Props {
		open: boolean;
		onclose: () => void;
		showChain?: boolean;
	}
	const { open, onclose, showChain = false }: Props = $props();

	function onKey(e: KeyboardEvent) {
		if (e.key === 'Escape') onclose();
	}

	const rows = $derived([
		{ keys: ['Space'], label: t('shortcutPlay') },
		{ keys: ['j', 'k'], label: t('shortcutNav') },
		{ keys: ['i', 'x', 'u'], label: t('shortcutDecisions') },
		{ keys: ['1', '…', '0'], label: t('shortcutCategory') },
		{ keys: ['/'], label: t('shortcutPalette') },
		{ keys: ['a'], label: t('shortcutAutoplay') },
		{ keys: ['l'], label: t('shortcutLoop') },
		{ keys: ['[', ']'], label: t('shortcutNudgeStart') },
		{ keys: ['{', '}'], label: t('shortcutNudgeEnd') },
		...(showChain ? [{ keys: ['c'], label: t('shortcutChain') }] : []),
		{ keys: ['?'], label: t('shortcutHelp') }
	]);
</script>

<svelte:window onkeydown={open ? onKey : undefined} />

{#if open}
	<div class="backdrop" role="presentation" onclick={onclose}></div>
	<div class="modal" role="dialog" aria-modal="true" aria-label={t('shortcutsModalTitle')}>
		<header>
			<h2>{t('shortcutsModalTitle')}</h2>
			<button type="button" class="close" onclick={onclose} aria-label={t('closeModal')}>✕</button>
		</header>
		<ul>
			{#each rows as r}
				<li>
					<span class="keys">
						{#each r.keys as k}
							{#if k === '…'}<span class="dots">{k}</span>{:else}<kbd>{k}</kbd>{/if}
						{/each}
					</span>
					<span class="label">{r.label}</span>
				</li>
			{/each}
		</ul>
	</div>
{/if}

<style>
	.backdrop {
		position: fixed; inset: 0; background: rgba(15, 23, 42, 0.45); z-index: 90;
	}
	.modal {
		position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
		width: min(440px, calc(100vw - 2rem)); max-height: 80vh;
		background: white; border-radius: 10px;
		box-shadow: 0 20px 50px rgba(15, 23, 42, 0.25); z-index: 100;
		display: flex; flex-direction: column; overflow: hidden;
	}
	header {
		display: flex; align-items: center; justify-content: space-between;
		padding: 0.6rem 0.9rem; border-bottom: 1px solid var(--border, #e2e8f0);
	}
	h2 { margin: 0; font-size: 0.95rem; }
	.close {
		background: transparent; border: 0; font-size: 1rem;
		cursor: pointer; color: var(--muted, #6b7280); padding: 0.1rem 0.4rem;
	}
	.close:hover { color: #0f172a; }
	ul {
		list-style: none; margin: 0; padding: 0.6rem 0.9rem 0.8rem;
		display: flex; flex-direction: column; gap: 0.35rem; overflow-y: auto;
	}
	li {
		display: flex; align-items: center; gap: 0.75rem;
		padding: 0.25rem 0; font-size: 0.85rem;
	}
	.keys {
		display: inline-flex; gap: 0.2rem; min-width: 110px; flex-shrink: 0;
	}
	.label { color: #334155; }
	.dots { color: #94a3b8; font-size: 0.85rem; }
	kbd {
		font-family: ui-monospace, monospace; font-size: 0.72rem;
		padding: 0.05rem 0.4rem; background: #f8fafc;
		border: 1px solid #cbd5e1; border-bottom-width: 2px;
		border-radius: 3px; color: #0f172a;
	}
</style>

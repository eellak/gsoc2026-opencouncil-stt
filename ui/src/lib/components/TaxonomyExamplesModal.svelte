<script lang="ts">
	import { TAXONOMY, TAXONOMY_GROUP_ORDER, TAXONOMY_GROUP_LABELS } from '$lib/shared/taxonomy';
	import { getLang, t } from '$lib/i18n.svelte';

	interface Props {
		open: boolean;
		onclose: () => void;
	}

	const { open, onclose }: Props = $props();
	const lang = $derived(getLang());

	const grouped = $derived(
		TAXONOMY_GROUP_ORDER.map((g) => ({
			group: g,
			label: TAXONOMY_GROUP_LABELS[g][lang],
			items: TAXONOMY.filter((cat) => cat.group === g)
		}))
	);

	function onKey(e: KeyboardEvent) {
		if (e.key === 'Escape') onclose();
	}
</script>

<svelte:window onkeydown={open ? onKey : undefined} />

{#if open}
	<div class="backdrop" role="presentation" onclick={onclose}></div>
	<div class="modal" role="dialog" aria-modal="true" aria-label={t('examplesModalTitle')}>
		<header>
			<h2>{t('examplesModalTitle')}</h2>
			<button type="button" class="close" onclick={onclose} aria-label={t('closeModal')}>✕</button>
		</header>
		<div class="body">
			{#each grouped as g}
				<section>
					<h3>{g.label}</h3>
					<ul>
						{#each g.items as cat}
							<li>
								<div class="row-head">
									{#if cat.shortcut}
										<kbd>{cat.shortcut}</kbd>
									{/if}
									<span class="cat-label">{cat[lang]}</span>
								</div>
								<div class="example">
									<code class="before">{cat.example_before}</code>
									<span class="arrow">→</span>
									<code class="after">{cat.example_after}</code>
								</div>
							</li>
						{/each}
					</ul>
				</section>
			{/each}
		</div>
	</div>
{/if}

<style>
	.backdrop {
		position: fixed;
		inset: 0;
		background: rgba(15, 23, 42, 0.45);
		z-index: 90;
	}

	.modal {
		position: fixed;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		width: min(720px, calc(100vw - 2rem));
		max-height: 80vh;
		background: white;
		border-radius: 8px;
		box-shadow: 0 20px 50px rgba(15, 23, 42, 0.25);
		z-index: 100;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.75rem 1rem;
		border-bottom: 1px solid var(--border, #e2e8f0);
	}

	h2 {
		margin: 0;
		font-size: 1rem;
	}

	.close {
		background: transparent;
		border: 0;
		font-size: 1rem;
		cursor: pointer;
		color: var(--muted, #6b7280);
	}

	.body {
		padding: 0.75rem 1rem 1rem;
		overflow-y: auto;
	}

	section + section {
		margin-top: 1rem;
	}

	h3 {
		margin: 0 0 0.4rem;
		font-size: 0.78rem;
		font-weight: 700;
		color: var(--muted, #6b7280);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}

	ul {
		list-style: none;
		padding: 0;
		margin: 0;
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.5rem 1rem;
	}

	@media (max-width: 600px) {
		ul {
			grid-template-columns: 1fr;
		}
	}

	li {
		padding: 0.4rem 0.5rem;
		border: 1px solid var(--border, #e2e8f0);
		border-radius: 6px;
		background: #fafafa;
	}

	.row-head {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		margin-bottom: 0.25rem;
	}

	.cat-label {
		font-weight: 600;
		font-size: 0.85rem;
	}

	kbd {
		font-family: monospace;
		font-size: 0.7rem;
		padding: 0.05rem 0.35rem;
		background: white;
		border: 1px solid var(--border, #d1d5db);
		border-bottom-width: 2px;
		border-radius: 3px;
		color: #0f172a;
	}

	.example {
		font-size: 0.8rem;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.35rem;
	}

	code {
		font-family: ui-monospace, monospace;
		padding: 0.05rem 0.3rem;
		border-radius: 3px;
	}

	.before {
		background: #fef2f2;
		color: #991b1b;
	}

	.after {
		background: #ecfdf5;
		color: #065f46;
	}

	.arrow {
		color: var(--muted, #6b7280);
	}
</style>

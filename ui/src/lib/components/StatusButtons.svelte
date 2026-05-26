<script lang="ts">
	import { t } from '$lib/i18n.svelte';
	import type { IncludeStatus } from '$lib/shared/types';

	interface Props {
		status: IncludeStatus;
		saving?: boolean;
		onchange: (v: IncludeStatus) => void;
	}

	const { status, saving = false, onchange }: Props = $props();
	const value = $derived(status);

	// Order matches the swipe convention: exclude (left) ← uncertain → include (right).
	const options: { status: IncludeStatus; key: Parameters<typeof t>[0]; color: string }[] = [
		{ status: 'exclude', key: 'exclude', color: 'red' },
		{ status: 'uncertain', key: 'uncertain', color: 'yellow' },
		{ status: 'include', key: 'include', color: 'green' }
	];
</script>

<div class="status-buttons" role="group" aria-label="Include status">
	{#each options as opt}
		<button
			type="button"
			class="btn {opt.color}"
			class:active={value === opt.status}
			disabled={saving}
			onclick={() => onchange(opt.status)}
			aria-pressed={value === opt.status}
		>
			{t(opt.key)}
		</button>
	{/each}
</div>

<style>
	.status-buttons {
		display: flex;
		gap: 0.5rem;
	}

	.btn {
		flex: 1;
		padding: 0.5rem 1rem;
		border-radius: 6px;
		border: 2px solid transparent;
		font-weight: 600;
		font-size: 0.9rem;
		cursor: pointer;
		transition: all 0.1s;
		background: var(--surface2, #f5f5f5);
		color: var(--text, #111);
	}

	.btn:hover { opacity: 0.85; }

	.btn.green { border-color: #16a34a; }
	.btn.green.active { background: #16a34a; color: white; }

	.btn.red { border-color: #dc2626; }
	.btn.red.active { background: #dc2626; color: white; }

	.btn.yellow { border-color: #ca8a04; }
	.btn.yellow.active { background: #ca8a04; color: white; }
</style>

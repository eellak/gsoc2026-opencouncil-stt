<script lang="ts">
	import { userStore } from '$lib/client/user-store.svelte';

	let { onclose = () => {} }: { onclose?: () => void } = $props();

	let existingUsernames = $state<string[]>([]);
	let newName = $state('');
	let loading = $state(true);
	let error = $state<string | null>(null);

	$effect(() => {
		fetch('/api/users')
			.then((r) => r.json())
			.then((d) => { existingUsernames = d.usernames ?? []; })
			.catch(() => {})
			.finally(() => { loading = false; });
	});

	function pick(name: string) {
		const trimmed = name.trim();
		if (!trimmed) return;
		userStore.set(trimmed);
		onclose();
	}

	function submit() {
		const trimmed = newName.trim();
		if (!trimmed) { error = 'Γράψε ένα όνομα.'; return; }
		pick(trimmed);
	}
</script>

<div class="overlay" role="dialog" aria-modal="true" aria-label="Επιλογή χρήστη">
	<div class="modal">
		<h2>Ποιος κάνει review;</h2>
		<p class="sub">Το όνομά σου αποθηκεύεται με κάθε annotation. Δεν χρειάζεται κωδικός.</p>

		{#if loading}
			<p class="hint">Φόρτωση...</p>
		{:else if existingUsernames.length > 0}
			<ul class="existing">
				{#each existingUsernames as name}
					<li>
						<button class="name-btn" onclick={() => pick(name)}>{name}</button>
					</li>
				{/each}
			</ul>
			<div class="divider">ή νέος χρήστης</div>
		{/if}

		<form onsubmit={(e) => { e.preventDefault(); submit(); }}>
			<div class="row">
				<input
					type="text"
					placeholder="π.χ. christos"
					bind:value={newName}
					aria-invalid={error !== null}
					autofocus
				/>
				<button type="submit" class="primary">Επιλογή</button>
			</div>
			{#if error}<span class="error">{error}</span>{/if}
		</form>
	</div>
</div>

<style>
	.overlay {
		position: fixed; inset: 0; background: rgba(0,0,0,0.45);
		display: flex; align-items: center; justify-content: center; z-index: 1000;
	}
	.modal {
		background: white; border-radius: 12px; padding: 1.75rem 2rem;
		width: min(380px, 90vw); box-shadow: 0 8px 32px rgba(0,0,0,0.18);
	}
	h2 { margin: 0 0 0.25rem; font-size: 1.15rem; }
	.sub { margin: 0 0 1.1rem; font-size: 0.82rem; color: #64748b; }
	.existing { list-style: none; padding: 0; margin: 0 0 0.5rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }
	.name-btn {
		padding: 0.35rem 0.8rem; border: 1px solid #e2e8f0; border-radius: 20px;
		background: #f8fafc; cursor: pointer; font-size: 0.9rem;
	}
	.name-btn:hover { background: #e0f2fe; border-color: #38bdf8; }
	.divider { font-size: 0.75rem; color: #94a3b8; margin: 0.6rem 0; text-align: center; }
	.row { display: flex; gap: 0.5rem; }
	input {
		flex: 1; padding: 0.5rem 0.75rem; font-size: 0.95rem;
		border: 1px solid #e2e8f0; border-radius: 8px;
	}
	input[aria-invalid='true'] { border-color: #dc2626; }
	.primary {
		padding: 0.5rem 1rem; background: #2563eb; color: white;
		border: none; border-radius: 8px; cursor: pointer; font-size: 0.9rem;
	}
	.primary:hover { background: #1d4ed8; }
	.error { font-size: 0.78rem; color: #dc2626; margin-top: 0.3rem; display: block; }
	.hint { color: #94a3b8; font-size: 0.85rem; }
</style>

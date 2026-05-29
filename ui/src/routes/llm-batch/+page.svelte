<script lang="ts">
	import { onMount } from 'svelte';

	interface BatchItem { id: string; b: string; a: string }
	interface Batch {
		batch_id: string;
		model: string;
		items: BatchItem[];
		prompt: string;
		remaining: number;
	}
	interface BatchSummary { batch_id: string; model: string; created_at: string; size: number }
	interface Stats {
		total: number;
		labeled: number;
		remaining: number;
		by_source: Record<string, number>;
	}
	interface IngestResult {
		batch_id: string;
		model: string;
		accepted: number;
		declined: string[];
		duplicates: string[];
		rejected: { id: string; reason: string }[];
		missing_from_batch: string[];
		extra_ids: string[];
	}

	const MODEL_OPTIONS = [
		'gemini-3.5-flash',
		'gemini-3.5-pro',
		'gemini-2.5-pro',
		'gemini-2.5-flash',
		'gpt-5',
		'claude-haiku-4.5',
		'other'
	];

	let stats = $state<Stats | null>(null);
	let batches = $state<BatchSummary[]>([]);
	let active = $state<Batch[]>([]);
	let n = $state(200);
	let model = $state('gemini-3.5-flash');
	let modelOther = $state('');
	let parallelCount = $state(1);
	let issueBusy = $state(false);
	let issueError = $state<string | null>(null);

	const effectiveModel = $derived(model === 'other' ? modelOther.trim() : model);

	async function refreshStats() {
		const res = await fetch('/api/llm-batch/stats');
		if (res.ok) stats = await res.json();
	}
	async function refreshBatches() {
		const res = await fetch('/api/llm-batch/list');
		if (res.ok) batches = await res.json();
	}

	async function issueBatch(count: number) {
		issueError = null;
		if (!effectiveModel) { issueError = 'Pick or type a model name'; return; }
		issueBusy = true;
		try {
			const results: Batch[] = [];
			for (let i = 0; i < count; i++) {
				const res = await fetch('/api/llm-batch/next', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ n, model: effectiveModel })
				});
				if (!res.ok) {
					issueError = `Failed: ${res.status} ${await res.text()}`;
					break;
				}
				const b = (await res.json()) as Batch;
				results.push(b);
				if (b.items.length === 0) {
					issueError = 'No unlabeled items remaining.';
					break;
				}
			}
			active = [...active, ...results];
			await refreshBatches();
		} finally {
			issueBusy = false;
		}
	}

	const copyStatus: Record<string, 'ok' | 'err' | null> = $state({});
	async function copyPrompt(b: Batch) {
		try {
			await navigator.clipboard.writeText(b.prompt);
			copyStatus[b.batch_id] = 'ok';
		} catch {
			copyStatus[b.batch_id] = 'err';
		}
		setTimeout(() => { copyStatus[b.batch_id] = null; }, 2000);
	}

	function dismiss(batch_id: string) {
		active = active.filter((b) => b.batch_id !== batch_id);
	}

	async function discard(batch_id: string) {
		await fetch(`/api/llm-batch/${batch_id}`, { method: 'DELETE' });
		dismiss(batch_id);
		await refreshBatches();
	}

	const ingestPaste: Record<string, string> = $state({});
	const ingestResult: Record<string, IngestResult | null> = $state({});
	const ingestError: Record<string, string | null> = $state({});

	const reopenedFor: Record<string, boolean> = $state({});
	function reopenBatch(batch_id: string) {
		reopenedFor[batch_id] = !reopenedFor[batch_id];
	}

	async function doReopenIngest(batch_id: string) {
		ingestError[batch_id] = null;
		ingestResult[batch_id] = null;
		const raw = ingestPaste[batch_id] ?? '';
		if (!raw.trim()) {
			ingestError[batch_id] = 'paste the model reply first';
			return;
		}
		const res = await fetch('/api/llm-batch/ingest', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ batch_id, raw })
		});
		if (!res.ok) {
			ingestError[batch_id] = `${res.status} ${await res.text()}`;
			return;
		}
		ingestResult[batch_id] = await res.json();
		await refreshStats();
		await refreshBatches();
	}

	async function doIngest(b: Batch) {
		ingestError[b.batch_id] = null;
		ingestResult[b.batch_id] = null;
		const raw = ingestPaste[b.batch_id] ?? '';
		if (!raw.trim()) {
			ingestError[b.batch_id] = 'paste the model reply first';
			return;
		}
		const res = await fetch('/api/llm-batch/ingest', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ batch_id: b.batch_id, raw })
		});
		if (!res.ok) {
			ingestError[b.batch_id] = `${res.status} ${await res.text()}`;
			return;
		}
		ingestResult[b.batch_id] = await res.json();
		await refreshStats();
		await refreshBatches();
	}

	onMount(() => {
		void refreshStats();
		void refreshBatches();
	});
</script>

<svelte:head><title>External-LLM categorization</title></svelte:head>

<main class="wrap">
	<header>
		<h1>External-LLM categorization</h1>
		<p class="lead">
			Hand off unlabeled utterance pairs to Gemini / GPT / Claude via copy-paste.
			Each batch carries a unique <code>BATCH_ID</code> so multiple parallel chats can't get mixed up.
		</p>
	</header>

	<section class="stats">
		<h2>Coverage</h2>
		{#if stats}
			<p>
				<strong>{stats.labeled.toLocaleString()}</strong> /
				{stats.total.toLocaleString()} labeled
				(<em>{stats.remaining.toLocaleString()} remaining</em>)
			</p>
			<ul class="src-list">
				{#each Object.entries(stats.by_source).sort(([, a], [, b]) => b - a) as [src, n] (src)}
					<li><code>{src}</code>: {n.toLocaleString()}</li>
				{/each}
			</ul>
		{:else}
			<p>loading…</p>
		{/if}
		<button onclick={refreshStats}>Refresh</button>
	</section>

	<section class="issue">
		<h2>Issue new batches</h2>
		<div class="row">
			<label>
				Items per batch:
				<input type="number" min="1" max="500" bind:value={n} />
			</label>
			<label>
				Model:
				<select bind:value={model}>
					{#each MODEL_OPTIONS as opt (opt)}<option value={opt}>{opt}</option>{/each}
				</select>
			</label>
			{#if model === 'other'}
				<label>
					Slug:
					<input type="text" placeholder="e.g. gemini-3-preview" bind:value={modelOther} />
				</label>
			{/if}
			<label>
				Parallel batches:
				<input type="number" min="1" max="10" bind:value={parallelCount} />
			</label>
			<button disabled={issueBusy} onclick={() => issueBatch(parallelCount)}>
				{issueBusy ? 'Issuing…' : `Issue ${parallelCount} batch${parallelCount > 1 ? 'es' : ''}`}
			</button>
			<a class="open-gemini" href="https://aistudio.google.com/" target="_blank" rel="noopener">
				Open AI Studio ↗
			</a>
		</div>
		{#if issueError}<p class="err">{issueError}</p>{/if}
	</section>

	<section class="active">
		<h2>Active batches</h2>
		{#if active.length === 0}
			<p class="muted">No batches in this session. Hit “Issue” above.</p>
		{:else}
			{#each active as b (b.batch_id)}
				<article class="batch">
					<header>
						<span class="badge">BATCH_ID {b.batch_id}</span>
						<span class="muted">{b.model} · {b.items.length} items · {b.remaining.toLocaleString()} repo-remaining</span>
						<button class="ghost" onclick={() => dismiss(b.batch_id)}>Dismiss</button>
						<button class="ghost danger" onclick={() => discard(b.batch_id)}>Discard issued-set</button>
					</header>

					<div class="row">
						<button onclick={() => copyPrompt(b)}>Copy prompt for batch {b.batch_id}</button>
						{#if copyStatus[b.batch_id] === 'ok'}<span class="ok">copied ✓</span>{/if}
						{#if copyStatus[b.batch_id] === 'err'}<span class="err">copy failed — select prompt manually</span>{/if}
					</div>

					<details>
						<summary>Show prompt</summary>
						<pre>{b.prompt}</pre>
					</details>

					<label class="ingest-label">
						Paste reply for <code>{b.batch_id}</code>:
						<textarea
							rows="6"
							placeholder={`Must contain "BATCH_ID: ${b.batch_id}" and a JSON array of {id, c}`}
							bind:value={ingestPaste[b.batch_id]}
						></textarea>
					</label>
					<button onclick={() => doIngest(b)}>Ingest reply</button>

					{#if ingestError[b.batch_id]}<p class="err">{ingestError[b.batch_id]}</p>{/if}
					{#if ingestResult[b.batch_id]}
						{@const r = ingestResult[b.batch_id]!}
						<p class="ok">
							accepted: <strong>{r.accepted}</strong> ·
							declined: {r.declined.length} ·
							duplicates: {r.duplicates.length} ·
							rejected: {r.rejected.length} ·
							missing: {r.missing_from_batch.length} ·
							extras: {r.extra_ids.length}
						</p>
						{#if r.rejected.length > 0}
							<details>
								<summary>{r.rejected.length} rejected</summary>
								<ul>{#each r.rejected as x (x.id)}<li><code>{x.id}</code> — {x.reason}</li>{/each}</ul>
							</details>
						{/if}
					{/if}
				</article>
			{/each}
		{/if}
	</section>

	<section class="persisted">
		<h2>Issued batches on disk</h2>
		<p class="muted">
			These batch-id files are still on disk. If you have a saved Gemini reply
			for any of them, open it here and ingest — duplicates are silently
			skipped, so it is safe to paste an old reply.
		</p>
		{#if batches.length === 0}
			<p class="muted">none</p>
		{:else}
			<table>
				<thead><tr><th>id</th><th>model</th><th>size</th><th>created</th><th></th></tr></thead>
				<tbody>
					{#each batches as bs (bs.batch_id)}
						<tr>
							<td><code>{bs.batch_id}</code></td>
							<td>{bs.model}</td>
							<td>{bs.size}</td>
							<td>{bs.created_at}</td>
							<td>
								<button class="ghost" onclick={() => reopenBatch(bs.batch_id)}>
									{reopenedFor[bs.batch_id] ? 'Hide' : 'Paste & ingest'}
								</button>
							</td>
						</tr>
						{#if reopenedFor[bs.batch_id]}
							<tr>
								<td colspan="5">
									<label class="ingest-label">
										Paste reply for <code>{bs.batch_id}</code>:
										<textarea
											rows="6"
											placeholder={`Must contain "BATCH_ID: ${bs.batch_id}" and a JSON array of {id, c}`}
											bind:value={ingestPaste[bs.batch_id]}
										></textarea>
									</label>
									<button onclick={() => doReopenIngest(bs.batch_id)}>Ingest reply</button>
									{#if ingestError[bs.batch_id]}<p class="err">{ingestError[bs.batch_id]}</p>{/if}
									{#if ingestResult[bs.batch_id]}
										{@const r = ingestResult[bs.batch_id]!}
										<p class="ok">
											accepted: <strong>{r.accepted}</strong> ·
											declined: {r.declined.length} ·
											duplicates: {r.duplicates.length} ·
											rejected: {r.rejected.length} ·
											missing: {r.missing_from_batch.length} ·
											extras: {r.extra_ids.length}
										</p>
									{/if}
								</td>
							</tr>
						{/if}
					{/each}
				</tbody>
			</table>
		{/if}
	</section>
</main>

<style>
	.wrap { max-width: 60rem; margin: 0 auto; padding: 2rem 1rem 6rem; font: 14px/1.5 system-ui, sans-serif; }
	h1 { font-size: 1.6rem; margin: 0 0 0.25rem; }
	h2 { font-size: 1.1rem; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }
	.lead { color: #555; margin: 0 0 1rem; }
	.row { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin: 0.5rem 0; }
	label { display: inline-flex; flex-direction: column; font-size: 0.85rem; color: #444; gap: 0.2rem; }
	input[type=number] { width: 5rem; }
	input[type=text] { width: 14rem; }
	button { padding: 0.4rem 0.8rem; cursor: pointer; }
	button.ghost { background: transparent; border: 1px solid #ccc; }
	button.ghost.danger { color: #a00; border-color: #d99; }
	.batch { border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem 1rem; margin: 1rem 0; background: #fafafa; }
	.batch > header { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.5rem; }
	.badge { background: #224; color: #fff; padding: 0.15rem 0.5rem; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 0.85rem; }
	.muted { color: #888; font-size: 0.85rem; }
	.err { color: #a00; }
	.ok { color: #060; }
	pre { background: #f4f4f4; padding: 0.5rem; border-radius: 4px; max-height: 24rem; overflow: auto; white-space: pre-wrap; font-size: 0.8rem; }
	textarea { width: 100%; font-family: ui-monospace, monospace; font-size: 0.85rem; }
	.ingest-label { display: block; margin: 0.5rem 0; }
	.src-list { margin: 0.25rem 0 0.5rem; padding-left: 1.25rem; }
	.open-gemini { margin-left: auto; }
	table { border-collapse: collapse; width: 100%; }
	th, td { border: 1px solid #ddd; padding: 0.25rem 0.5rem; text-align: left; }
	code { font-family: ui-monospace, monospace; background: #eee; padding: 0 0.2rem; }
</style>

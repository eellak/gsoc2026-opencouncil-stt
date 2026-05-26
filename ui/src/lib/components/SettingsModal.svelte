<script lang="ts">
	import { t } from '$lib/i18n.svelte';
	import { playbackPrefs } from '$lib/client/playback-prefs.svelte';
	import { reviewPrefs } from '$lib/client/review-prefs.svelte';

	interface Props {
		open: boolean;
		onclose: () => void;
	}
	const { open, onclose }: Props = $props();

	function onKey(e: KeyboardEvent) {
		if (e.key === 'Escape') onclose();
	}
</script>

<svelte:window onkeydown={open ? onKey : undefined} />

{#if open}
	<div class="backdrop" role="presentation" onclick={onclose}></div>
	<div class="modal" role="dialog" aria-modal="true" aria-label={t('settingsTitle')}>
		<header>
			<h2>{t('settingsTitle')}</h2>
			<button type="button" class="close" onclick={onclose} aria-label={t('closeModal')}>✕</button>
		</header>
		<div class="body">
			<section>
				<h3>{t('settingsPlaybackHeading')}</h3>
				<label class="row">
					<input type="checkbox" checked={playbackPrefs.autoplay} onchange={() => playbackPrefs.toggleAutoplay()} />
					<span class="rlabel">{t('settingsAutoplay')}</span>
					<small class="hint">{t('settingsAutoplayHint')}</small>
				</label>
				<label class="row">
					<input type="checkbox" checked={playbackPrefs.loop} onchange={() => playbackPrefs.toggleLoop()} />
					<span class="rlabel">{t('settingsLoop')}</span>
					<small class="hint">{t('settingsLoopHint')}</small>
				</label>
			</section>

			<section>
				<h3>{t('settingsWorkflowHeading')}</h3>
				<label class="row">
					<input type="checkbox" checked={reviewPrefs.autoAdvance} onchange={() => reviewPrefs.toggleAutoAdvance()} />
					<span class="rlabel">{t('settingsAutoAdvance')}</span>
					<small class="hint">{t('settingsAutoAdvanceHint')}</small>
				</label>
				<label class="row">
					<input type="checkbox" checked={reviewPrefs.mobileMode} onchange={() => reviewPrefs.toggleMobileMode()} />
					<span class="rlabel">{t('settingsMobileMode')}</span>
					<small class="hint">{t('settingsMobileModeHint')}</small>
				</label>
			</section>

			<section class="coming-soon">
				<h3>{t('settingsLayoutHeading')}</h3>
				<p class="placeholder">{t('settingsComingSoon')}</p>
			</section>

		</div>
	</div>
{/if}

<style>
	.backdrop {
		position: fixed; inset: 0; background: rgba(15, 23, 42, 0.45); z-index: 90;
	}
	.modal {
		position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
		width: min(520px, calc(100vw - 2rem)); max-height: 84vh;
		background: white; border-radius: 10px;
		box-shadow: 0 20px 50px rgba(15, 23, 42, 0.25); z-index: 100;
		display: flex; flex-direction: column; overflow: hidden;
	}
	header {
		display: flex; align-items: center; justify-content: space-between;
		padding: 0.65rem 1rem; border-bottom: 1px solid var(--border, #e2e8f0);
	}
	h2 { margin: 0; font-size: 1rem; }
	.close {
		background: transparent; border: 0; font-size: 1rem;
		cursor: pointer; color: var(--muted, #6b7280); padding: 0.15rem 0.45rem;
		border-radius: 4px;
	}
	.close:hover { color: #0f172a; background: #f1f5f9; }
	.body {
		padding: 0.65rem 1rem 1rem; overflow-y: auto;
		display: flex; flex-direction: column; gap: 1rem;
	}
	section { display: flex; flex-direction: column; gap: 0.5rem; }
	h3 {
		margin: 0; font-size: 0.72rem; font-weight: 700; color: #64748b;
		text-transform: uppercase; letter-spacing: 0.06em;
	}
	.row {
		display: grid; grid-template-columns: auto 1fr; column-gap: 0.55rem; row-gap: 0;
		align-items: center; padding: 0.35rem 0; cursor: pointer;
	}
	.row input { accent-color: #2563eb; }
	.rlabel { font-size: 0.9rem; color: #0f172a; }
	.hint { grid-column: 2; font-size: 0.72rem; color: #64748b; }
	.coming-soon h3 { color: #cbd5e1; }
	.placeholder {
		margin: 0; font-size: 0.78rem; color: #94a3b8; font-style: italic;
		padding: 0.4rem 0.6rem; background: #f8fafc;
		border: 1px dashed #e2e8f0; border-radius: 6px;
	}
</style>

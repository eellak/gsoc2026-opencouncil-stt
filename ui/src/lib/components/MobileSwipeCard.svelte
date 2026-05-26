<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		onInclude: () => void;
		onExclude: () => void;
		onTap?: () => void;
		labelInclude?: string;
		labelExclude?: string;
		children: Snippet;
	}

	const {
		onInclude,
		onExclude,
		onTap,
		labelInclude = '✓ INCLUDE',
		labelExclude = '✗ EXCLUDE',
		children
	}: Props = $props();

	let dx = $state(0);
	let dragging = $state(false);
	let dropping = $state<'include' | 'exclude' | null>(null);
	let startX = 0;
	let startY = 0;
	let startT = 0;
	let cardWidth = 0;
	let cardEl: HTMLDivElement | null = $state(null);
	let pointerId: number | null = null;

	// Below this horizontal travel, treat as a tap (or as scroll intent).
	const TAP_PX = 8;
	// Above this vertical-to-horizontal ratio, abandon the drag — assume scroll.
	const VERT_ABANDON = 1.2;
	// Threshold for committing a decision (fraction of card width).
	const THRESHOLD = 0.3;
	const MIN_THRESHOLD_PX = 90;

	let abandoned = false;

	function threshold(): number {
		return Math.max(MIN_THRESHOLD_PX, cardWidth * THRESHOLD);
	}

	function onPointerDown(e: PointerEvent) {
		// Mouse: only left button. Touch/pen: any.
		if (e.pointerType === 'mouse' && e.button !== 0) return;
		// Ignore interactive children — let buttons/inputs/links work normally.
		const target = e.target as HTMLElement;
		if (target.closest('button, a, input, textarea, select, [contenteditable]')) return;
		dragging = true;
		abandoned = false;
		startX = e.clientX;
		startY = e.clientY;
		startT = performance.now();
		pointerId = e.pointerId;
		cardWidth = cardEl?.getBoundingClientRect().width ?? 600;
		cardEl?.setPointerCapture(e.pointerId);
	}

	function onPointerMove(e: PointerEvent) {
		if (!dragging || abandoned) return;
		const cx = e.clientX - startX;
		const cy = e.clientY - startY;
		// Mostly-vertical motion → abandon (give scroll back).
		if (Math.abs(cy) > Math.abs(cx) * VERT_ABANDON && Math.abs(cy) > TAP_PX) {
			abandoned = true;
			dx = 0;
			return;
		}
		dx = cx;
	}

	function commit(action: 'include' | 'exclude') {
		dropping = action;
		// Animate out, then trigger callback.
		const off = action === 'include' ? cardWidth + 100 : -(cardWidth + 100);
		dx = off;
		setTimeout(() => {
			(action === 'include' ? onInclude : onExclude)();
			// Reset after navigation; if no navigation happens, snap back.
			setTimeout(() => { dx = 0; dropping = null; }, 250);
		}, 180);
	}

	function snapBack() {
		dx = 0;
	}

	function onPointerUp(e: PointerEvent) {
		if (!dragging) return;
		const wasDragging = dragging;
		const wasAbandoned = abandoned;
		dragging = false;
		if (pointerId !== null) {
			try { cardEl?.releasePointerCapture(pointerId); } catch { /* fine */ }
			pointerId = null;
		}
		if (!wasDragging) return;
		const dist = Math.abs(dx);
		const dt = performance.now() - startT;
		// Tap detection: small movement and short press.
		if (!wasAbandoned && dist < TAP_PX && dt < 350) {
			if (onTap) onTap();
			dx = 0;
			return;
		}
		if (wasAbandoned) { dx = 0; return; }
		const thr = threshold();
		if (dx >= thr) commit('include');
		else if (dx <= -thr) commit('exclude');
		else snapBack();
	}

	function onPointerCancel() {
		dragging = false;
		abandoned = false;
		dx = 0;
		if (pointerId !== null) {
			try { cardEl?.releasePointerCapture(pointerId); } catch { /* fine */ }
			pointerId = null;
		}
	}

	const intensity = $derived.by(() => {
		if (!cardWidth) return 0;
		const t = threshold();
		return Math.min(1, Math.abs(dx) / t);
	});
	const rotate = $derived(dx / 25); // deg
	const overlay = $derived<'include' | 'exclude' | null>(
		dx > TAP_PX ? 'include' : dx < -TAP_PX ? 'exclude' : null
	);
</script>

<div class="swipe-wrap">
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		bind:this={cardEl}
		class="swipe-card"
		class:dragging
		class:dropping-include={dropping === 'include'}
		class:dropping-exclude={dropping === 'exclude'}
		style="transform: translateX({dx}px) rotate({rotate}deg);"
		onpointerdown={onPointerDown}
		onpointermove={onPointerMove}
		onpointerup={onPointerUp}
		onpointercancel={onPointerCancel}
	>
		{#if overlay}
			<div
				class="overlay {overlay}"
				style="opacity: {intensity};"
				aria-hidden="true"
			>
				<span class="label">{overlay === 'include' ? labelInclude : labelExclude}</span>
			</div>
		{/if}
		{@render children()}
	</div>
</div>

<style>
	.swipe-wrap {
		position: relative;
		touch-action: pan-y;
		overscroll-behavior-x: contain;
	}
	.swipe-card {
		position: relative;
		background: var(--surface, #fff);
		border: 1px solid var(--border, #e2e8f0);
		border-radius: var(--radius, 10px);
		padding: 1rem 1.1rem;
		box-shadow: var(--shadow, 0 1px 3px rgba(0,0,0,.08));
		transition: transform 0.18s ease-out;
		will-change: transform;
		user-select: none;
		-webkit-user-select: none;
		cursor: grab;
	}
	.swipe-card.dragging { cursor: grabbing; }
	.swipe-card.dragging { transition: none; }
	.swipe-card.dropping-include,
	.swipe-card.dropping-exclude { transition: transform 0.18s ease-in; }

	.overlay {
		position: absolute; inset: 0;
		display: flex; align-items: center; justify-content: center;
		pointer-events: none; z-index: 5;
		border-radius: var(--radius, 10px);
		font-weight: 800; font-size: 1.6rem; letter-spacing: 0.04em;
	}
	.overlay.include {
		background: rgba(34, 197, 94, 0.18);
		color: #15803d;
		border: 3px solid #22c55e;
	}
	.overlay.exclude {
		background: rgba(239, 68, 68, 0.18);
		color: #b91c1c;
		border: 3px solid #ef4444;
	}
	.overlay .label {
		padding: 0.35rem 0.9rem;
		background: rgba(255, 255, 255, 0.85);
		border-radius: 6px;
	}
</style>

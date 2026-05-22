import { browser } from '$app/environment';
import { strings } from './i18n/strings';
import type { Lang } from './shared/taxonomy';

const STORAGE_KEY = 'oc_lang';

function getInitialLang(): Lang {
	if (browser) {
		const stored = localStorage.getItem(STORAGE_KEY);
		if (stored === 'en' || stored === 'el') return stored as Lang;
	}
	return 'el';
}

let lang = $state<Lang>(getInitialLang());

export function getLang() {
	return lang;
}

export function toggleLang() {
	lang = lang === 'el' ? 'en' : 'el';
	if (browser) localStorage.setItem(STORAGE_KEY, lang);
}

export function t(key: keyof typeof strings, params?: Record<string, string | number>): string {
	const raw = strings[key][lang];
	if (!params) return raw;
	// Simple `{name}` substitution. Missing keys leave the placeholder in place.
	return raw.replace(/\{(\w+)\}/g, (_, name) =>
		name in params ? String(params[name]) : `{${name}}`
	);
}

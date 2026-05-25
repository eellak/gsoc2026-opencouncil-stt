/**
 * Persisted reviewer identity — stored in localStorage so it survives page
 * reloads. Empty string means "not yet identified".
 */

const KEY = 'oc:review:username';

function createUserStore() {
	let value = $state<string>(
		typeof localStorage !== 'undefined' ? (localStorage.getItem(KEY) ?? '') : ''
	);

	return {
		get value() { return value; },
		set(name: string) {
			value = name;
			if (typeof localStorage !== 'undefined') localStorage.setItem(KEY, name);
		},
		clear() {
			value = '';
			if (typeof localStorage !== 'undefined') localStorage.removeItem(KEY);
		}
	};
}

export const userStore = createUserStore();

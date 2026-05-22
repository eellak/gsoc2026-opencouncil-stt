import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

// Unit-test config that skips the SvelteKit Vite plugin so pglite-backed
// repo tests boot quickly without dragging the whole Kit dev server in.
export default defineConfig({
	test: {
		include: ['tests/**/*.test.ts'],
		environment: 'node',
		pool: 'forks',
		fileParallelism: false,
		testTimeout: 20_000
	},
	resolve: {
		alias: {
			$lib: resolve(__dirname, 'src/lib')
		}
	}
});

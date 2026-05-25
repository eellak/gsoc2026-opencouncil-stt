import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
	testDir: 'e2e',
	timeout: 60_000,
	fullyParallel: false,
	workers: 1,
	reporter: process.env.CI ? 'list' : [['list'], ['html', { open: 'never' }]],
	use: {
		baseURL: 'http://127.0.0.1:5174',
		trace: 'retain-on-failure',
		video: 'off',
		screenshot: 'only-on-failure'
	},
	projects: [
		{
			name: 'chromium',
			use: { ...devices['Desktop Chrome'] }
		}
	],
	webServer: {
		command: 'bun run dev',
		port: 5174,
		reuseExistingServer: true,
		stdout: 'pipe',
		stderr: 'pipe',
		timeout: 120_000
	}
});

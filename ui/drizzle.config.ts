import { defineConfig } from 'drizzle-kit';

const url = process.env.DATABASE_URL;
if (!url) {
	throw new Error('DATABASE_URL is not set — add it to ui/.env.local');
}

export default defineConfig({
	dialect: 'postgresql',
	schema: './drizzle/schema.ts',
	out: './drizzle/migrations',
	dbCredentials: { url },
	strict: true,
	verbose: true,
});

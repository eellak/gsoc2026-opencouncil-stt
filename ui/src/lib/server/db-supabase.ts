import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from '../../../drizzle/schema';

const url = process.env.DATABASE_URL;
if (!url) {
	throw new Error('DATABASE_URL is not set — required for Supabase Postgres');
}

// Supabase transaction-mode pooler (port 6543) does not support prepared
// statements — `prepare: false` is mandatory.
const client = postgres(url, { prepare: false, max: 1 });

export const db = drizzle(client, { schema });
export { schema };

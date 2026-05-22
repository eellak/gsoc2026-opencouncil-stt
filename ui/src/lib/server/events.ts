import { getReadyClient } from './db';

export interface LabelEvent {
	ts: string;
	edit_id: string;
	field: string;
	old: unknown;
	new: unknown;
	actor: 'human';
}

export function appendEvent(event: LabelEvent): void {
	getReadyClient().then(client =>
		client.execute({
			sql: 'INSERT INTO events (ts, edit_id, field, old_val, new_val, actor) VALUES (?, ?, ?, ?, ?, ?)',
			args: [event.ts, event.edit_id, event.field, JSON.stringify(event.old), JSON.stringify(event.new), event.actor]
		})
	).catch(() => {/* audit log — ignore failures */});
}

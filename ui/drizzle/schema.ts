import {
	pgTable,
	text,
	doublePrecision,
	integer,
	boolean,
	bigserial,
	index,
} from 'drizzle-orm/pg-core';

// Per-meeting metadata, normalised out of `corrections` to stop duplicating
// audio_url/youtube_url/meeting_name/meeting_date/city_id across hundreds of
// thousands of correction rows.
export const meetings = pgTable(
	'meetings',
	{
		meetingId: text('meeting_id').primaryKey(),
		meetingName: text('meeting_name'),
		meetingDate: text('meeting_date'),
		cityId: text('city_id'),
		audioUrl: text('audio_url'),
		audioCdnUrl: text('audio_cdn_url'),
		youtubeUrl: text('youtube_url'),
	},
	(t) => ({
		cityIdx: index('idx_meetings_city').on(t.cityId),
	}),
);

export const corrections = pgTable(
	'corrections',
	{
		editId: text('edit_id').primaryKey(),
		utteranceId: text('utterance_id'),
		meetingId: text('meeting_id').references(() => meetings.meetingId),
		latestPerUtterance: boolean('latest_per_utterance').notNull().default(true),
		editTimestamp: text('edit_timestamp').notNull(),
		editUpdatedAt: text('edit_updated_at'),
		beforeText: text('before_text').notNull(),
		afterText: text('after_text').notNull(),
		editedBy: text('edited_by'),
		utteranceStart: doublePrecision('utterance_start').notNull(),
		utteranceEnd: doublePrecision('utterance_end').notNull(),
		ingestCategory: text('ingest_category'),
		cleaningApplied: text('cleaning_applied'),
	},
	(t) => ({
		ingestCategoryIdx: index('idx_corrections_ingest_category').on(t.ingestCategory),
		utteranceIdx: index('idx_corrections_utterance_id').on(t.utteranceId),
		meetingIdIdx: index('idx_corrections_meeting_id').on(t.meetingId),
		latestIdx: index('idx_corrections_latest').on(t.latestPerUtterance),
	}),
);

export const reviewLabels = pgTable(
	'review_labels',
	{
		editId: text('edit_id')
			.primaryKey()
			.references(() => corrections.editId),
		errorCategory: text('error_category'),
		includeStatus: text('include_status').notNull().default('unreviewed'),
		adjustedStart: doublePrecision('adjusted_start'),
		adjustedEnd: doublePrecision('adjusted_end'),
		reviewerNotes: text('reviewer_notes'),
		humanUpdatedAt: text('human_updated_at'),
	},
	(t) => ({
		statusIdx: index('idx_labels_status').on(t.includeStatus),
		categoryIdx: index('idx_labels_category').on(t.errorCategory),
	}),
);

export const events = pgTable(
	'events',
	{
		id: bigserial('id', { mode: 'number' }).primaryKey(),
		ts: text('ts').notNull(),
		editId: text('edit_id').notNull(),
		field: text('field').notNull(),
		oldVal: text('old_val'),
		newVal: text('new_val'),
		actor: text('actor').notNull(),
	},
	(t) => ({
		editIdIdx: index('idx_events_edit_id').on(t.editId),
	}),
);

export const categoryDescriptions = pgTable('category_descriptions', {
	category: text('category').primaryKey(),
	labelEl: text('label_el').notNull(),
	reasonEl: text('reason_el').notNull(),
	isRejected: integer('is_rejected').notNull().default(0),
});

export type Meeting = typeof meetings.$inferSelect;
export type Correction = typeof corrections.$inferSelect;
export type NewCorrection = typeof corrections.$inferInsert;
export type ReviewLabel = typeof reviewLabels.$inferSelect;
export type NewReviewLabel = typeof reviewLabels.$inferInsert;
export type EventRow = typeof events.$inferSelect;
export type NewEventRow = typeof events.$inferInsert;
export type CategoryDescription = typeof categoryDescriptions.$inferSelect;

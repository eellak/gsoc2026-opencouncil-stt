/**
 * Global test setup.
 *
 * Disable the meeting-eligibility filter by default so the existing suites,
 * whose fixtures use `edited_by: 'alice'/'bob'` (not 'user') and assert
 * full-corpus counts, keep their semantics. Tests that exercise the filter
 * pass `meetingMinHumanUtterances` to the repo load() explicitly, which takes
 * precedence over this env default.
 */
process.env.MEETING_MIN_HUMAN_UTTERANCES = '0';

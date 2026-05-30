import { execSync } from 'node:child_process';
import { writeFileSync, readFileSync, appendFileSync, existsSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const RUNNER = (process.env.OC_RUNNER || 'opencode').toLowerCase();
const LOG_FILE = join(homedir(), `oc-categorize-${RUNNER}.log`);
const STATE_FILE = join(homedir(), `oc-categorize-${RUNNER}.state.json`);
const SERVER_URL = process.env.OC_SERVER_URL || 'http://127.0.0.1:5174';

// Fallback chain — when current model hits quota, rotate to the next.
const DEFAULT_MODELS =
	RUNNER === 'gemini'
		? 'gemini-2.5-pro,gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash,gemini-2.0-flash-lite,gemini-1.5-pro,gemini-1.5-flash,gemini-1.5-flash-8b'
		: [
				// OpenCode Zen free tier
				'opencode/big-pickle',
				'opencode/deepseek-v4-flash-free',
				'opencode/nemotron-3-super-free',
				// Nvidia — confirmed working/free
				'nvidia/deepseek-ai/deepseek-v4-flash',
				'nvidia/qwen/qwen3.5-122b-a10b',
				'nvidia/mistralai/mistral-large-3-675b-instruct-2512',
				'nvidia/deepseek-ai/deepseek-v4-pro',
				'nvidia/mistralai/mistral-small-4-119b-2603',
				'nvidia/qwen/qwen3-coder-480b-a35b-instruct',
				'nvidia/z-ai/glm-5.1',
		  ].join(',');
const MODELS = (
	process.env.OC_OPENCODE_MODELS ||
	process.env.OC_GEMINI_MODELS ||
	DEFAULT_MODELS
)
	.split(',')
	.map((s) => s.trim())
	.filter(Boolean);

let modelIdx = 0;
const exhausted = new Set<string>();
const consecFailures: Record<string, number> = {};
const FAILURE_ROTATE_THRESHOLD = 3;
function currentModel(): string | null {
	while (modelIdx < MODELS.length && exhausted.has(MODELS[modelIdx])) modelIdx++;
	return modelIdx < MODELS.length ? MODELS[modelIdx] : null;
}

interface BatchResponse {
	batch_id: string;
	model: string;
	items: any[];
	prompt: string;
	remaining: number;
}

interface IngestResponse {
	accepted: number;
	declined: string[];
	duplicates: string[];
	rejected: string[];
	missing_from_batch: string[];
	extra_ids: string[];
}

function log(msg: string) {
	const timestamp = new Date().toISOString();
	const line = `[${timestamp}] ${msg}\n`;
	process.stdout.write(line);
	appendFileSync(LOG_FILE, line);
}

function updateState(batchId: string, result: IngestResponse) {
	let state = { processed: [] as string[], totals: { accepted: 0, declined: 0, rejected: 0, duplicates: 0 } };
	if (existsSync(STATE_FILE)) {
		try {
			state = JSON.parse(readFileSync(STATE_FILE, 'utf8'));
		} catch (e) {
			log(`Error parsing state file: ${e}`);
		}
	}
	state.processed.push(batchId);
	state.totals.accepted += result.accepted;
	state.totals.declined += result.declined.length;
	state.totals.rejected += result.rejected.length;
	state.totals.duplicates += result.duplicates.length;
	writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

async function run() {
	let consecutiveEmpty = 0;
	log('Starting categorization loop...');

	while (true) {
		const model = currentModel();
		if (!model) {
			log('All models exhausted — stopping');
			break;
		}
		try {
			log(`Fetching next batch (model=${model})...`);
			const nextRes = await fetch(`${SERVER_URL}/api/llm-batch/next`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ n: 100, model, include_empty: true })
			});

			if (!nextRes.ok) {
				if (nextRes.status === 429) {
					log('Rate limited by server. Waiting 60s...');
					await new Promise((r) => setTimeout(r, 60000));
					continue;
				}
				log(`Server error: ${nextRes.status} ${nextRes.statusText}`);
				process.exit(1);
			}

			const batch = (await nextRes.json()) as BatchResponse;
			if (!batch.items || batch.items.length === 0) {
				log('Queue empty — stopping');
				break;
			}

			log(`Batch ${batch.batch_id} issued (${batch.items.length} items, ${batch.remaining} remaining). Sending prompt to Gemini...`);

			const promptFile = join(process.cwd(), `.temp_prompt_${batch.batch_id}.txt`);
			writeFileSync(promptFile, batch.prompt);

			const tStart = Date.now();
			let reply = '';
			try {
				const cmd =
					RUNNER === 'gemini'
						? `gemini -y -m ${model} -p "$(cat "${promptFile}")"`
						: `opencode run --model ${model} --dangerously-skip-permissions "$(cat "${promptFile}")"`;
				reply = execSync(cmd, {
					encoding: 'utf8',
					maxBuffer: 32 * 1024 * 1024,
					shell: '/bin/bash',
					timeout: 8 * 60 * 1000,
					killSignal: 'SIGKILL'
				});
			} catch (err: any) {
				const msg = (err.message || '') + '\n' + (err.stderr?.toString?.() || '');
				log(`OpenCode error (${model}): ${msg.slice(0, 500)}`);
				const isQuota =
					/429|quota|rate.?limit|RESOURCE_EXHAUSTED|exceeded|daily limit/i.test(msg);
				const isTimeout = /ETIMEDOUT|timed? ?out/i.test(msg);
				consecFailures[model] = (consecFailures[model] || 0) + 1;
				if (isQuota) {
					log(`Model ${model} hit quota — marking exhausted and rotating`);
					exhausted.add(model);
					modelIdx++;
					if (existsSync(promptFile)) unlinkSync(promptFile);
					continue;
				}
				if (consecFailures[model] >= FAILURE_ROTATE_THRESHOLD) {
					log(
						`Model ${model} has ${consecFailures[model]} consecutive failures (likely throttled) — marking exhausted and rotating`
					);
					exhausted.add(model);
					modelIdx++;
					if (existsSync(promptFile)) unlinkSync(promptFile);
					continue;
				}
				const waitMs = isTimeout ? 30000 : 15000;
				log(`Transient error, waiting ${waitMs / 1000}s before retry (${consecFailures[model]}/${FAILURE_ROTATE_THRESHOLD})`);
				await new Promise((r) => setTimeout(r, waitMs));
				if (existsSync(promptFile)) unlinkSync(promptFile);
				continue;
			} finally {
				if (existsSync(promptFile)) {
					unlinkSync(promptFile);
				}
			}

			if (!reply || reply.trim().length === 0) {
				log(`Received empty reply for ${batch.batch_id}.`);
				consecutiveEmpty++;
			} else {
				const elapsedMs = Date.now() - tStart;
				log(`Received reply for ${batch.batch_id} (${reply.length} chars) in ${(elapsedMs / 1000).toFixed(1)}s (${(elapsedMs / batch.items.length).toFixed(0)}ms/item). Ingesting...`);
				// Strip ANSI escape sequences (opencode emits color codes), then trim
				// to the first '[' / last ']' to tolerate trailing/leading prose.
				let cleaned = reply.replace(/\x1b\[[0-9;]*[A-Za-z]/g, '');
				const firstBracket = cleaned.indexOf('[');
				const lastBracket = cleaned.lastIndexOf(']');
				if (firstBracket >= 0 && lastBracket > firstBracket) {
					cleaned = cleaned.slice(firstBracket, lastBracket + 1);
				}
				const ingestRes = await fetch(`${SERVER_URL}/api/llm-batch/ingest`, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ batch_id: batch.batch_id, raw: cleaned })
				});

				if (!ingestRes.ok) {
					log(`Ingest error: ${ingestRes.status} ${ingestRes.statusText}`);
					const errorText = await ingestRes.text();
					log(`Response: ${errorText}`);
					consecFailures[model] = (consecFailures[model] || 0) + 1;
					if (consecFailures[model] >= FAILURE_ROTATE_THRESHOLD) {
						log(`Model ${model} produced ${consecFailures[model]} malformed replies — marking exhausted and rotating`);
						exhausted.add(model);
						modelIdx++;
						consecutiveEmpty = 0;
						continue;
					}
				} else {
					const result = (await ingestRes.json()) as IngestResponse;
					log(
						`Result: accepted=${result.accepted}, declined=${result.declined.length}, rejected=${result.rejected.length}, duplicates=${result.duplicates.length}`
					);

					updateState(batch.batch_id, result);

					if (result.accepted + result.declined.length === 0) {
						consecutiveEmpty++;
					} else {
						consecutiveEmpty = 0;
						consecFailures[model] = 0;
					}
				}
			}

			if (consecutiveEmpty >= 3) {
				log('3 consecutive empty / malformed batches — stopping');
				break;
			}
		} catch (err: any) {
			log(`Unexpected error: ${err.message}`);
			// Wait a bit before exiting or retrying
			await new Promise((r) => setTimeout(r, 5000));
			process.exit(1);
		}
	}
}

run();

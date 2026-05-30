import { writeFileSync, readFileSync, appendFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const STATE_FILE = join(homedir(), 'oc-antigravity.state.json');
const LOG_FILE = join(homedir(), 'oc-antigravity.log');
const SERVER_URL = 'http://127.0.0.1:5174';

interface State {
  remaining: number;
  by_source: Record<string, number>;
  empty_streak: number;
  shutdown_flag: boolean;
  workers: Record<string, {
    status: 'idle' | 'running';
    batch_id: string | null;
    conversation_id: string | null;
    consecutive_empty: number;
    failed: boolean;
  }>;
  totals: {
    accepted: number;
    declined: number;
    rejected: number;
    duplicates: number;
  };
}

function loadState(): State {
  if (existsSync(STATE_FILE)) {
    try {
      return JSON.parse(readFileSync(STATE_FILE, 'utf8'));
    } catch (e) {
      console.error(`Error reading state file: ${e}`);
    }
  }
  return {
    remaining: 0,
    by_source: {},
    empty_streak: 0,
    shutdown_flag: false,
    workers: {},
    totals: { accepted: 0, declined: 0, rejected: 0, duplicates: 0 }
  };
}

function saveState(state: State) {
  writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function logCoordinator(msg: string) {
  const timestamp = new Date().toISOString();
  const line = `[${timestamp}] ${msg}\n`;
  process.stderr.write(line);
  appendFileSync(LOG_FILE, line);
}

function logWorker(workerId: string, batchId: string, result: { accepted: number; declined: number; rejected: number; duplicates: number }) {
  const timestamp = new Date().toISOString();
  const workerLogFile = join(homedir(), `oc-antigravity-worker-${workerId}.log`);
  const logEntry = JSON.stringify({
    ts: timestamp,
    worker_id: parseInt(workerId, 10),
    batch_id: batchId,
    accepted: result.accepted,
    declined: result.declined,
    rejected: result.rejected,
    duplicates: result.duplicates
  });
  appendFileSync(workerLogFile, logEntry + '\n');
}

function init() {
  const state: State = {
    remaining: 0,
    by_source: {},
    empty_streak: 0,
    shutdown_flag: false,
    workers: {
      "1": { status: "idle", batch_id: null, conversation_id: null, consecutive_empty: 0, failed: false },
      "2": { status: "idle", batch_id: null, conversation_id: null, consecutive_empty: 0, failed: false },
      "3": { status: "idle", batch_id: null, conversation_id: null, consecutive_empty: 0, failed: false },
      "4": { status: "idle", batch_id: null, conversation_id: null, consecutive_empty: 0, failed: false }
    },
    totals: { accepted: 0, declined: 0, rejected: 0, duplicates: 0 }
  };
  saveState(state);
  logCoordinator("Initialized state file.");
  console.log(JSON.stringify(state));
}

async function next(workerId: string) {
  const state = loadState();
  if (state.shutdown_flag) {
    console.log(JSON.stringify({ error: "Shutdown flag set, no new batches." }));
    return;
  }
  const worker = state.workers[workerId];
  if (!worker) {
    console.log(JSON.stringify({ error: `Worker ${workerId} not found.` }));
    return;
  }

  logCoordinator(`Worker ${workerId} pulling next batch...`);
  try {
    const nextRes = await fetch(`${SERVER_URL}/api/llm-batch/next`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n: 200, model: 'gemini-3.5-flash', include_empty: true })
    });

    if (!nextRes.ok) {
      logCoordinator(`Worker ${workerId} /next failed: ${nextRes.status} ${nextRes.statusText}`);
      console.log(JSON.stringify({ error: `HTTP ${nextRes.status} ${nextRes.statusText}` }));
      return;
    }

    const batch = await nextRes.json() as {
      batch_id: string;
      model: string;
      items: any[];
      prompt: string;
      remaining: number;
    };

    if (!batch.items || batch.items.length === 0) {
      logCoordinator(`Worker ${workerId} pulled empty batch (remaining ${batch.remaining})`);
      worker.status = 'idle';
      worker.batch_id = null;
      saveState(state);
      console.log(JSON.stringify({ itemsCount: 0 }));
      return;
    }

    worker.status = 'running';
    worker.batch_id = batch.batch_id;
    saveState(state);

    logCoordinator(`Worker ${workerId} pulled batch ${batch.batch_id} (${batch.items.length} items)`);
    console.log(JSON.stringify({
      batch_id: batch.batch_id,
      prompt: batch.prompt,
      remaining: batch.remaining,
      itemsCount: batch.items.length
    }));
  } catch (err: any) {
    logCoordinator(`Worker ${workerId} error in next: ${err.message}`);
    console.log(JSON.stringify({ error: err.message }));
  }
}

async function ingest(workerId: string, batchId: string, rawFile: string) {
  const state = loadState();
  const worker = state.workers[workerId];
  if (!worker) {
    console.log(JSON.stringify({ error: `Worker ${workerId} not found.` }));
    return;
  }

  logCoordinator(`Worker ${workerId} ingesting batch ${batchId}...`);
  try {
    const raw = readFileSync(rawFile, 'utf8');
    const ingestRes = await fetch(`${SERVER_URL}/api/llm-batch/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batch_id: batchId, raw })
    });

    if (!ingestRes.ok) {
      const errorText = await ingestRes.text();
      logCoordinator(`Worker ${workerId} /ingest failed with status ${ingestRes.status}: ${errorText}`);
      console.log(JSON.stringify({ error: `HTTP ${ingestRes.status}`, text: errorText }));
      return;
    }

    const result = await ingestRes.json() as {
      accepted: number;
      declined: string[];
      duplicates: string[];
      rejected: string[];
      missing_from_batch: string[];
      extra_ids: string[];
    };

    const accepted = result.accepted;
    const declined = result.declined?.length ?? 0;
    const rejected = result.rejected?.length ?? 0;
    const duplicates = result.duplicates?.length ?? 0;

    logCoordinator(`Worker ${workerId} ingest result: accepted=${accepted}, declined=${declined}, rejected=${rejected}, duplicates=${duplicates}`);
    logWorker(workerId, batchId, { accepted, declined, rejected, duplicates });

    // Update state
    worker.status = 'idle';
    worker.batch_id = null;
    worker.conversation_id = null;
    state.totals.accepted += accepted;
    state.totals.declined += declined;
    state.totals.rejected += rejected;
    state.totals.duplicates += duplicates;

    if (accepted + declined === 0) {
      worker.consecutive_empty++;
      logCoordinator(`Worker ${workerId} consecutive empty/garbage count: ${worker.consecutive_empty}`);
      if (worker.consecutive_empty >= 3) {
        worker.failed = true;
        logCoordinator(`Worker ${workerId} terminated: 3 consecutive empty/garbage batches.`);
      }
    } else {
      worker.consecutive_empty = 0;
    }

    saveState(state);
    console.log(JSON.stringify(result));
  } catch (err: any) {
    logCoordinator(`Worker ${workerId} error in ingest: ${err.message}`);
    console.log(JSON.stringify({ error: err.message }));
  }
}

async function stats() {
  const state = loadState();
  try {
    const statsRes = await fetch(`${SERVER_URL}/api/llm-batch/stats`);
    if (!statsRes.ok) {
      logCoordinator(`Stats poll failed: ${statsRes.status}`);
      console.log(JSON.stringify({ error: `HTTP ${statsRes.status}` }));
      return;
    }

    const currentStats = await statsRes.json() as {
      total: number;
      labeled: number;
      remaining: number;
      by_source: Record<string, number>;
    };

    const prevRemaining = state.remaining;
    state.remaining = currentStats.remaining;
    state.by_source = currentStats.by_source;

    // Log stats line
    const activeWorkers = Object.values(state.workers).filter(w => w.status === 'running').length;
    logCoordinator(`Stats: remaining=${currentStats.remaining}, labeled=${currentStats.labeled}/${currentStats.total}, active_workers=${activeWorkers}, totals_processed=${JSON.stringify(state.totals)}`);

    // Check progress stop condition (reuse activeWorkers computed above)
    if (currentStats.remaining === prevRemaining) {
      if (activeWorkers === 0) {
        state.empty_streak++;
        logCoordinator(`Remaining flat and no active workers. Empty streak: ${state.empty_streak}`);
      } else {
        logCoordinator(`Remaining flat but ${activeWorkers} workers are active. Not incrementing empty streak.`);
      }
    } else {
      state.empty_streak = 0;
    }

    if (state.empty_streak >= 3 || currentStats.remaining === 0) {
      state.shutdown_flag = true;
      logCoordinator(`Shutdown condition met (remaining: ${currentStats.remaining}, streak: ${state.empty_streak})`);
    }

    saveState(state);
    console.log(JSON.stringify(currentStats));
  } catch (err: any) {
    logCoordinator(`Error in stats: ${err.message}`);
    console.log(JSON.stringify({ error: err.message }));
  }
}

function registerSubagent(workerId: string, conversationId: string) {
  const state = loadState();
  const worker = state.workers[workerId];
  if (!worker) {
    console.log(JSON.stringify({ error: `Worker ${workerId} not found.` }));
    return;
  }
  worker.conversation_id = conversationId;
  saveState(state);
  logCoordinator(`Worker ${workerId} mapped to subagent conversation ID: ${conversationId}`);
  console.log(JSON.stringify(worker));
}

function failWorker(workerId: string, batchId: string, reason: string) {
  const state = loadState();
  const worker = state.workers[workerId];
  if (!worker) {
    console.log(JSON.stringify({ error: `Worker ${workerId} not found.` }));
    return;
  }
  logCoordinator(`Worker ${workerId} failed: ${reason}`);
  logWorker(workerId, batchId, { accepted: 0, declined: 0, rejected: 1, duplicates: 0 });
  worker.status = 'idle';
  worker.batch_id = null;
  worker.conversation_id = null;
  worker.failed = true;
  saveState(state);
  console.log(JSON.stringify(worker));
}

const args = process.argv.slice(2);
const command = args[0];

function getArg(name: string): string | null {
  const idx = args.indexOf(name);
  if (idx !== -1 && idx + 1 < args.length) {
    return args[idx + 1];
  }
  return null;
}

async function run() {
  switch (command) {
    case 'init':
      init();
      break;
    case 'next': {
      const workerId = getArg('--workerId');
      if (!workerId) {
        console.error("Missing --workerId");
        process.exit(1);
      }
      await next(workerId);
      break;
    }
    case 'ingest': {
      const workerId = getArg('--workerId');
      const batchId = getArg('--batchId');
      const rawFile = getArg('--rawFile');
      if (!workerId || !batchId || !rawFile) {
        console.error("Missing options for ingest");
        process.exit(1);
      }
      await ingest(workerId, batchId, rawFile);
      break;
    }
    case 'stats':
      await stats();
      break;
    case 'register-subagent': {
      const workerId = getArg('--workerId');
      const conversationId = getArg('--conversationId');
      if (!workerId || !conversationId) {
        console.error("Missing options for register-subagent");
        process.exit(1);
      }
      registerSubagent(workerId, conversationId);
      break;
    }
    case 'fail': {
      const workerId = getArg('--workerId');
      const batchId = getArg('--batchId');
      const reason = getArg('--reason');
      if (!workerId || !batchId || !reason) {
        console.error("Missing options for fail");
        process.exit(1);
      }
      failWorker(workerId, batchId, reason);
      break;
    }
    default:
      console.error(`Unknown command: ${command}`);
      process.exit(1);
  }
}

run();

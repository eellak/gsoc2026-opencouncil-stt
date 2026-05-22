const fs = require('fs');
const path = require('path');

const batchPath = '/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments/batch-441.json';
const outDir = '/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments';

const data = JSON.parse(fs.readFileSync(batchPath, 'utf8'));

const chunkSize = 200;
for (let i = 0; i < 5; i++) {
  const start = i * chunkSize;
  const end = start + chunkSize;
  const chunk = data.slice(start, end);
  const outPath = path.join(outDir, `batch-441.chunk-${i + 1}.json`);
  fs.writeFileSync(outPath, JSON.stringify(chunk, null, 2), 'utf8');
  console.log(`Wrote chunk ${i + 1} (${chunk.length} items) to ${outPath}`);
}

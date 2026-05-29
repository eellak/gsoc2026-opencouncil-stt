import { execSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';

for (let w = 1; w <= 4; w++) {
  try {
    const res = execSync(`bun scripts/oc-coordinator-helper.ts next --workerId ${w}`, { encoding: 'utf8' });
    const parsed = JSON.parse(res);
    if (parsed.error) {
      console.log(`Worker ${w} error: ${parsed.error}`);
    } else if (parsed.itemsCount === 0) {
      console.log(`Worker ${w} queue empty.`);
    } else {
      console.log(`Worker ${w} pulled batch ${parsed.batch_id}`);
      writeFileSync(`prompt_${w}.txt`, parsed.prompt);
    }
  } catch (err: any) {
    console.log(`Worker ${w} execution error: ${err.message}`);
  }
}

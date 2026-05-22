#!/usr/bin/env node
/**
 * Post-install hook: patch svelte/package.json so the compiler export
 * points at the pre-built CJS bundle instead of raw source (which
 * requires compilation and fails under certain bundler configs).
 *
 * Also creates a tiny ESM wrapper (compiler-esm.js) that re-exports
 * the CJS build's named exports.
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';

const svelteDir = join(process.cwd(), 'node_modules', 'svelte');
const pkgPath = join(svelteDir, 'package.json');

if (!existsSync(pkgPath)) {
	console.log('⏭ svelte not installed yet, skipping patch');
	process.exit(0);
}

// 1. Patch package.json: compiler export → pre-built bundle
const pkg = JSON.parse(readFileSync(pkgPath, 'utf8'));
const compilerExport = pkg.exports?.['./compiler'];

if (compilerExport) {
	const current = compilerExport.default;
	if (current && current.includes('src/compiler')) {
		compilerExport.default = './compiler-esm.js';
		compilerExport.import = './compiler-esm.js';
		writeFileSync(pkgPath, JSON.stringify(pkg, null, '\t') + '\n');
		console.log('✓ Patched svelte/package.json (./compiler → pre-built bundle)');
	} else {
		console.log('✓ svelte/package.json already patched');
	}
} else {
	console.log('⏭ No ./compiler export found in svelte/package.json');
}

// 2. Create ESM shim for the compiler
const shimPath = join(svelteDir, 'compiler-esm.js');
if (!existsSync(shimPath)) {
	const shim = `// ESM wrapper for svelte/compiler — re-exports named exports from the CJS build
import cjs from './compiler/index.js';
export const {
\tVERSION, compile, compileModule, parse, parseCss, migrate, preprocess, print
} = cjs;
export default cjs;
`;
	writeFileSync(shimPath, shim);
	console.log('✓ Created svelte/compiler-esm.js shim');
} else {
	console.log('✓ svelte/compiler-esm.js already exists');
}

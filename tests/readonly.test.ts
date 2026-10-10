// The TypeScript side must never be able to place an order: no POST/DELETE/PUT, no order endpoints, no key material.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { readOnlyFetch } from '../src/kalshi/gateway.ts';

const files = (d: string): string[] => readdirSync(d).flatMap((f) => { const p = join(d, f); return statSync(p).isDirectory() ? files(p) : [p]; });
const SRC = new URL('../src', import.meta.url).pathname;

test('src/ contains no order endpoints, write methods or credential reads', () => {
  for (const f of files(SRC)) {
    const s = readFileSync(f, 'utf8');
    for (const bad of [/\/portfolio\//, /method:\s*['"](POST|PUT|DELETE|PATCH)/i, /KALSHI_(KEY|PRIVATE)/, /private[_-]?key/i, /createSign/])
      assert.ok(!bad.test(s), `${f} matches ${bad}`);
  }
});

test('readOnlyFetch refuses non-GET', () => {
  const f = readOnlyFetch((async () => new Response('{}')) as any);
  assert.throws(() => f('https://x', { method: 'POST' }), /SHADOW_READ_ONLY/);
});

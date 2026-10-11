// Loads a league's model: the drive tables plus, when present, the end-of-half play table.
import { existsSync, readFileSync } from 'node:fs';
import { DriveModel } from './drive.ts';
import { PlayModel } from './plays.ts';

const root = new URL('../..', import.meta.url).pathname;
export function loadModel(league: string, season?: number, overrides: Record<string, unknown> = {}): DriveModel {
  const lg = league.toLowerCase();
  const j = { ...JSON.parse(readFileSync(`${root}data/models/${lg}_drive_v2.json`, 'utf8')), ...overrides };
  const pp = `${root}data/models/${lg}_plays_v1.json`;
  const plays = existsSync(pp) && overrides.plays !== false ? new PlayModel(JSON.parse(readFileSync(pp, 'utf8'))) : null;
  return new DriveModel(j, season, plays);
}

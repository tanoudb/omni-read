import type { HistoryPatch } from '../types';

export const createHistoryPatch = (description: string): HistoryPatch => ({
  id: crypto.randomUUID(),
  timestamp: Date.now(),
  description,
});

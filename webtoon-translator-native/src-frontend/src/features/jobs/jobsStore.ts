import { create } from 'zustand';
import type { JobsStoreActions, JobsStoreState } from '../../shared/types';

type JobsStore = JobsStoreState & JobsStoreActions;

export const useJobsStore = create<JobsStore>((set) => ({
  activeJobId: null,
  status: 'idle',
  logs: [],
  lastError: null,
  startJob: (jobId) => set({ activeJobId: jobId, status: 'queued', logs: [], lastError: null }),
  setStatus: (status) => set({ status }),
  pushLogs: (logs) => set((state) => ({ logs: [...state.logs, ...logs] })),
  setError: (message) => set({ lastError: message }),
  reset: () => set({ activeJobId: null, status: 'idle', logs: [], lastError: null }),
}));

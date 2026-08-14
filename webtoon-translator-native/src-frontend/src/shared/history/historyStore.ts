import { create } from 'zustand';
import { Patch, applyPatches } from 'immer';
import { useProjectStore } from '../../features/project/projectStore';

interface HistoryEntry {
  patches: Patch[];
  inversePatches: Patch[];
}

interface HistoryStoreState {
  past: HistoryEntry[];
  future: HistoryEntry[];
  addHistory: (patches: Patch[], inversePatches: Patch[]) => void;
  undo: () => void;
  redo: () => void;
  clearHistory: () => void;
}

export const useHistoryStore = create<HistoryStoreState>((set, get) => ({
  past: [],
  future: [],

  addHistory: (patches, inversePatches) => {
    if (patches.length === 0) return;
    set((state) => ({
      past: [...state.past, { patches, inversePatches }].slice(-50), // Limite à 50
      future: [],
    }));
  },

  undo: () => {
    const { past, future } = get();
    if (past.length === 0) return;

    const previous = past[past.length - 1];
    const newPast = past.slice(0, past.length - 1);

    // Appliquer les inversePatches sur le projet actuel
    const currentProject = useProjectStore.getState().project;
    if (currentProject) {
      const nextProject = applyPatches(currentProject, previous.inversePatches);
      useProjectStore.getState().loadProject(nextProject); // Recharge le projet sans marquer dirty = false
      useProjectStore.getState().markDirty();
    }

    set({
      past: newPast,
      future: [previous, ...future],
    });
  },

  redo: () => {
    const { past, future } = get();
    if (future.length === 0) return;

    const next = future[0];
    const newFuture = future.slice(1);

    // Appliquer les patches sur le projet actuel
    const currentProject = useProjectStore.getState().project;
    if (currentProject) {
      const nextProject = applyPatches(currentProject, next.patches);
      useProjectStore.getState().loadProject(nextProject);
      useProjectStore.getState().markDirty();
    }

    set({
      past: [...past, next],
      future: newFuture,
    });
  },

  clearHistory: () => set({ past: [], future: [] }),
}));

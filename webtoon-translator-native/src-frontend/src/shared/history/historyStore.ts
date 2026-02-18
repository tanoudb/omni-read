import { create } from 'zustand';
import type { HistoryPatch, HistoryStoreActions, HistoryStoreState } from '../types';

type HistoryStore = HistoryStoreState & HistoryStoreActions;

export const useHistoryStore = create<HistoryStore>((set) => ({
  undoStack: [],
  redoStack: [],
  limit: 50,
  pushPatch: (patch: HistoryPatch) =>
    set((state) => {
      const nextUndo = [...state.undoStack, patch].slice(-state.limit);
      return { undoStack: nextUndo, redoStack: [] };
    }),
  undo: () => set(() => ({})),
  redo: () => set(() => ({})),
  clear: () => set({ undoStack: [], redoStack: [] }),
}));

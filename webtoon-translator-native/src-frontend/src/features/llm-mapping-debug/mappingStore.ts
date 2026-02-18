import { create } from 'zustand';
import type { MappingStoreActions, MappingStoreState } from '../../shared/types';

type MappingStore = MappingStoreState & MappingStoreActions;

export const useMappingStore = create<MappingStore>((set) => ({
  rows: [],
  setRows: (rows) => set({ rows }),
  updateOutputIndex: (bubbleId, outputIndex) =>
    set((state) => ({
      rows: state.rows.map((row) =>
        row.bubble_id === bubbleId ? { ...row, output_index: outputIndex } : row
      ),
    })),
}));

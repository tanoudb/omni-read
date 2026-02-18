import { create } from 'zustand';
import type { RenderStoreActions, RenderStoreState } from '../../shared/types';

type RenderStore = RenderStoreState & RenderStoreActions;

export const useRenderStore = create<RenderStore>((set) => ({
  isRendering: false,
  previewPath: null,
  setRendering: (value) => set({ isRendering: value }),
  setPreviewPath: (path) => set({ previewPath: path }),
}));

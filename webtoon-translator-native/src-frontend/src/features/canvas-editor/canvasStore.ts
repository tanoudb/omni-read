import { create } from 'zustand';
import type { CanvasStoreActions, CanvasStoreState } from '../../shared/types';

type CanvasStore = CanvasStoreState & CanvasStoreActions;

export const useCanvasStore = create<CanvasStore>((set) => ({
  activePageId: null,
  activeBubbleId: null,
  tool: 'select',
  viewport: {
    zoom: 1,
    pan_x: 0,
    pan_y: 0,
    show_translated: true,
  },
  showTranslated: true,
  zoomEnabled: false,
  setTool: (tool) => set({ tool }),
  setActivePage: (activePageId) => set({ activePageId }),
  setActiveBubble: (activeBubbleId) => set({ activeBubbleId }),
  setViewport: (viewport) =>
    set((state) => ({ viewport: { ...state.viewport, ...viewport } })),
  setZoomEnabled: (zoomEnabled) => set({ zoomEnabled }),
}));

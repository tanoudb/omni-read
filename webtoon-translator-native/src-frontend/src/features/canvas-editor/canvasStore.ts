import { create } from 'zustand';
import type { CanvasStoreActions, CanvasStoreState } from '../../shared/types';

type Tool = 'select' | 'draw' | 'pan' | 'delete';

interface ExtendedCanvasState extends CanvasStoreState {
  tool: Tool;
  hoveredBubbleId: string | null;
  showOriginal: boolean;
  drawingRect: { x: number; y: number; w: number; h: number } | null;
}

interface ExtendedCanvasActions extends CanvasStoreActions {
  setTool: (tool: Tool) => void;
  setHoveredBubble: (id: string | null) => void;
  setShowOriginal: (show: boolean) => void;
  setDrawingRect: (rect: { x: number; y: number; w: number; h: number } | null) => void;
  toggleShowOriginal: () => void;
}

type CanvasStore = ExtendedCanvasState & ExtendedCanvasActions;

export const useCanvasStore = create<CanvasStore>((set) => ({
  activePageId: null,
  activeBubbleId: null,
  hoveredBubbleId: null,
  tool: 'select',
  showOriginal: false,
  drawingRect: null,
  viewport: {
    zoom: 1,
    pan_x: 0,
    pan_y: 0,
    show_translated: true,
  },
  showTranslated: true,
  zoomEnabled: true,
  setTool: (tool) => set({ tool }),
  setActivePage: (activePageId) => set({ activePageId, activeBubbleId: null }),
  setActiveBubble: (activeBubbleId) => set({ activeBubbleId }),
  setHoveredBubble: (hoveredBubbleId) => set({ hoveredBubbleId }),
  setShowOriginal: (showOriginal) => set({ showOriginal }),
  toggleShowOriginal: () => set((s) => ({ showOriginal: !s.showOriginal })),
  setDrawingRect: (drawingRect) => set({ drawingRect }),
  setViewport: (viewport) =>
    set((state) => ({ viewport: { ...state.viewport, ...viewport } })),
  setZoomEnabled: (zoomEnabled) => set({ zoomEnabled }),
}));

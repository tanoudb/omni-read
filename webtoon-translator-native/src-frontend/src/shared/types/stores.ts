import type { Bubble, Page, Project, ViewportState } from './domain';
import type { JobLogLine } from './api';

export type ToolMode = 'select' | 'draw' | 'moveResize' | 'delete' | 'pan' | 'brush';

export interface ProjectStoreState {
  project: Project | null;
  isDirty: boolean;
  isSaving: boolean;
  lastSavedAt: string | null;
  saveError: string | null;
}

export interface ProjectStoreActions {
  createEmptyProject: () => void;
  loadProject: (project: Project) => void;
  setProject: (project: Project | null) => void;
  setProjectPages: (pages: Page[]) => void;
  setPageBubbles: (pageId: string, bubbles: Bubble[]) => void;
  patchPageBubbles: (pageId: string, patches: Array<Partial<Bubble> & { id: string }>) => void;
  setPagePreviewPath: (pageId: string, previewPath: string | null) => void;
  updateBubbleOverrides: (
    pageId: string,
    bubbleId: string,
    overrides: { source_override?: string | null; translated_override?: string | null }
  ) => void;
  saveProject: () => Promise<void>;
  setSavingState: (isSaving: boolean, error: string | null, time?: string) => void;
  updateGlossary: (glossary: Record<string, string>) => void;
  markDirty: () => void;
}

export interface CanvasStoreState {
  activePageId: string | null;
  activeBubbleId: string | null;
  hoveredBubbleId: string | null;
  tool: ToolMode;
  viewport: ViewportState;
  showTranslated: boolean;
  showOriginal: boolean;
  zoomEnabled: boolean;
  drawingRect: { x: number; y: number; w: number; h: number } | null;
}

export interface CanvasStoreActions {
  setTool: (tool: ToolMode) => void;
  setActivePage: (pageId: string | null) => void;
  setActiveBubble: (bubbleId: string | null) => void;
  setHoveredBubble: (id: string | null) => void;
  setViewport: (viewport: Partial<ViewportState>) => void;
  setZoomEnabled: (enabled: boolean) => void;
  setShowOriginal: (show: boolean) => void;
  toggleShowOriginal: () => void;
  setDrawingRect: (rect: { x: number; y: number; w: number; h: number } | null) => void;
}

export interface JobsStoreState {
  activeJobId: string | null;
  status: 'idle' | 'queued' | 'running' | 'done' | 'failed';
  logs: JobLogLine[];
  lastError: string | null;
}

export interface JobsStoreActions {
  startJob: (jobId: string) => void;
  setStatus: (status: JobsStoreState['status']) => void;
  pushLogs: (logs: JobLogLine[]) => void;
  setError: (message: string | null) => void;
  reset: () => void;
}

export interface MappingStoreState {
  rows: Array<{ bubble_id: string; input_index: number; output_index: number }>;
}

export interface MappingStoreActions {
  setRows: (rows: MappingStoreState['rows']) => void;
  updateOutputIndex: (bubbleId: string, outputIndex: number) => void;
}

export interface SettingsStoreState {
  apiBaseUrl: string;
  cacheEnabled: boolean;
}

export interface SettingsStoreActions {
  setApiBaseUrl: (value: string) => void;
  setCacheEnabled: (value: boolean) => void;
}

export interface HistoryPatch {
  id: string;
  timestamp: number;
  description: string;
}

export interface HistoryStoreState {
  undoStack: HistoryPatch[];
  redoStack: HistoryPatch[];
  limit: number;
}

export interface HistoryStoreActions {
  pushPatch: (patch: HistoryPatch) => void;
  undo: () => void;
  redo: () => void;
  clear: () => void;
}

export interface ImageListStoreState {
  pages: Page[];
  selectedPageId: string | null;
}

export interface ImageListStoreActions {
  setPages: (pages: Page[]) => void;
  selectPage: (pageId: string | null) => void;
}

export interface RenderStoreState {
  isRendering: boolean;
  previewPath: string | null;
}

export interface RenderStoreActions {
  setRendering: (value: boolean) => void;
  setPreviewPath: (path: string | null) => void;
}

export interface BubbleInspectorState {
  bubble: Bubble | null;
}

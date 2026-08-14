import { create } from 'zustand';
import type { ProjectStoreActions, ProjectStoreState } from '../../shared/types';
import { produceWithPatches } from 'immer';
import { useHistoryStore } from '../../shared/history/historyStore';

type ProjectStore = ProjectStoreState & ProjectStoreActions;

export const useProjectStore = create<ProjectStore>((set) => ({
  project: null,
  isDirty: false,
  isSaving: false,
  lastSavedAt: null,
  saveError: null,
  createEmptyProject: () =>
    set({
      project: {
        schema_version: '1.0.0',
        project_id: crypto.randomUUID(),
        name: 'Nouveau projet',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        settings: {
          source_lang: 'en',
          target_lang: 'fr',
          cache_enabled: true,
        },
        pages: [],
      },
      isDirty: true,
    }),
  loadProject: (project) => {
    useHistoryStore.getState().clearHistory();
    set({ project, isDirty: false });
  },
  setProject: (project) => {
    useHistoryStore.getState().clearHistory();
    set({ project, isDirty: true });
  },
  setProjectPages: (pages) =>
    set((state) => {
      if (!state.project) return {};
      const [nextProject, patches, inversePatches] = produceWithPatches(state.project, (draft) => {
        draft.pages = pages;
        draft.updated_at = new Date().toISOString();
      });
      useHistoryStore.getState().addHistory(patches, inversePatches);
      return { project: nextProject, isDirty: true };
    }),
  setPageBubbles: (pageId, bubbles) =>
    set((state) => {
      if (!state.project) return {};
      const [nextProject, patches, inversePatches] = produceWithPatches(state.project, (draft) => {
        const page = draft.pages.find((p) => p.id === pageId);
        if (page) {
          page.bubbles = bubbles;
        }
        draft.updated_at = new Date().toISOString();
      });
      useHistoryStore.getState().addHistory(patches, inversePatches);
      return { project: nextProject, isDirty: true };
    }),
  patchPageBubbles: (pageId, patchesToApply) =>
    set((state) => {
      if (!state.project) return {};
      const [nextProject, patches, inversePatches] = produceWithPatches(state.project, (draft) => {
        const page = draft.pages.find((p) => p.id === pageId);
        if (page) {
          const patchMap = new Map(patchesToApply.map((p) => [p.id, p]));
          page.bubbles = page.bubbles.map((bubble) => {
            const patch = patchMap.get(bubble.id);
            if (!patch) return bubble;
            return {
              ...bubble,
              ...patch,
              bbox: patch.bbox ?? bubble.bbox,
              text_style: patch.text_style ?? bubble.text_style,
              mask_strokes: patch.mask_strokes ?? bubble.mask_strokes,
              errors: patch.errors ?? bubble.errors,
            };
          });
        }
        draft.updated_at = new Date().toISOString();
      });
      useHistoryStore.getState().addHistory(patches, inversePatches);
      return { project: nextProject, isDirty: true };
    }),
  setPagePreviewPath: (pageId, previewPath) =>
    set((state) => {
      if (!state.project) return {};
      const [nextProject, patches, inversePatches] = produceWithPatches(state.project, (draft) => {
        const page = draft.pages.find((p) => p.id === pageId);
        if (page) {
          page.preview_path = previewPath;
        }
        draft.updated_at = new Date().toISOString();
      });
      // Do not add preview path changes to undo/redo history to avoid visual glitches
      return { project: nextProject, isDirty: true };
    }),
  updateBubbleOverrides: (pageId, bubbleId, overrides) =>
    set((state) => {
      if (!state.project) return {};
      const [nextProject, patches, inversePatches] = produceWithPatches(state.project, (draft) => {
        const page = draft.pages.find((p) => p.id === pageId);
        if (page) {
          const bubble = page.bubbles.find((b) => b.id === bubbleId);
          if (bubble) {
            if (overrides.source_override !== undefined) bubble.source_override = overrides.source_override;
            if (overrides.translated_override !== undefined) bubble.translated_override = overrides.translated_override;
          }
        }
        draft.updated_at = new Date().toISOString();
      });
      useHistoryStore.getState().addHistory(patches, inversePatches);
      return { project: nextProject, isDirty: true };
    }),
  saveProject: async () => {
    set({ isSaving: true, saveError: null });
    const { project } = get();
    if (!project) {
      set({ isSaving: false, saveError: 'No project' });
      return;
    }
    try {
      const data = JSON.stringify(project, null, 2);
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${project.name || 'project'}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      set({ isSaving: false, lastSavedAt: new Date().toISOString(), isDirty: false });
    } catch (err) {
      set({ isSaving: false, saveError: err instanceof Error ? err.message : 'Save failed' });
    }
  },
  setSavingState: (isSaving, error, time) =>
    set((state) => ({
      isSaving,
      saveError: error ?? state.saveError,
      lastSavedAt: time ?? state.lastSavedAt,
      isDirty: isSaving === false && !error ? false : state.isDirty
    })),
  updateGlossary: (glossary) =>
    set((state) => {
      if (!state.project) return state;
      const nextProject = produce(state.project, (draft) => {
        draft.settings.glossary = glossary;
        draft.updated_at = new Date().toISOString();
      });
      return { project: nextProject, isDirty: true };
    }),
  markDirty: () => set({ isDirty: true }),
}));

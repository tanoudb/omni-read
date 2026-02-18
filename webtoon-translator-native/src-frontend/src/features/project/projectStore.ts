import { create } from 'zustand';
import type { ProjectStoreActions, ProjectStoreState } from '../../shared/types';

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
  loadProject: (project) => set({ project, isDirty: false }),
  setProject: (project) => set({ project, isDirty: true }),
  setProjectPages: (pages) =>
    set((state) => {
      if (!state.project) {
        return {};
      }

      return {
        project: {
          ...state.project,
          pages,
          updated_at: new Date().toISOString(),
        },
        isDirty: true,
      };
    }),
  setPageBubbles: (pageId, bubbles) =>
    set((state) => {
      if (!state.project) {
        return {};
      }

      return {
        project: {
          ...state.project,
          pages: state.project.pages.map((page) =>
            page.id === pageId
              ? {
                  ...page,
                  bubbles,
                }
              : page
          ),
          updated_at: new Date().toISOString(),
        },
        isDirty: true,
      };
    }),
  patchPageBubbles: (pageId, patches) =>
    set((state) => {
      if (!state.project) {
        return {};
      }

      const patchMap = new Map(patches.map((patch) => [patch.id, patch]));
      return {
        project: {
          ...state.project,
          pages: state.project.pages.map((page) => {
            if (page.id !== pageId) {
              return page;
            }

            return {
              ...page,
              bubbles: page.bubbles.map((bubble) => {
                const patch = patchMap.get(bubble.id);
                if (!patch) {
                  return bubble;
                }

                return {
                  ...bubble,
                  ...patch,
                  bbox: patch.bbox ?? bubble.bbox,
                  text_style: patch.text_style ?? bubble.text_style,
                  mask_strokes: patch.mask_strokes ?? bubble.mask_strokes,
                  errors: patch.errors ?? bubble.errors,
                };
              }),
            };
          }),
          updated_at: new Date().toISOString(),
        },
        isDirty: true,
      };
    }),
  setPagePreviewPath: (pageId, previewPath) =>
    set((state) => {
      if (!state.project) {
        return {};
      }

      return {
        project: {
          ...state.project,
          pages: state.project.pages.map((page) =>
            page.id === pageId
              ? {
                  ...page,
                  preview_path: previewPath,
                }
              : page
          ),
          updated_at: new Date().toISOString(),
        },
        isDirty: true,
      };
    }),
  updateBubbleOverrides: (pageId, bubbleId, overrides) =>
    set((state) => {
      if (!state.project) {
        return {};
      }

      const nextPages = state.project.pages.map((page) => {
        if (page.id !== pageId) {
          return page;
        }

        return {
          ...page,
          bubbles: page.bubbles.map((bubble) => {
            if (bubble.id !== bubbleId) {
              return bubble;
            }

            return {
              ...bubble,
              source_override:
                overrides.source_override !== undefined
                  ? overrides.source_override
                  : bubble.source_override,
              translated_override:
                overrides.translated_override !== undefined
                  ? overrides.translated_override
                  : bubble.translated_override,
            };
          }),
        };
      });

      return {
        project: {
          ...state.project,
          pages: nextPages,
          updated_at: new Date().toISOString(),
        },
        isDirty: true,
      };
    }),
  saveProject: async () => set({ isSaving: false, lastSavedAt: new Date().toISOString(), saveError: null }),
  markDirty: () => set({ isDirty: true }),
}));

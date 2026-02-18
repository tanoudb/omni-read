import { create } from 'zustand';
import type { ImageListStoreActions, ImageListStoreState } from '../../shared/types';

type ImageListStore = ImageListStoreState & ImageListStoreActions;

export const useImageListStore = create<ImageListStore>((set) => ({
  pages: [],
  selectedPageId: null,
  setPages: (pages) => set({ pages }),
  selectPage: (pageId) => set({ selectedPageId: pageId }),
}));

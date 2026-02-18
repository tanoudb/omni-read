import { create } from 'zustand';
import type { SettingsStoreActions, SettingsStoreState } from '../../shared/types';

type SettingsStore = SettingsStoreState & SettingsStoreActions;

export const useSettingsStore = create<SettingsStore>((set) => ({
  apiBaseUrl: 'http://127.0.0.1:8000/api/v1',
  cacheEnabled: true,
  setApiBaseUrl: (value) => set({ apiBaseUrl: value }),
  setCacheEnabled: (value) => set({ cacheEnabled: value }),
}));

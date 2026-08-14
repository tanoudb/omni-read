import { useEffect } from 'react';
import { useProjectStore } from './projectStore';

// Helper pour IndexedDB
const DB_NAME = 'WebtoonTranslatorDB';
const STORE_NAME = 'autosave';

const getDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

export const saveToAutosaveDB = async (data: string): Promise<void> => {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    store.put(data, 'latest_project');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
};

export const loadFromAutosaveDB = async (): Promise<string | null> => {
  const db = await getDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.get('latest_project');
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
};

export const useAutosave = () => {
  const project = useProjectStore((s) => s.project);
  const isDirty = useProjectStore((s) => s.isDirty);
  const setSavingState = useProjectStore((s) => s.setSavingState);

  useEffect(() => {
    if (!isDirty || !project) return;

    const timeout = setTimeout(async () => {
      try {
        setSavingState(true, null);
        const data = JSON.stringify(project);
        await saveToAutosaveDB(data);
        setSavingState(false, null, new Date().toISOString());
      } catch (err) {
        setSavingState(false, err instanceof Error ? err.message : 'Autosave failed');
      }
    }, 10000);

    return () => clearTimeout(timeout);
  }, [project, isDirty, setSavingState]);
};

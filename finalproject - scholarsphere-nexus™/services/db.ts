
import { openDB, DBSchema, IDBPDatabase } from 'idb';
import { PdfFile, ChatSession } from '../types';

interface ScholarSphereDB extends DBSchema {
  files: {
    key: string;
    value: PdfFile;
  };
  sessions: {
    key: string;
    value: ChatSession;
  };
  keyval: {
    key: string;
    value: any;
  };
}

let dbPromise: Promise<IDBPDatabase<ScholarSphereDB>>;

export const initDB = () => {
  if (!dbPromise) {
    dbPromise = openDB<ScholarSphereDB>('ScholarSphereDB', 2, {
      upgrade(db, oldVersion, newVersion, transaction) {
        // Store for PDFs (Key: ID)
        if (!db.objectStoreNames.contains('files')) {
          db.createObjectStore('files', { keyPath: 'id' });
        }
        // General Key-Value store for settings, state
        if (!db.objectStoreNames.contains('keyval')) {
          db.createObjectStore('keyval');
        }
        // Store for Chat Sessions (Key: ID)
        if (!db.objectStoreNames.contains('sessions')) {
          db.createObjectStore('sessions', { keyPath: 'id' });
        }
      },
    });
  }
  return dbPromise;
};

// --- File Operations ---

export const getAllFiles = async (): Promise<PdfFile[]> => {
  const db = await initDB();
  return db.getAll('files');
};

export const addFile = async (file: PdfFile) => {
  const db = await initDB();
  return db.put('files', file);
};

export const updateFile = async (file: PdfFile) => {
  const db = await initDB();
  return db.put('files', file);
};

export const deleteFile = async (id: string) => {
  const db = await initDB();
  return db.delete('files', id);
};

// --- Session Operations ---

export const getAllSessions = async (): Promise<ChatSession[]> => {
  const db = await initDB();
  return db.getAll('sessions');
};

export const saveSession = async (session: ChatSession) => {
  const db = await initDB();
  return db.put('sessions', session);
};

export const deleteSession = async (id: string) => {
  const db = await initDB();
  return db.delete('sessions', id);
};

// --- Key-Value Operations (Settings) ---

export const getMeta = async (key: string) => {
  const db = await initDB();
  return db.get('keyval', key);
};

export const setMeta = async (key: string, val: any) => {
  const db = await initDB();
  return db.put('keyval', val, key);
};

export const clearAllData = async () => {
  const db = await initDB();
  await db.clear('files');
  await db.clear('keyval');
  await db.clear('sessions');
};

export const getStorageEstimate = async () => {
  if (navigator.storage && navigator.storage.estimate) {
    const estimate = await navigator.storage.estimate();
    return {
      usage: estimate.usage || 0,
      quota: estimate.quota || 0
    };
  }
  return null;
};

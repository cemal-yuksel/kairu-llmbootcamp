

export enum ChatMode {
  STANDARD = 'STANDARD',
  FAST = 'FAST',
  DEEP_THINK = 'DEEP_THINK',
  DEEP_SEARCH = 'DEEP_SEARCH' // New Citation Chain Verification Mode
}

export interface RagChunk {
  id: string;
  fileId: string;
  content: string;
  pageNumber: number;
  tokenCountEstimate: number; // Approx 4 chars per token
  embedding?: number[]; // Vector representation for Semantic Search
  score?: number; // Cosine Similarity Score (0-1)
  sourceTitle?: string; // Hydrated title for display
}

export interface Message {
  id: string;
  role: 'user' | 'model';
  content: string;
  timestamp: Date;
  isThinking?: boolean; // Visual state for loading
  retrievedChunks?: RagChunk[]; // The actual RAG chunks used for this answer
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  selectedFileIds: string[]; // IDs of specific PDFs active for this session
  createdAt: Date;
  lastModified: Date;
}

export enum View {
  LANDING = 'LANDING',
  CHAT = 'CHAT',
  LIBRARY = 'LIBRARY',
  PAPER_XRAY = 'PAPER_XRAY',
  RESEARCH_GAPS = 'RESEARCH_GAPS',
  CO_AUTHOR = 'CO_AUTHOR',
  ACADEMIC_TOOLS = 'ACADEMIC_TOOLS', // Replaces KNOWLEDGE_GRAPH
  PODCAST = 'PODCAST', // New Audio Overview
  SETTINGS = 'SETTINGS'
}

export interface SidebarProps {
  currentView: View;
  setCurrentView: (view: View) => void;
  isMobileOpen: boolean;
  setIsMobileOpen: (isOpen: boolean) => void;
}

export interface GraphEntities {
    concepts: string[];
    authors: string[];
}

export interface PdfFile {
  id: string;
  name: string; // Original filename
  title: string; // Metadata title or fallback to name
  size: number; // in bytes
  uploadDate: Date;
  isSelected: boolean; // Global selection (kept for backward compatibility or bulk ops)
  text?: string; // Kept for backwards compatibility/preview
  chunks: RagChunk[]; // The "Memory" structure
  pageCount?: number;
  isProcessing: boolean; // Loading state during extraction
  graphData?: GraphEntities; // Knowledge Graph Entities
}

export interface ResearchGap {
  title: string;
  description: string;
  impact: string; // Why is this important?
}

// --- TOAST NOTIFICATION TYPES ---
export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface ToastMessage {
  id: string;
  message: string;
  type: ToastType;
}
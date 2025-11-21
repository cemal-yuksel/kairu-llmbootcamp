

import React, { useState, useRef, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import PdfUploadArea from './components/PdfUploadArea';
import PaperXRayModal from './components/PaperXRayModal';
import CoAuthorStudio from './components/CoAuthorStudio';
import PaperXRayView from './components/PaperXRayView';
import ResearchGapView from './components/ResearchGapView';
import AcademicToolsView from './components/AcademicToolsView'; // Updated Import
import PodcastView from './components/PodcastView';
import SettingsView from './components/SettingsView';
import LandingView from './components/LandingView';
import Toast from './components/Toast';
import { View, PdfFile, ToastMessage, ToastType } from './types';
import { Menu } from 'lucide-react';
import { runPaperXRayAnalysis } from './services/geminiService';
import * as pdfjsLib from 'pdfjs-dist';
import * as db from './services/db';

// Ensure PDF worker is set
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@4.8.69/build/pdf.worker.min.mjs`;

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<View>(View.LANDING);
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  
  // Centralized State for Files (Loaded from DB)
  const [libraryFiles, setLibraryFiles] = useState<PdfFile[]>([]);
  
  // PaperX-Ray State
  const [isXRayOpen, setIsXRayOpen] = useState(false);
  const [xRayLoading, setXRayLoading] = useState(false);
  const [xRayContent, setXRayContent] = useState('');
  const [xRayContext, setXRayContext] = useState(''); 
  const [xRaySelectedFile, setXRaySelectedFile] = useState<string>('');
  
  // Toast State
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  // Ref for Quick Upload
  const quickUploadRef = useRef<HTMLInputElement>(null);

  // --- TOAST MANAGER ---
  const addToast = (message: string, type: ToastType = 'info') => {
      const id = Date.now().toString();
      setToasts(prev => [...prev, { id, message, type }]);
  };

  const removeToast = (id: string) => {
      setToasts(prev => prev.filter(t => t.id !== id));
  };

  // --- PERSISTENCE: Library Hydration from IndexedDB ---
  useEffect(() => {
      const loadLibrary = async () => {
          try {
              const files = await db.getAllFiles();
              // Sort by upload date descending
              const sorted = files.sort((a, b) => new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime());
              setLibraryFiles(sorted);
          } catch (e) {
              console.error("Failed to load library from DB", e);
              addToast("Kütüphane yüklenemedi.", 'error');
          }
      };
      loadLibrary();
  }, []);

  // --- DB HANDLERS ---
  const handleAddFiles = async (newFiles: PdfFile[]) => {
      setLibraryFiles(prev => [...newFiles, ...prev]);
      // Persist to DB
      for (const file of newFiles) {
          await db.addFile(file);
      }
  };

  const handleUpdateFile = async (updatedFile: PdfFile) => {
      setLibraryFiles(prev => prev.map(f => f.id === updatedFile.id ? updatedFile : f));
      await db.updateFile(updatedFile);
  };

  const handleDeleteFile = async (id: string) => {
      if (!id) return;

      // Backup current state for potential rollback
      const previousFiles = [...libraryFiles];

      // 1. Optimistic Update: Remove from UI immediately
      setLibraryFiles(prev => prev.filter(f => f.id !== id));

      try {
          // 2. Remove from IndexedDB
          await db.deleteFile(id);
          addToast("Dosya ve verileri kalıcı olarak silindi.", 'success');
      } catch (error) {
          console.error("Delete error", error);
          // 3. Rollback UI if DB fails
          setLibraryFiles(previousFiles);
          addToast("Dosya silinemedi. Lütfen tekrar deneyin.", 'error');
      }
  };

  // Helper: Build context string from a single file
  const buildContextFromFile = (file: PdfFile): string => {
      if (file.chunks && file.chunks.length > 0) {
          return `MAKALENİN ADI: ${file.title}\n` + file.chunks.map(c => c.content).join('\n');
      }
      return `MAKALENİN ADI: ${file.title}\n${file.text || ''}`;
  };

  // Triggered by PaperXRayView
  const handleDirectAnalysis = (file: PdfFile) => {
      setXRaySelectedFile(file.title);
      startXRayProcess(file);
  };

  const startXRayProcess = async (file: PdfFile) => {
    setIsXRayOpen(true);
    setXRayLoading(true);
    setXRayContent('');
    
    const contextString = buildContextFromFile(file);
    setXRayContext(contextString);

    try {
      const report = await runPaperXRayAnalysis(contextString);
      setXRayContent(report);
      addToast("Analiz başarıyla tamamlandı.", 'success');
    } catch (error: any) {
      setXRayContent("Hata: Analiz raporu oluşturulamadı.");
      addToast(error.message || "Analiz sırasında hata oluştu.", 'error');
    } finally {
      setXRayLoading(false);
    }
  };

  const handleQuickUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
          const file = e.target.files[0];
          if (file.type !== 'application/pdf') {
              addToast("Sadece PDF dosyaları desteklenmektedir.", 'warning');
              return;
          }
          
          addToast("Dosya işleniyor...", 'info');

          try {
             const arrayBuffer = await file.arrayBuffer();
             const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
             const pdf = await loadingTask.promise;
             
             // Smart Title Detection (Duplicated from PdfUploadArea logic for consistency)
             let metaTitle = null;
             try {
                const metadata = await pdf.getMetadata().catch(() => ({ info: null }));
                metaTitle = metadata?.info?.Title;
             } catch(e) {
                console.warn("Metadata failed in Quick Upload");
             }

             const cleanFilename = file.name.replace(/\.pdf$/i, '').replace(/_/g, ' ').trim();
             let finalTitle = cleanFilename;

             if (metaTitle && typeof metaTitle === 'string') {
                 const trimmed = metaTitle.trim();
                 const lower = trimmed.toLowerCase();
                 if (trimmed.length > 3 && 
                     lower !== 'untitled' && 
                     !lower.startsWith('microsoft word') &&
                     lower !== cleanFilename.toLowerCase()) {
                     finalTitle = trimmed;
                 }
             }

             let fullText = "";
             for (let i = 1; i <= pdf.numPages; i++) {
                 const page = await pdf.getPage(i);
                 const textContent = await page.getTextContent();
                 fullText += textContent.items.map((item: any) => item.str).join(' ') + "\n";
             }

             const newFile: PdfFile = {
                 id: Math.random().toString(36).substr(2, 9),
                 name: file.name,
                 title: finalTitle,
                 size: file.size,
                 uploadDate: new Date(),
                 isSelected: true,
                 isProcessing: false,
                 text: fullText,
                 chunks: [], 
                 pageCount: pdf.numPages
             };

             // Use new handler
             handleAddFiles([newFile]);
             addToast(`${finalTitle} kütüphaneye eklendi.`, 'success');
             
             if (currentView === View.PAPER_XRAY) {
                 setXRaySelectedFile(newFile.title);
                 startXRayProcess(newFile);
             }

          } catch (error) {
              console.error("Quick Upload Error", error);
              addToast("Dosya okunamadı veya bozuk.", 'error');
          }
      }
  };

  const renderContent = () => {
    switch (currentView) {
      case View.LANDING:
        return <LandingView setCurrentView={setCurrentView} />;
      case View.CHAT:
        // Passing all libraryFiles allows the user to select any file for the chat context, regardless of global selection
        return <ChatArea activeFiles={libraryFiles} addToast={addToast} />;
      case View.LIBRARY:
        return <PdfUploadArea 
                  files={libraryFiles} 
                  onAddFiles={handleAddFiles}
                  onUpdateFile={handleUpdateFile}
                  onDeleteFile={handleDeleteFile}
               />;
      case View.PAPER_XRAY:
        return <PaperXRayView 
                  files={libraryFiles} 
                  onAnalyze={handleDirectAnalysis} 
                  onUploadRequest={() => quickUploadRef.current?.click()} 
               />;
      case View.RESEARCH_GAPS:
        return <ResearchGapView files={libraryFiles} />;
      case View.ACADEMIC_TOOLS: // Updated Case
        return <AcademicToolsView />;
      case View.PODCAST:
        return <PodcastView files={libraryFiles} />;
      case View.CO_AUTHOR:
        return <CoAuthorStudio activeFiles={libraryFiles.filter(f => f.isSelected)} />;
      case View.SETTINGS:
        return <SettingsView />;
      default:
        return <LandingView setCurrentView={setCurrentView} />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#0B0C15] font-sans text-slate-200">
      
      <Toast toasts={toasts} removeToast={removeToast} />

      <input 
        type="file" 
        ref={quickUploadRef} 
        className="hidden" 
        accept="application/pdf" 
        onChange={handleQuickUpload} 
      />

      <Sidebar 
        currentView={currentView} 
        setCurrentView={setCurrentView}
        isMobileOpen={isMobileOpen}
        setIsMobileOpen={setIsMobileOpen}
      />

      <div className="flex-1 flex flex-col h-screen overflow-hidden w-full">
        
        {/* Mobile Header */}
        <header className="lg:hidden h-16 border-b border-white/10 flex items-center px-4 justify-between shrink-0 z-20 bg-[#0B0C15] text-white">
          <button 
            onClick={() => setIsMobileOpen(true)}
            className="p-2 rounded-md transition-colors hover:bg-white/10 text-white"
          >
            <Menu size={24} />
          </button>
          <span className="font-semibold">ScholarSphere Nexus</span>
          <div className="w-8" />
        </header>

        <main className="flex-1 overflow-hidden relative">
          {renderContent()}
        </main>

        <PaperXRayModal 
            isOpen={isXRayOpen} 
            onClose={() => setIsXRayOpen(false)} 
            content={xRayContent}
            contextString={xRayContext}
            isLoading={xRayLoading}
            fileName={xRaySelectedFile || 'Hızlı Analiz'}
        />
      </div>
    </div>
  );
};

export default App;
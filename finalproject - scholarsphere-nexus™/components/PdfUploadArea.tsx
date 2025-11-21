
import React, { useRef, useState } from 'react';
import { 
  UploadCloud, 
  Trash2, 
  Loader2, 
  Database, 
  Search, 
  Library, 
  FileText, 
  BookOpen,
  Plus,
  BrainCircuit,
  Check
} from 'lucide-react';
import { PdfFile, RagChunk } from '../types';
import * as pdfjsLib from 'pdfjs-dist';
import { getEmbeddings } from '../services/geminiService';

// Initialize PDF worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@4.8.69/build/pdf.worker.min.mjs`;

interface PdfUploadAreaProps {
  files: PdfFile[];
  onAddFiles: (files: PdfFile[]) => void;
  onUpdateFile: (file: PdfFile) => void;
  onDeleteFile: (id: string) => void;
}

// Type for tracking local progress state
interface ProcessStats {
  current: number;
  total: number;
  phase: string; // 'Extracting', 'Vectorizing', etc.
}

const PdfUploadArea: React.FC<PdfUploadAreaProps> = ({ files, onAddFiles, onUpdateFile, onDeleteFile }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Local state for granular progress tracking (avoids DB spam)
  const [processingStats, setProcessingStats] = useState<Record<string, ProcessStats>>({});
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Filter files based on search
  const filteredFiles = files.filter(f => 
    f.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const activeCount = files.filter(f => f.isSelected).length;
  const totalPages = files.reduce((acc, f) => acc + (f.pageCount || 0), 0);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
    }
  };

  // --- CORE LOGIC ---

  const getSmartTitle = (metaTitle: string | undefined | null, filename: string): string => {
    const cleanFilename = filename.replace(/\.pdf$/i, '').replace(/_/g, ' ').trim();
    if (!metaTitle || typeof metaTitle !== 'string') return cleanFilename;

    const trimmedMeta = metaTitle.trim();
    const lowerMeta = trimmedMeta.toLowerCase();

    if (
        trimmedMeta.length < 3 || 
        lowerMeta === 'untitled' || 
        lowerMeta.startsWith('microsoft word') || 
        lowerMeta.startsWith('presentation') ||
        lowerMeta.includes('.qxd') || 
        lowerMeta === cleanFilename.toLowerCase()
    ) {
        return cleanFilename;
    }
    return trimmedMeta;
  };

  const extractVisualTitle = async (pdf: any): Promise<string | null> => {
    try {
        const page = await pdf.getPage(1);
        const textContent = await page.getTextContent();
        
        let maxArea = 0;
        let likelyTitle = "";

        textContent.items.forEach((item: any) => {
            if (item.str.trim().length > 3) {
                const fontSize = Math.abs(item.transform[0]);
                if (fontSize > maxArea) {
                    maxArea = fontSize;
                    likelyTitle = item.str;
                } else if (Math.abs(fontSize - maxArea) < 0.1) {
                    likelyTitle += " " + item.str;
                }
            }
        });

        return likelyTitle.trim().length > 3 ? likelyTitle.trim() : null;
    } catch (e) {
        return null;
    }
  };

  const createChunksFromPage = (text: string, pageNum: number, fileId: string): RagChunk[] => {
    const cleanText = text.replace(/\s+/g, ' ').trim();
    if (cleanText.length === 0) return [];

    if (cleanText.length < 800) {
        return [{
            id: `${fileId}-p${pageNum}-c1`,
            fileId,
            pageNumber: pageNum,
            content: cleanText,
            tokenCountEstimate: Math.ceil(cleanText.length / 4)
        }];
    }

    const sentences = cleanText.match(/[^.!?]+[.!?]+["']?|.+$/g) || [cleanText];
    const chunks: RagChunk[] = [];
    let currentChunk = "";
    let chunkIndex = 1;

    sentences.forEach((sentence) => {
        if ((currentChunk.length + sentence.length) > 800) {
            chunks.push({
                id: `${fileId}-p${pageNum}-c${chunkIndex}`,
                fileId,
                pageNumber: pageNum,
                content: currentChunk.trim(),
                tokenCountEstimate: Math.ceil(currentChunk.length / 4)
            });
            currentChunk = sentence;
            chunkIndex++;
        } else {
            currentChunk += " " + sentence;
        }
    });

    if (currentChunk.trim().length > 0) {
        chunks.push({
            id: `${fileId}-p${pageNum}-c${chunkIndex}`,
            fileId,
            pageNumber: pageNum,
            content: currentChunk.trim(),
            tokenCountEstimate: Math.ceil(currentChunk.length / 4)
        });
    }
    return chunks;
  };

  const extractPdfData = async (file: File, fileId: string): Promise<{ chunks: RagChunk[]; title: string; pageCount: number }> => {
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    
    let metaTitle = null;
    try {
        const metadata = await pdf.getMetadata().catch(() => ({ info: null }));
        metaTitle = metadata?.info?.Title;
    } catch (e) { /* ignore */ }

    const visualTitle = await extractVisualTitle(pdf);
    let finalTitle = file.name.replace('.pdf', '');
    
    if (visualTitle) {
        finalTitle = visualTitle;
    } else {
        finalTitle = getSmartTitle(metaTitle, file.name);
    }

    const allChunks: RagChunk[] = [];
    
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map((item: any) => item.str).join(' ');
      const pageChunks = createChunksFromPage(pageText, i, fileId);
      allChunks.push(...pageChunks);
    }

    return {
      chunks: allChunks,
      title: finalTitle,
      pageCount: pdf.numPages
    };
  };

  /**
   * --- STRICT SEQUENTIAL VECTORIZATION QUEUE ---
   * Processes chunks one by one to prevent API Rate Limits and Browser Freezes.
   */
  const vectorizeChunksQueue = async (
      chunks: RagChunk[], 
      onProgress: (completed: number, total: number) => void
  ): Promise<RagChunk[]> => {
      // Small delay to allow UI to render updates
      const DELAY_MS = 20; 
      
      const vectorizedChunks: RagChunk[] = [];
      const total = chunks.length;

      for (let i = 0; i < total; i++) {
          const chunk = chunks[i];
          
          try {
              const embedding = await getEmbeddings(chunk.content);
              vectorizedChunks.push({ ...chunk, embedding });
          } catch (e) {
              // On error, push without embedding to save the text at least
              vectorizedChunks.push(chunk);
          }

          onProgress(i + 1, total);

          // Yield to main thread
          await new Promise(resolve => setTimeout(resolve, DELAY_MS));
      }

      return vectorizedChunks;
  };

  const processFiles = async (fileList: FileList) => {
    const incomingFiles = Array.from(fileList).filter(file => file.type === 'application/pdf');
    if (incomingFiles.length === 0) {
       alert("Sadece PDF dosyaları desteklenmektedir.");
       return;
    }

    // Initial state: Processing = true
    const newFileEntries: PdfFile[] = incomingFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      title: file.name.replace('.pdf', ''), 
      size: file.size,
      uploadDate: new Date(),
      isSelected: true,
      isProcessing: true,
      chunks: []
    }));

    onAddFiles(newFileEntries);

    for (let i = 0; i < incomingFiles.length; i++) {
        const file = incomingFiles[i];
        const fileEntry = newFileEntries[i];
        const fileId = fileEntry.id;

        // Set initial status for this file
        setProcessingStats(prev => ({
            ...prev,
            [fileId]: { current: 0, total: 1, phase: 'Metin Analizi Yapılıyor...' }
        }));

        try {
            const extractionResult = await extractPdfData(file, fileId);
            
            // Update status before vectorization
            setProcessingStats(prev => ({
                ...prev,
                [fileId]: { current: 0, total: extractionResult.chunks.length, phase: 'Vektörler Hesaplanıyor...' }
            }));

            // Save metadata but keep processing = true. 
            // We DO NOT change title here, keeping UI clean.
            onUpdateFile({
                 ...fileEntry,
                 ...extractionResult,
                 isProcessing: true 
            });

            const chunksWithVectors = await vectorizeChunksQueue(
                extractionResult.chunks, 
                (completed, total) => {
                    // Update LOCAL state only - super fast, no DB thrashing
                    setProcessingStats(prev => ({
                        ...prev,
                        [fileId]: { current: completed, total: total, phase: 'Vektörleniyor...' }
                    }));
                }
            );

            // Final Save with completed vectors
            onUpdateFile({
                ...fileEntry,
                ...extractionResult,
                chunks: chunksWithVectors,
                text: chunksWithVectors.map(c => c.content).join('\n\n'),
                title: extractionResult.title,
                isProcessing: false
            });

            // Clear local stats
            setProcessingStats(prev => {
                const next = { ...prev };
                delete next[fileId];
                return next;
            });

        } catch (error) {
            console.error("Processing Critical Error:", error);
            onUpdateFile({
                ...fileEntry,
                title: `Hata: ${file.name} (İşlenemedi)`,
                isProcessing: false,
                isSelected: false
            });
            setProcessingStats(prev => {
                const next = { ...prev };
                delete next[fileId];
                return next;
            });
        }
    }
  };

  const toggleSelection = (file: PdfFile) => {
    onUpdateFile({ ...file, isSelected: !file.isSelected });
  };

  const deleteFileHandler = (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation(); 
    
    if (window.confirm("Bu belgeyi kütüphaneden ve veritabanından kalıcı olarak silmek istiyor musunuz?")) {
      onDeleteFile(id);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="h-full flex flex-col bg-[#0B0C15] overflow-hidden font-sans text-slate-200">
      
      {/* --- HEADER SECTION --- */}
      <div className="px-8 py-6 bg-[#0B0C15] border-b border-white/10 shadow-sm z-40 flex flex-col md:flex-row md:items-center justify-between gap-4 relative">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2 tracking-tight">
            <Library className="text-teal-500" size={28} />
            Dijital Kütüphane
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Araştırma materyallerinizi yönetin ve yapay zeka bağlamını düzenleyin.
          </p>
        </div>

        <div className="flex items-center gap-3">
           <div className="relative group">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-teal-400 transition-colors" size={18} />
              <input 
                type="text" 
                placeholder="Belgelerde ara..." 
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2.5 bg-[#0F111A] border border-white/10 rounded-lg text-sm w-64 focus:bg-[#151720] focus:border-teal-500/50 focus:ring-2 focus:ring-teal-500/20 outline-none transition-all text-slate-200 placeholder:text-slate-600"
              />
           </div>

           <button 
             onClick={() => fileInputRef.current?.click()}
             className="flex items-center gap-2 px-4 py-2.5 bg-teal-600 hover:bg-teal-700 text-white rounded-lg text-sm font-medium transition-all shadow-sm active:scale-95"
           >
             <Plus size={18} />
             <span className="hidden md:inline">Belge Ekle</span>
           </button>
           <input type="file" ref={fileInputRef} onChange={handleFileSelect} className="hidden" accept="application/pdf" multiple />
        </div>
      </div>

      {/* --- STATS DASHBOARD --- */}
      <div className="px-8 py-4 grid grid-cols-1 md:grid-cols-3 gap-4 border-b border-white/10 bg-[#0F111A]/50 backdrop-blur-sm z-30 relative">
          <div className="bg-[#0F111A] p-3 rounded-lg border border-white/10 shadow-sm flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 text-blue-400 rounded-md">
                  <BookOpen size={20} />
              </div>
              <div>
                  <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">Toplam Belge</p>
                  <p className="text-lg font-bold text-slate-200">{files.length}</p>
              </div>
          </div>
          <div className="bg-[#0F111A] p-3 rounded-lg border border-white/10 shadow-sm flex items-center gap-3">
              <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-md">
                  <Database size={20} />
              </div>
              <div>
                  <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">Aktif Bellek</p>
                  <p className="text-lg font-bold text-slate-200">{activeCount} <span className="text-xs text-slate-500 font-normal">/ {files.length} seçili</span></p>
              </div>
          </div>
          <div className="bg-[#0F111A] p-3 rounded-lg border border-white/10 shadow-sm flex items-center gap-3">
              <div className="p-2 bg-purple-500/10 text-purple-400 rounded-md">
                  <FileText size={20} />
              </div>
              <div>
                  <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">Toplam Sayfa</p>
                  <p className="text-lg font-bold text-slate-200">{totalPages}</p>
              </div>
          </div>
      </div>

      {/* --- MAIN CONTENT AREA --- */}
      <div 
        className="flex-1 p-8 overflow-y-auto relative z-0"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isDragging && (
            <div className="absolute inset-4 z-50 bg-[#0B0C15]/90 border-2 border-dashed border-teal-500 rounded-2xl flex flex-col items-center justify-center backdrop-blur-sm">
                <UploadCloud size={64} className="text-teal-500 mb-4" />
                <h3 className="text-2xl font-bold text-teal-200">Belgeleri Buraya Bırakın</h3>
            </div>
        )}

        {files.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto">
                <div className="w-24 h-24 bg-[#0F111A] rounded-full flex items-center justify-center mb-6 shadow-inner border border-white/5">
                    <Library size={48} className="text-slate-600" />
                </div>
                <h2 className="text-xl font-bold text-slate-300">Kütüphaneniz Boş</h2>
                <p className="text-slate-500 mt-2 mb-8">
                    Analiz etmek istediğiniz makaleleri, kitap bölümlerini veya raporları sürükleyip bırakın.
                </p>
                <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="px-6 py-3 bg-teal-700 hover:bg-teal-600 text-white rounded-xl font-medium shadow-lg shadow-teal-900/20 transition-all hover:scale-105"
                >
                    İlk Belgeyi Yükle
                </button>
            </div>
        ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-20">
                {filteredFiles.map((file) => (
                    <div 
                        key={file.id}
                        className={`group relative bg-[#0F111A] rounded-xl border transition-all duration-300 flex flex-col overflow-hidden
                            ${file.isSelected 
                                ? 'border-teal-500/40 ring-1 ring-teal-500/20 shadow-md shadow-teal-900/10' 
                                : 'border-white/5 shadow-sm hover:shadow-md hover:border-white/10'}
                        `}
                    >
                        {/* PROCESSING OVERLAY */}
                        {file.isProcessing && (
                            <div className="absolute inset-0 bg-[#0B0C15]/95 backdrop-blur-sm z-20 flex flex-col items-center justify-center px-6">
                                {processingStats[file.id] ? (
                                    <div className="w-full max-w-[200px] flex flex-col items-center">
                                        <Loader2 size={28} className="animate-spin text-teal-500 mb-3" />
                                        
                                        <div className="w-full flex justify-between text-xs font-bold text-slate-300 mb-1.5">
                                            <span>{processingStats[file.id].phase}</span>
                                            <span>{Math.round((processingStats[file.id].current / processingStats[file.id].total) * 100)}%</span>
                                        </div>
                                        
                                        <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden border border-white/10">
                                            <div 
                                                className="bg-teal-500 h-full transition-all duration-200 ease-out"
                                                style={{ width: `${(processingStats[file.id].current / processingStats[file.id].total) * 100}%` }}
                                            />
                                        </div>
                                        
                                        <div className="mt-2 text-[10px] text-slate-400 font-medium bg-[#0F111A] px-2 py-1 rounded-md border border-white/10">
                                            Parçacık: {processingStats[file.id].current} / {processingStats[file.id].total}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center">
                                        <Loader2 size={32} className="animate-spin text-teal-500 mb-2" />
                                        <span className="text-xs font-medium text-teal-300 animate-pulse">Hazırlanıyor...</span>
                                    </div>
                                )}
                            </div>
                        )}

                        <div className="p-4 flex items-start justify-between gap-3 border-b border-white/5 bg-[#13151C] relative z-10">
                             <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 shadow-sm
                                 ${file.isSelected ? 'bg-gradient-to-br from-teal-500 to-teal-700 text-white' : 'bg-[#0B0C15] text-slate-500 border border-white/10'}
                             `}>
                                <FileText size={20} />
                             </div>
                             <div className="flex-1 min-w-0 pt-0.5">
                                 <h4 className="text-sm font-bold text-slate-200 truncate leading-tight" title={file.title}>
                                     {file.title.replace(/\(.*\)/, '')}
                                 </h4>
                                 <p className="text-slate-500 text-sm truncate mt-0.5">
                                     {file.name}
                                 </p>
                             </div>
                             
                             <button 
                                onClick={(e) => deleteFileHandler(e, file.id)}
                                className="z-50 p-2 text-slate-600 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all duration-200 cursor-pointer"
                                title="Belgeyi Kalıcı Olarak Sil"
                                type="button"
                             >
                                <Trash2 size={20} />
                             </button>
                        </div>

                        <div className="p-4 flex-1 flex flex-col justify-between gap-4 relative z-0">
                             <div className="flex flex-wrap gap-2">
                                 <div className="px-2 py-1 bg-[#0B0C15] rounded-md text-[10px] font-medium text-slate-400 flex items-center gap-1 border border-white/5">
                                     <Database size={10} />
                                     {formatBytes(file.size)}
                                 </div>
                                 {file.chunks && file.chunks.length > 0 && file.chunks[0].embedding && (
                                     <div className="px-2 py-1 bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 rounded-md text-[10px] font-medium flex items-center gap-1" title="Vektör Arama Hazır">
                                         <BrainCircuit size={10} />
                                         Vektör
                                     </div>
                                 )}
                             </div>

                             <div className="pt-3 mt-auto border-t border-white/5 flex items-center justify-between">
                                 <span className={`text-xs font-medium ${file.isSelected ? 'text-teal-400' : 'text-slate-500'}`}>
                                     {file.isSelected ? 'Analize Dahil' : 'Analiz Dışı'}
                                 </span>
                                 
                                 <button 
                                    onClick={() => toggleSelection(file)}
                                    className={`w-10 h-5 rounded-full relative transition-colors duration-300 focus:outline-none
                                        ${file.isSelected ? 'bg-teal-600' : 'bg-slate-700'}
                                    `}
                                 >
                                     <div className={`w-3.5 h-3.5 bg-white rounded-full absolute top-0.5 transition-transform duration-300 shadow-sm
                                         ${file.isSelected ? 'left-[calc(100%-1.125rem)]' : 'left-0.5'}
                                     `} />
                                 </button>
                             </div>
                        </div>
                    </div>
                ))}
            </div>
        )}
      </div>
    </div>
  );
};

export default PdfUploadArea;
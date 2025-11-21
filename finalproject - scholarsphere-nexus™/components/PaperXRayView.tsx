
import React from 'react';
import { ScanEye, UploadCloud, FileText, Calendar, Play } from 'lucide-react';
import { PdfFile } from '../types';

interface PaperXRayViewProps {
  files: PdfFile[];
  onAnalyze: (file: PdfFile) => void;
  onUploadRequest: () => void;
}

const PaperXRayView: React.FC<PaperXRayViewProps> = ({ files, onAnalyze, onUploadRequest }) => {
  return (
    <div className="h-full flex flex-col bg-[#0B0C15] font-sans overflow-y-auto text-slate-200">
      
      {/* Hero Header */}
      <div className="bg-gradient-to-b from-purple-900/20 to-[#0B0C15] text-white py-12 px-8 relative overflow-hidden shrink-0 border-b border-white/5">
        <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl -translate-y-12 translate-x-12 pointer-events-none"></div>
        <div className="relative z-10 max-w-4xl">
          <div className="flex items-center gap-3 mb-4 text-purple-400">
            <ScanEye size={32} />
            <span className="font-bold tracking-wider uppercase text-sm">PaperX-Ray™ Intelligence</span>
          </div>
          <h1 className="text-3xl md:text-4xl font-bold mb-4 leading-tight">
             Yapay Zeka Destekli <br/>
             <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">Akademik Hakem Simülasyonu</span>
          </h1>
          <p className="text-slate-400 max-w-2xl text-lg">
             Derin öğrenme modellerimiz makalelerinizi analiz eder; metodoloji, bulgular ve literatür katkısını "Peer Reviewer" gözüyle raporlar. Ayrıca gizli araştırma fırsatlarını tespit eder.
          </p>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 p-8">
        <div className="max-w-6xl mx-auto">
            
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-slate-200">Analiz İçin Belge Seçin</h2>
                <button 
                    onClick={onUploadRequest}
                    className="flex items-center gap-2 px-4 py-2 bg-[#0F111A] border border-white/10 hover:bg-white/5 text-slate-300 rounded-lg font-medium transition-colors shadow-sm"
                >
                    <UploadCloud size={18} />
                    Yeni Yükle
                </button>
            </div>

            {files.length === 0 ? (
                <div className="bg-[#0F111A] border-2 border-dashed border-white/10 rounded-2xl p-12 text-center flex flex-col items-center justify-center min-h-[400px]">
                    <div className="w-20 h-20 bg-purple-500/10 text-purple-400 rounded-full flex items-center justify-center mb-6">
                        <ScanEye size={40} />
                    </div>
                    <h3 className="text-2xl font-bold text-slate-200 mb-2">Analiz Edilecek Belge Yok</h3>
                    <p className="text-slate-500 max-w-md mb-8">
                        PaperX-Ray analizi için kütüphanenize bir makale yükleyin. 
                        Sistem, metodolojiyi ve araştırma boşluklarını otomatik olarak tarayacaktır.
                    </p>
                    <button 
                        onClick={onUploadRequest}
                        className="px-8 py-4 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-bold shadow-lg shadow-purple-900/20 transition-all hover:scale-105 flex items-center gap-2"
                    >
                        <UploadCloud size={20} />
                        Makale Yükle ve Başlat
                    </button>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {files.map((file) => (
                        <div key={file.id} className="bg-[#0F111A] rounded-xl border border-white/10 shadow-sm hover:shadow-lg hover:border-purple-500/30 transition-all duration-300 group overflow-hidden flex flex-col">
                            <div className="p-6 flex-1">
                                <div className="w-12 h-12 bg-[#0B0C15] border border-white/5 rounded-lg flex items-center justify-center text-slate-500 mb-4 group-hover:bg-purple-500/10 group-hover:text-purple-400 transition-colors">
                                    <FileText size={24} />
                                </div>
                                <h3 className="font-bold text-slate-200 mb-2 line-clamp-2 h-12" title={file.title}>
                                    {file.title}
                                </h3>
                                <div className="flex items-center gap-4 text-xs text-slate-500 mt-4">
                                    <span className="flex items-center gap-1">
                                        <Calendar size={12} />
                                        {file.uploadDate.toLocaleDateString()}
                                    </span>
                                    <span className="px-2 py-0.5 bg-[#0B0C15] rounded border border-white/5">
                                        {file.pageCount ? `${file.pageCount} Sayfa` : 'PDF'}
                                    </span>
                                </div>
                            </div>
                            
                            <div className="p-4 bg-[#0B0C15] border-t border-white/5">
                                <button 
                                    onClick={() => onAnalyze(file)}
                                    className="w-full flex items-center justify-center gap-2 py-3 bg-[#0F111A] border border-purple-500/30 text-purple-300 hover:bg-purple-600 hover:text-white rounded-lg font-bold transition-all shadow-sm"
                                >
                                    <ScanEye size={18} />
                                    Analizi Başlat
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
      </div>
    </div>
  );
};

export default PaperXRayView;
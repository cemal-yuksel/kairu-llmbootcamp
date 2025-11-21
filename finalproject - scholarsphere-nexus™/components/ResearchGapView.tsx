
import React, { useState } from 'react';
import { Radar, Lightbulb, FileText, ArrowRight, Sparkles, Loader2, AlertCircle, Download } from 'lucide-react';
import { PdfFile, ResearchGap } from '../types';
import { detectResearchGaps } from '../services/geminiService';
import { generateResearchReport } from '../services/pdfService';

interface ResearchGapViewProps {
  files: PdfFile[];
}

const ResearchGapView: React.FC<ResearchGapViewProps> = ({ files }) => {
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [gaps, setGaps] = useState<ResearchGap[]>([]);
  const [loading, setLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [scannedFileTitle, setScannedFileTitle] = useState("");

  const handleSelectFile = async (file: PdfFile) => {
      if (loading) return;
      
      setSelectedFileId(file.id);
      setLoading(true);
      setGaps([]);
      setScannedFileTitle(file.title);

      let context = "";
      if (file.chunks && file.chunks.length > 0) {
          context = file.chunks.map(c => c.content).join('\n');
      } else {
          context = file.text || "";
      }

      try {
          const results = await detectResearchGaps(context);
          setGaps(results);
      } catch (error) {
          console.error("Gap Detection Failed", error);
      } finally {
          setLoading(false);
      }
  };

  const handleExportPDF = async () => {
      if (gaps.length === 0 || isExporting) return;
      setIsExporting(true);
      
      try {
          const sections = gaps.map((gap, i) => ({
              title: `Fırsat ${i + 1}: ${gap.title}`,
              content: `Tanım:\n${gap.description}\n\nPotansiyel Etki:\n${gap.impact}`
          }));

          await generateResearchReport(
              "Research Gaps™ Analiz Raporu",
              scannedFileTitle || "Belge",
              sections
          );
      } catch (e) {
          console.error(e);
          alert("PDF oluşturulamadı.");
      } finally {
          setIsExporting(false);
      }
  };

  return (
    <div className="h-full flex bg-[#0B0C15] font-sans overflow-hidden text-slate-200">
        
        {/* LEFT PANEL: File List */}
        <div className="w-80 bg-[#0F111A] border-r border-white/10 flex flex-col shrink-0">
            <div className="p-6 border-b border-white/10">
                <div className="flex items-center gap-2 text-amber-500 mb-1">
                    <Radar size={24} />
                    <h2 className="font-bold text-lg">Research Gaps™</h2>
                </div>
                <p className="text-xs text-slate-400">
                    Hangi makaledeki eksikleri bulmak istiyorsunuz?
                </p>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {files.length === 0 && (
                    <div className="text-center p-4 text-slate-500 text-sm">
                        Listeniz boş. Lütfen kütüphaneye PDF yükleyin.
                    </div>
                )}
                {files.map(file => (
                    <button
                        key={file.id}
                        onClick={() => handleSelectFile(file)}
                        className={`w-full text-left p-3 rounded-xl border transition-all duration-200 flex items-start gap-3 group
                            ${selectedFileId === file.id 
                                ? 'bg-amber-500/10 border-amber-500/30 shadow-sm' 
                                : 'bg-[#0B0C15] border-white/5 hover:border-amber-500/20 hover:bg-white/5'}
                        `}
                    >
                        <div className={`mt-0.5 p-1.5 rounded-md shrink-0 transition-colors
                            ${selectedFileId === file.id ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-800 text-slate-500 group-hover:text-amber-500'}
                        `}>
                            <FileText size={16} />
                        </div>
                        <div className="min-w-0">
                            <h4 className={`text-sm font-medium truncate ${selectedFileId === file.id ? 'text-amber-400' : 'text-slate-300'}`}>
                                {file.title}
                            </h4>
                            <span className="text-[10px] text-slate-500">{file.pageCount || '?'} Sayfa</span>
                        </div>
                        {selectedFileId === file.id && (
                            <div className="ml-auto self-center">
                                {loading ? <Loader2 size={14} className="animate-spin text-amber-500" /> : <ArrowRight size={14} className="text-amber-500" />}
                            </div>
                        )}
                    </button>
                ))}
            </div>
        </div>

        {/* RIGHT PANEL: Results */}
        <div className="flex-1 overflow-y-auto p-8 relative">
            {/* Background Decor */}
            <div className="absolute top-0 right-0 w-96 h-96 bg-amber-500/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none"></div>

            {!selectedFileId ? (
                <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto opacity-60">
                    <div className="w-24 h-24 bg-amber-500/10 rounded-full flex items-center justify-center mb-6 text-amber-500">
                        <Lightbulb size={40} />
                    </div>
                    <h2 className="text-2xl font-bold text-slate-300 mb-2">Bir Kaynak Seçin</h2>
                    <p className="text-slate-500">
                        Yapay zeka; makalenin "Future Work" ve "Conclusion" bölümlerini tarayarak, henüz keşfedilmemiş araştırma fırsatlarını (Novelty) tespit edecektir.
                    </p>
                </div>
            ) : (
                <div className="max-w-4xl mx-auto">
                    <div className="flex items-center justify-between mb-8">
                        <div>
                             <h1 className="text-2xl font-bold text-slate-200">Tespit Edilen Fırsatlar</h1>
                             <p className="text-slate-400 text-sm mt-1">
                                Kaynak: <span className="font-semibold text-amber-500">{scannedFileTitle}</span>
                             </p>
                        </div>
                        <div className="flex gap-2">
                             {loading && (
                                <div className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 text-amber-400 rounded-lg border border-amber-500/20 text-xs font-bold animate-pulse">
                                    <Radar size={14} className="animate-spin" />
                                    TARAMA YAPILIYOR...
                                </div>
                             )}
                             {!loading && gaps.length > 0 && (
                                <button
                                    onClick={handleExportPDF}
                                    disabled={isExporting}
                                    className="flex items-center gap-2 px-4 py-2 bg-amber-700 hover:bg-amber-600 text-white rounded-lg shadow-sm transition-all text-xs font-medium disabled:opacity-70"
                                >
                                    {isExporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                                    {isExporting ? 'Hazırlanıyor...' : 'Raporu İndir'}
                                </button>
                             )}
                        </div>
                    </div>

                    {loading ? (
                        <div className="space-y-6">
                            {[1, 2, 3].map(i => (
                                <div key={i} className="h-40 rounded-xl bg-[#0F111A] animate-pulse border border-white/5"></div>
                            ))}
                        </div>
                    ) : gaps.length === 0 ? (
                        <div className="p-8 bg-[#0F111A] rounded-xl border border-white/10 text-center">
                            <AlertCircle className="mx-auto text-slate-500 mb-2" size={32} />
                            <p className="text-slate-400 font-medium">Bu makalede belirgin bir araştırma boşluğu tespit edilemedi veya içerik yeterli değil.</p>
                        </div>
                    ) : (
                        <div className="space-y-6 animate-in slide-in-from-bottom duration-500 fade-in">
                            {gaps.map((gap, index) => (
                                <div 
                                    key={index} 
                                    className="relative bg-[#0F111A] border border-amber-500/20 rounded-2xl p-6 hover:shadow-lg hover:border-amber-500/40 hover:bg-amber-900/5 transition-all duration-300 group"
                                >
                                    {/* Number Badge */}
                                    <div className="absolute -top-3 -left-3 w-8 h-8 bg-amber-600 text-white rounded-lg flex items-center justify-center font-bold shadow-md border-2 border-[#0B0C15]">
                                        {index + 1}
                                    </div>

                                    <div className="flex gap-4">
                                        <div className="mt-1 p-3 bg-[#0B0C15] rounded-xl shadow-sm text-amber-500 border border-white/10 shrink-0 h-fit">
                                            <Lightbulb size={24} className="group-hover:text-amber-400 transition-colors" />
                                        </div>
                                        <div className="flex-1">
                                            <h3 className="text-lg font-bold text-slate-200 mb-2 group-hover:text-amber-400 transition-colors">
                                                {gap.title}
                                            </h3>
                                            <p className="text-slate-400 leading-relaxed mb-4">
                                                {gap.description}
                                            </p>
                                            
                                            <div className="flex items-center gap-2">
                                                <Sparkles size={14} className="text-purple-500" />
                                                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Potansiyel Etki:</span>
                                            </div>
                                            <div className="mt-2 p-3 bg-[#0B0C15] rounded-lg border border-white/10 text-sm text-slate-300 italic">
                                                "{gap.impact}"
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    </div>
  );
};

export default ResearchGapView;

import React, { useState, useEffect } from 'react';
import { X, Copy, ScanEye, Check, Lightbulb, Radar, Loader2, Download } from 'lucide-react';
import { ResearchGap } from '../types';
import { detectResearchGaps } from '../services/geminiService';
import { generateResearchReport } from '../services/pdfService';

interface PaperXRayModalProps {
  isOpen: boolean;
  onClose: () => void;
  content: string;
  contextString: string; // Original text for the Gap Detector
  isLoading: boolean;
  fileName?: string;
}

const PaperXRayModal: React.FC<PaperXRayModalProps> = ({ isOpen, onClose, content, contextString, isLoading, fileName }) => {
  const [copied, setCopied] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  
  // Gap Detector State
  const [gaps, setGaps] = useState<ResearchGap[]>([]);
  const [isGapsLoading, setIsGapsLoading] = useState(false);
  const [hasScanned, setHasScanned] = useState(false);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (!isOpen) {
        setGaps([]);
        setHasScanned(false);
        setIsGapsLoading(false);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleExportPDF = async () => {
      if (isExporting) return;
      setIsExporting(true);
      try {
          // Parse markdown content into sections for the PDF
          // Format is usually "### Title \n Content"
          const sections = content.split('###').filter(s => s.trim()).map(s => {
              const [title, ...body] = s.trim().split('\n');
              return {
                  title: title.trim(),
                  content: body.join('\n').trim()
              };
          });

          // Add Gaps if available
          if (gaps.length > 0) {
              let gapsContent = "";
              gaps.forEach((gap, i) => {
                  gapsContent += `${i+1}. ${gap.title}\n${gap.description}\nEtki: ${gap.impact}\n\n`;
              });
              sections.push({
                  title: "Tespit Edilen Araştırma Fırsatları",
                  content: gapsContent
              });
          }

          await generateResearchReport(
              "PaperX-Ray™ Derin Analiz Raporu",
              fileName || "Bilinmeyen Dosya",
              sections
          );
      } catch (error) {
          console.error("PDF generation failed", error);
          alert("PDF oluşturulurken bir hata oluştu.");
      } finally {
          setIsExporting(false);
      }
  };

  const handleScanGaps = async () => {
      setIsGapsLoading(true);
      try {
          const results = await detectResearchGaps(contextString);
          setGaps(results);
          setHasScanned(true);
      } catch (error) {
          console.error("Gap scan failed", error);
      } finally {
          setIsGapsLoading(false);
      }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/80 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Modal Container - Widened for Split View */}
      <div className="relative w-full max-w-7xl max-h-[90vh] bg-[#0F111A] rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in duration-200 border border-white/10">
        
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between bg-[#0B0C15]/80 backdrop-blur shrink-0">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/10 text-purple-400 rounded-lg">
                <ScanEye size={24} />
            </div>
            <div>
                <h2 className="text-lg font-bold text-slate-200">PaperX-Ray™ Derin Analiz</h2>
                {fileName && <p className="text-xs text-slate-500">Kaynak: {fileName}</p>}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {!isLoading && (
                <>
                    <button 
                        onClick={handleCopy}
                        className="flex items-center gap-2 px-3 py-1.5 bg-[#0F111A] border border-white/10 hover:bg-white/5 text-slate-300 rounded-lg shadow-sm transition-all text-xs font-medium"
                    >
                        {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
                        {copied ? "Kopyalandı" : "Metni Kopyala"}
                    </button>
                    <button 
                        onClick={handleExportPDF}
                        disabled={isExporting}
                        className="flex items-center gap-2 px-3 py-1.5 bg-teal-700 hover:bg-teal-600 text-white rounded-lg shadow-sm transition-all text-xs font-medium disabled:opacity-70 disabled:cursor-wait"
                    >
                        {isExporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                        {isExporting ? "Hazırlanıyor..." : "PDF İndir"}
                    </button>
                </>
            )}
            <div className="w-px h-6 bg-white/10 mx-2"></div>
            <button 
                onClick={onClose}
                className="p-2 hover:bg-white/10 rounded-full transition-colors text-slate-400"
            >
                <X size={20} />
            </button>
          </div>
        </div>

        {/* Body - Split View */}
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
            
            {/* LEFT PANEL: Peer Review Report (70%) */}
            <div className="flex-1 lg:flex-[0.7] overflow-y-auto p-6 sm:p-8 bg-[#0F111A] border-r border-white/10 relative text-slate-300">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center h-full min-h-[300px] space-y-6">
                        <div className="relative">
                            <div className="w-16 h-16 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin"></div>
                            <div className="absolute inset-0 flex items-center justify-center">
                                <ScanEye size={24} className="text-purple-500 animate-pulse" />
                            </div>
                        </div>
                        <div className="text-center">
                            <h3 className="text-lg font-semibold text-slate-300">Makale İnceleniyor...</h3>
                            <p className="text-sm text-slate-500 max-w-xs mx-auto mt-2">
                                Akademik hakem modülü, metodoloji ve bulguları analiz ediyor. Bu işlem derinlemesine düşünme gerektirdiğinden biraz sürebilir.
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="prose prose-invert max-w-none prose-headings:text-purple-400 prose-headings:font-bold prose-h3:text-lg prose-h3:border-b prose-h3:border-white/10 prose-h3:pb-2 prose-h3:mt-6 prose-p:text-slate-300">
                        <div className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
                            {content.split('###').map((section, index) => {
                                if (index === 0) return <span key={index}>{section}</span>;
                                const [title, ...rest] = section.split('\n');
                                return (
                                    <div key={index} className="mb-6">
                                        <h3 className="text-lg font-bold text-purple-300 border-b border-purple-500/20 pb-2 mb-3 flex items-center gap-2">
                                            <span className="w-2 h-6 bg-purple-500 rounded-sm inline-block"></span>
                                            {title}
                                        </h3>
                                        <div className="text-slate-300 pl-4">{rest.join('\n')}</div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>

            {/* RIGHT PANEL: Research Gap Detector (30%) */}
            <div className="flex-1 lg:flex-[0.3] bg-[#0B0C15] overflow-y-auto p-6 border-t lg:border-t-0">
                <div className="flex items-center gap-2 mb-4">
                    <Radar className="text-amber-500" size={20} />
                    <h3 className="font-bold text-slate-200">Araştırma Fırsatları</h3>
                </div>

                {!hasScanned && !isGapsLoading && (
                    <div className="p-6 bg-[#0F111A] rounded-xl border border-dashed border-white/20 text-center">
                        <Lightbulb className="mx-auto text-slate-600 mb-3" size={32} />
                        <p className="text-sm text-slate-400 mb-4">
                            Makaledeki eksik yönleri ve gelecek çalışma önerilerini tespit etmek için tarayın.
                        </p>
                        <button 
                            onClick={handleScanGaps}
                            disabled={isLoading} // Disable if main report is still loading
                            className="w-full flex items-center justify-center gap-2 py-2.5 bg-amber-500/10 hover:bg-amber-500/20 text-amber-500 border border-amber-500/20 rounded-lg font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Radar size={18} />
                            Fırsatları Tara
                        </button>
                    </div>
                )}

                {isGapsLoading && (
                    <div className="flex flex-col items-center justify-center py-12">
                        <Loader2 size={32} className="animate-spin text-amber-500 mb-3" />
                        <p className="text-xs font-medium text-amber-600">Literatür boşlukları taranıyor...</p>
                    </div>
                )}

                {hasScanned && gaps.length > 0 && (
                    <div className="space-y-4 animate-in slide-in-from-right duration-500">
                        {gaps.map((gap, idx) => (
                            <div key={idx} className="bg-[#0F111A] border border-amber-500/20 rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow hover:border-amber-500/40">
                                <div className="flex items-start gap-3">
                                    <div className="p-1.5 bg-amber-500/10 rounded-full text-amber-500 shrink-0 mt-0.5">
                                        <Lightbulb size={16} />
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-bold text-slate-200 leading-tight mb-1">{gap.title}</h4>
                                        <p className="text-xs text-slate-400 mb-2 leading-relaxed">{gap.description}</p>
                                        <div className="text-[10px] font-medium text-amber-400 bg-amber-500/10 px-2 py-1 rounded inline-block">
                                            Etki: {gap.impact}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                        <button 
                            onClick={handleScanGaps}
                            className="w-full mt-4 py-2 text-xs text-slate-500 hover:text-slate-300 flex items-center justify-center gap-1"
                        >
                            <Radar size={14} /> Yeniden Tara
                        </button>
                    </div>
                )}
            </div>
        </div>
      </div>
    </div>
  );
};

export default PaperXRayModal;
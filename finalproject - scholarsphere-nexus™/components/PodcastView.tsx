
import React, { useState, useRef, useEffect } from 'react';
import { Headphones, Play, Pause, Download, FileText, Music2, Loader2 } from 'lucide-react';
import { PdfFile } from '../types';
import { generatePodcastScript, generatePodcastAudio } from '../services/geminiService';

interface PodcastViewProps {
  files: PdfFile[];
}

const PodcastView: React.FC<PodcastViewProps> = ({ files }) => {
  const [selectedFileIds, setSelectedFileIds] = useState<Set<string>>(new Set());
  const [isGenerating, setIsGenerating] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [script, setScript] = useState("");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const audioRef = useRef<HTMLAudioElement>(null);

  const toggleFileSelection = (id: string) => {
    const newSet = new Set(selectedFileIds);
    if (newSet.has(id)) newSet.delete(id);
    else newSet.add(id);
    setSelectedFileIds(newSet);
  };

  const handleGenerate = async () => {
    if (selectedFileIds.size === 0) return;
    
    setIsGenerating(true);
    setScript("");
    setAudioUrl(null);

    try {
        // 1. Context Building
        setStatusText("Senaryo yazılıyor (Gemini 3 Pro)...");
        const selectedFiles = files.filter(f => selectedFileIds.has(f.id));
        const context = selectedFiles.map(f => `Content from ${f.title}:\n${f.text?.substring(0, 20000)}`).join('\n\n');

        // 2. Script Generation
        const generatedScript = await generatePodcastScript(context);
        setScript(generatedScript);

        // 3. Audio Synthesis
        setStatusText("Seslendiriliyor (Gemini TTS - Jane & Joe)...");
        const audioBlob = await generatePodcastAudio(generatedScript);

        if (audioBlob) {
            const url = URL.createObjectURL(audioBlob);
            setAudioUrl(url);
        } else {
            setStatusText("Ses oluşturulamadı.");
        }

    } catch (e) {
        console.error(e);
        setStatusText("Hata oluştu.");
    } finally {
        setIsGenerating(false);
    }
  };

  const togglePlay = () => {
    if (audioRef.current) {
        if (isPlaying) audioRef.current.pause();
        else audioRef.current.play();
        setIsPlaying(!isPlaying);
    }
  };

  useEffect(() => {
    if (audioRef.current) {
        const ref = audioRef.current;
        const onEnded = () => setIsPlaying(false);
        const onPause = () => setIsPlaying(false);
        const onPlay = () => setIsPlaying(true);
        
        ref.addEventListener('ended', onEnded);
        ref.addEventListener('pause', onPause);
        ref.addEventListener('play', onPlay);
        
        return () => {
            ref.removeEventListener('ended', onEnded);
            ref.removeEventListener('pause', onPause);
            ref.removeEventListener('play', onPlay);
        };
    }
  }, [audioUrl]);

  return (
    <div className="h-full flex bg-[#0B0C15] font-sans overflow-hidden">
        
        {/* LEFT PANEL: Configuration */}
        <div className="w-80 bg-[#0F111A] border-r border-white/10 flex flex-col shrink-0 h-full">
            
            {/* Header */}
            <div className="p-6 border-b border-white/10 bg-[#0B0C15] text-white shrink-0">
                <div className="flex items-center gap-2 mb-1 text-rose-400">
                    <Headphones size={24} />
                    <h2 className="font-bold text-lg">Audio Overview™</h2>
                </div>
                <p className="text-xs text-slate-400">
                    Makalelerinizi dinlenebilir, dinamik bir podcast bölümüne dönüştürün.
                </p>
            </div>

            {/* File List Area */}
            <div className="flex-1 overflow-y-auto p-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 sticky top-0 bg-[#0F111A] py-2 z-10">Kaynak Seçimi</h3>
                <div className="space-y-2">
                    {files.length === 0 ? (
                        <div className="text-slate-500 text-xs text-center py-4 border border-dashed border-white/10 rounded-lg">
                            Kütüphane boş. PDF yükleyin.
                        </div>
                    ) : files.map(f => (
                        <div 
                            key={f.id} 
                            onClick={() => toggleFileSelection(f.id)}
                            className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all
                                ${selectedFileIds.has(f.id) 
                                    ? 'bg-rose-500/10 border-rose-500/30 shadow-sm' 
                                    : 'bg-[#151720] border-white/5 hover:bg-[#1A1D26]'}
                            `}
                        >
                            <div className={`w-5 h-5 rounded border flex items-center justify-center shrink-0 ${selectedFileIds.has(f.id) ? 'bg-rose-500 border-rose-500 text-white' : 'bg-[#0B0C15] border-slate-700'}`}>
                                {selectedFileIds.has(f.id) && <FileText size={12} />}
                            </div>
                            <span className="text-sm font-medium text-slate-300 truncate select-none">{f.title}</span>
                        </div>
                    ))}
                </div>
            </div>
            
            {/* Footer Action Area */}
            <div className="p-4 border-t border-white/10 bg-[#0B0C15] shrink-0 z-20 shadow-[0_-5px_20px_rgba(0,0,0,0.2)]">
                <button
                    onClick={handleGenerate}
                    disabled={isGenerating || selectedFileIds.size === 0}
                    className="w-full py-3 bg-rose-600 hover:bg-rose-700 text-white rounded-xl font-bold shadow-lg shadow-rose-900/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 active:scale-95"
                >
                    {isGenerating ? <Loader2 size={18} className="animate-spin" /> : <Music2 size={18} />}
                    {isGenerating ? 'Üretiliyor...' : 'Bölümü Oluştur'}
                </button>
                
                {isGenerating && (
                    <p className="text-center text-xs text-slate-400 mt-2 animate-pulse">{statusText}</p>
                )}
            </div>
        </div>

        {/* RIGHT PANEL: Player & Script */}
        <div className="flex-1 overflow-hidden flex flex-col relative bg-[#0B0C15]">
            
            {/* Visualizer Background (CSS Art) */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-rose-500/10 rounded-full blur-[100px] transition-all duration-700 ${isPlaying ? 'scale-125 opacity-50' : 'scale-100 opacity-30'}`}></div>
                <div className={`absolute top-1/3 left-1/3 w-[300px] h-[300px] bg-blue-500/10 rounded-full blur-[80px] transition-all duration-1000 ${isPlaying ? 'scale-110' : 'scale-100'}`}></div>
            </div>

            <div className="relative z-10 flex-1 flex flex-col items-center justify-center p-12 text-center">
                {!audioUrl ? (
                    <div className="text-slate-500 flex flex-col items-center">
                        <div className="w-24 h-24 rounded-full bg-white/5 border border-white/10 flex items-center justify-center mb-6">
                            <Headphones size={40} className="opacity-50" />
                        </div>
                        <h2 className="text-2xl font-bold text-white mb-2">Henüz Bir Kayıt Yok</h2>
                        <p className="max-w-md">Sol menüden makale seçin ve yapay zeka sunucularımızın sizin için özel bir tartışma programı hazırlamasını bekleyin.</p>
                    </div>
                ) : (
                    <div className="w-full max-w-2xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl animate-in fade-in slide-in-from-bottom">
                        
                        {/* Cover Art Placeholder */}
                        <div className="w-32 h-32 mx-auto bg-gradient-to-br from-rose-500 to-orange-600 rounded-2xl shadow-lg mb-8 flex items-center justify-center border border-white/10">
                            <Music2 size={48} className="text-white" />
                        </div>

                        <h2 className="text-2xl font-bold text-white mb-2">Academic Deep Dive</h2>
                        <p className="text-rose-200 text-sm font-medium mb-8">Host: Jane • Expert: Joe</p>

                        {/* Custom Audio Controls */}
                        <div className="flex items-center justify-center gap-6 mb-8">
                            <button className="text-slate-400 hover:text-white transition-colors">
                                <span className="text-xs font-bold">-10s</span>
                            </button>
                            
                            <button 
                                onClick={togglePlay}
                                className="w-16 h-16 bg-white text-rose-600 rounded-full flex items-center justify-center shadow-lg hover:scale-105 transition-transform"
                            >
                                {isPlaying ? <Pause size={28} fill="currentColor" /> : <Play size={28} fill="currentColor" className="ml-1" />}
                            </button>
                            
                            <button className="text-slate-400 hover:text-white transition-colors">
                                <span className="text-xs font-bold">+10s</span>
                            </button>
                        </div>
                        
                        <audio ref={audioRef} src={audioUrl} className="hidden" />

                        {/* Script Accordion */}
                        <div className="text-left bg-black/40 rounded-xl border border-white/5 p-4 max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-white/20">
                            <h4 className="text-xs font-bold text-slate-400 uppercase mb-2 sticky top-0">Transkript</h4>
                            <div className="whitespace-pre-wrap text-slate-300 text-sm font-mono leading-relaxed">
                                {script}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    </div>
  );
};

export default PodcastView;


import React, { useState } from 'react';
import { 
  MessageSquare, 
  Library, 
  ScanEye, 
  Radar, 
  Users, 
  ArrowUpRight, 
  Sparkles,
  Settings,
  Headphones,
  Cpu,
  Briefcase, // Updated
  Command
} from 'lucide-react';
import { View } from '../types';

interface LandingViewProps {
  setCurrentView: (view: View) => void;
}

const LandingView: React.FC<LandingViewProps> = ({ setCurrentView }) => {
  
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  return (
    <div className="h-full w-full bg-[#0B0C15] font-sans relative overflow-y-auto selection:bg-scholar-teal-400/30 selection:text-scholar-teal-200">
      
      {/* Background Gradients */}
      <div className="fixed inset-0 pointer-events-none">
         <div className="absolute top-[-20%] left-[-10%] w-[800px] h-[800px] bg-purple-900/20 rounded-full blur-[120px] opacity-50 mix-blend-screen animate-pulse"></div>
         <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-teal-900/20 rounded-full blur-[100px] opacity-40 mix-blend-screen"></div>
         <div className="absolute top-[20%] right-[20%] w-[400px] h-[400px] bg-indigo-900/10 rounded-full blur-[80px] opacity-30"></div>
      </div>

      {/* Content Container */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 py-12 lg:py-16 flex flex-col min-h-screen justify-center">
        
        {/* Minimal Header */}
        <div className="flex items-center justify-between mb-12 px-2 animate-in slide-in-from-top-8 duration-700">
            <div className="flex items-center gap-3 group cursor-default">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-400 to-emerald-600 flex items-center justify-center shadow-lg shadow-teal-900/30 group-hover:rotate-12 transition-transform duration-500">
                    <Sparkles className="text-white" size={20} />
                </div>
                <div>
                    <h1 className="text-2xl font-bold text-white tracking-tight">ScholarSphere <span className="text-teal-400">Nexus</span></h1>
                    <p className="text-[10px] font-medium text-slate-500 uppercase tracking-[0.2em]">Academic OS v2.0</p>
                </div>
            </div>

            <div className="hidden md:flex items-center gap-4 text-slate-400 text-xs font-medium bg-white/5 px-4 py-2 rounded-full border border-white/5 backdrop-blur-md">
                <span className="flex items-center gap-1.5 hover:text-teal-300 transition-colors cursor-help">
                    <div className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse"></div>
                    Gemini 3.0 Pro
                </span>
                <span className="w-px h-3 bg-white/10"></span>
                <span className="flex items-center gap-1.5 hover:text-indigo-300 transition-colors cursor-help">
                    <Cpu size={12} />
                    Neural Engine
                </span>
            </div>
        </div>

        {/* BENTO GRID - 4 COLUMNS PACKED */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 auto-rows-[minmax(180px,auto)] gap-5">
            
            {/* 1. NEXUS GPT (2x2) */}
            <button 
                onClick={() => setCurrentView(View.CHAT)}
                onMouseEnter={() => setHoveredCard(View.CHAT)}
                onMouseLeave={() => setHoveredCard(null)}
                className="group relative col-span-1 md:col-span-2 lg:col-span-2 row-span-2 bg-gradient-to-br from-slate-900 to-slate-900/80 rounded-[2rem] border border-white/10 p-8 overflow-hidden text-left hover:border-teal-500/50 hover:shadow-[0_0_40px_-10px_rgba(45,212,191,0.3)] transition-all duration-500 flex flex-col justify-between"
            >
                <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-[0.05] mix-blend-overlay"></div>
                
                {/* Typing Animation Visual */}
                <div className="relative z-10 space-y-4 mb-8">
                    <div className="flex items-center gap-3 opacity-60">
                         <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center"><Users size={14} className="text-slate-300"/></div>
                         <div className="bg-slate-800 px-4 py-2 rounded-2xl rounded-tl-none text-xs text-slate-300 border border-white/5">
                            Kuantum dolanıklık makalesini özetle.
                         </div>
                    </div>
                    <div className="flex items-center gap-3">
                         <div className="w-8 h-8 rounded-full bg-teal-600 flex items-center justify-center"><Sparkles size={14} className="text-white"/></div>
                         <div className="bg-gradient-to-r from-teal-900/50 to-slate-800 px-4 py-3 rounded-2xl rounded-tl-none text-sm text-slate-200 border border-teal-500/20 shadow-lg backdrop-blur-sm max-w-[80%]">
                            <div className="flex space-x-1">
                                <span className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                <span className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                            </div>
                         </div>
                    </div>
                </div>

                <div className="relative z-10">
                    <div className="w-12 h-12 bg-teal-500/20 rounded-2xl flex items-center justify-center mb-4 border border-teal-500/30 group-hover:bg-teal-500 group-hover:text-white text-teal-400 transition-colors duration-300">
                        <MessageSquare size={24} />
                    </div>
                    <h2 className="text-3xl font-bold text-white mb-2">NexusGPT™</h2>
                    <p className="text-slate-400 text-sm font-medium max-w-md group-hover:text-slate-300 transition-colors">
                        RAG teknolojisi ile kütüphanenizle konuşun. Halüsinasyon yok, sadece kanıtlı akademik cevaplar.
                    </p>
                </div>

                {/* Hover Glow */}
                <div className="absolute -bottom-32 -right-32 w-64 h-64 bg-teal-500/30 rounded-full blur-[80px] group-hover:opacity-100 opacity-0 transition-opacity duration-700 pointer-events-none"></div>
            </button>

            {/* 2. CO-AUTHOR STUDIO (2x1 - Wide) */}
            <button 
                onClick={() => setCurrentView(View.CO_AUTHOR)}
                onMouseEnter={() => setHoveredCard(View.CO_AUTHOR)}
                onMouseLeave={() => setHoveredCard(null)}
                className="group relative col-span-1 md:col-span-2 lg:col-span-2 row-span-1 bg-[#0F111A] rounded-[2rem] border border-white/10 p-8 overflow-hidden text-left hover:border-indigo-500/50 transition-all duration-500"
            >
                <div className="absolute top-8 right-8 text-indigo-900/20 group-hover:text-indigo-500/20 transition-colors duration-500">
                    <Command size={120} className={`transform transition-transform duration-[2000ms] ${hoveredCard === View.CO_AUTHOR ? 'rotate-12 scale-110' : 'rotate-0'}`} />
                </div>

                <div className="relative z-10 h-full flex flex-col justify-between">
                    <div className="flex justify-between items-start">
                        <div className="w-10 h-10 bg-indigo-500/20 rounded-xl flex items-center justify-center border border-indigo-500/30 text-indigo-400 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                            <Users size={20} />
                        </div>
                        <ArrowUpRight size={20} className="text-slate-600 group-hover:text-white group-hover:translate-x-1 group-hover:-translate-y-1 transition-all" />
                    </div>
                    
                    <div className="mt-auto">
                        <h3 className="text-xl font-bold text-white mb-1">Co-Author Studio™</h3>
                        <p className="text-slate-500 text-xs group-hover:text-slate-300 transition-colors">
                            APA 7 formatında, sizinle birlikte makale yazan asistan.
                        </p>
                    </div>
                </div>
            </button>

            {/* 3. PAPER X-RAY (1x1) */}
            <button 
                onClick={() => setCurrentView(View.PAPER_XRAY)}
                onMouseEnter={() => setHoveredCard(View.PAPER_XRAY)}
                onMouseLeave={() => setHoveredCard(null)}
                className="group relative col-span-1 row-span-1 bg-[#0F111A] rounded-[2rem] border border-white/10 p-6 overflow-hidden text-left hover:border-purple-500/50 transition-all duration-500"
            >
                <div className="absolute inset-0 flex flex-col items-center justify-center opacity-10 group-hover:opacity-30 transition-opacity">
                    <div className="w-32 h-40 border-2 border-slate-500 rounded-lg relative overflow-hidden bg-slate-800/50">
                        <div className="w-full h-1 bg-purple-400 shadow-[0_0_15px_rgba(192,132,252,0.8)] absolute top-0 group-hover:animate-scan"></div>
                    </div>
                </div>

                <div className="relative z-10 flex flex-col h-full justify-end">
                    <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center mb-3 border border-purple-500/30 text-purple-400 group-hover:bg-purple-600 group-hover:text-white transition-colors">
                        <ScanEye size={20} />
                    </div>
                    <h3 className="text-lg font-bold text-white leading-tight">Paper X-Ray™</h3>
                    <p className="text-slate-500 text-[10px] mt-1">AI Hakem Simülasyonu</p>
                </div>
            </button>

            {/* 4. PODCAST (1x1) */}
            <button 
                onClick={() => setCurrentView(View.PODCAST)}
                onMouseEnter={() => setHoveredCard(View.PODCAST)}
                onMouseLeave={() => setHoveredCard(null)}
                className="group relative col-span-1 row-span-1 bg-[#0F111A] rounded-[2rem] border border-white/10 p-6 overflow-hidden text-left hover:border-rose-500/50 transition-all duration-500"
            >
                 <div className="absolute inset-0 flex items-center justify-center gap-1 opacity-20 group-hover:opacity-60 transition-opacity">
                     <div className="w-3 bg-rose-500 rounded-full h-8 group-hover:animate-equalizer" style={{ animationDelay: '0ms' }}></div>
                     <div className="w-3 bg-rose-500 rounded-full h-12 group-hover:animate-equalizer" style={{ animationDelay: '100ms' }}></div>
                     <div className="w-3 bg-rose-500 rounded-full h-6 group-hover:animate-equalizer" style={{ animationDelay: '200ms' }}></div>
                 </div>

                <div className="relative z-10 flex flex-col h-full justify-end">
                    <div className="w-10 h-10 bg-rose-500/20 rounded-xl flex items-center justify-center mb-3 border border-rose-500/30 text-rose-400 group-hover:bg-rose-600 group-hover:text-white transition-colors">
                        <Headphones size={20} />
                    </div>
                    <h3 className="text-lg font-bold text-white leading-tight">Audio Overview™</h3>
                    <p className="text-slate-500 text-[10px] mt-1">Makaleleri Dinle</p>
                </div>
            </button>

            {/* 5. RESEARCH GAPS (1x1) */}
            <button 
                onClick={() => setCurrentView(View.RESEARCH_GAPS)}
                onMouseEnter={() => setHoveredCard(View.RESEARCH_GAPS)}
                onMouseLeave={() => setHoveredCard(null)}
                className="group relative col-span-1 row-span-1 bg-[#0F111A] rounded-[2rem] border border-white/10 p-6 overflow-hidden text-left hover:border-amber-500/50 transition-all duration-500"
            >
                 <div className="absolute -right-6 -top-6 w-32 h-32 border border-amber-500/20 rounded-full flex items-center justify-center opacity-30 group-hover:opacity-60 transition-opacity">
                     <div className="w-1/2 h-1/2 bg-gradient-to-tr from-transparent to-amber-500/40 rounded-full absolute top-0 right-0 origin-bottom-left group-hover:animate-radar"></div>
                 </div>

                <div className="relative z-10 flex flex-col h-full justify-end">
                    <div className="w-10 h-10 bg-amber-500/20 rounded-xl flex items-center justify-center mb-3 border border-amber-500/30 text-amber-400 group-hover:bg-amber-600 group-hover:text-white transition-colors">
                        <Radar size={20} />
                    </div>
                    <h3 className="text-lg font-bold text-white leading-tight">Research Gaps™</h3>
                    <p className="text-slate-500 text-[10px] mt-1">Literatür Boşlukları</p>
                </div>
            </button>

            {/* 6. ACADEMIC TOOLS (1x1) - REPLACED KNOWLEDGE GRAPH */}
            <button 
                onClick={() => setCurrentView(View.ACADEMIC_TOOLS)}
                onMouseEnter={() => setHoveredCard(View.ACADEMIC_TOOLS)}
                onMouseLeave={() => setHoveredCard(null)}
                className="group relative col-span-1 row-span-1 bg-[#0F111A] rounded-[2rem] border border-white/10 p-6 overflow-hidden text-left hover:border-blue-500/50 transition-all duration-500"
            >
                 <div className="absolute top-4 right-4 text-blue-900/30 group-hover:text-blue-500/30 transition-colors">
                     <Briefcase size={64} className="group-hover:scale-110 transition-transform duration-700" />
                 </div>

                <div className="relative z-10 flex flex-col h-full justify-end">
                    <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center mb-3 border border-blue-500/30 text-blue-400 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                        <Briefcase size={20} />
                    </div>
                    <h3 className="text-lg font-bold text-white leading-tight">Academic Tools™</h3>
                    <p className="text-slate-500 text-[10px] mt-1">6 Yardımcı Araç</p>
                </div>
            </button>

            {/* 7. LIBRARY (1x1) */}
            <button 
                onClick={() => setCurrentView(View.LIBRARY)}
                onMouseEnter={() => setHoveredCard(View.LIBRARY)}
                onMouseLeave={() => setHoveredCard(null)}
                className="group relative col-span-1 row-span-1 bg-[#0F111A] rounded-[2rem] border border-white/10 p-6 overflow-hidden text-left hover:border-emerald-500/50 transition-all duration-500"
            >
                <div className="absolute top-4 right-4 text-emerald-900/30 group-hover:text-emerald-500/30 transition-colors">
                     <Library size={64} className="group-hover:-translate-x-2 transition-transform duration-500" />
                </div>

                <div className="relative z-10 flex flex-col h-full justify-end">
                    <div className="w-10 h-10 bg-emerald-500/20 rounded-xl flex items-center justify-center mb-3 border border-emerald-500/30 text-emerald-400 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                        <Library size={20} />
                    </div>
                    <h3 className="text-lg font-bold text-white leading-tight">Kütüphane</h3>
                    <p className="text-slate-500 text-[10px] mt-1">Kaynak Yönetimi</p>
                </div>
            </button>

             {/* 8. SETTINGS (1x1) */}
             <button 
                onClick={() => setCurrentView(View.SETTINGS)}
                onMouseEnter={() => setHoveredCard(View.SETTINGS)}
                onMouseLeave={() => setHoveredCard(null)}
                className="group relative col-span-1 row-span-1 bg-[#0F111A] rounded-[2rem] border border-white/10 p-6 overflow-hidden text-left hover:border-slate-500/50 transition-all duration-500"
            >
                 <div className="absolute top-4 right-4 text-slate-800 group-hover:text-slate-600 transition-colors">
                     <Settings size={64} className="group-hover:rotate-90 transition-transform duration-700" />
                 </div>

                <div className="relative z-10 flex flex-col h-full justify-end">
                    <div className="w-10 h-10 bg-slate-800 rounded-xl flex items-center justify-center mb-3 border border-slate-700 text-slate-400 group-hover:bg-slate-700 group-hover:text-white transition-colors">
                        <Settings size={20} />
                    </div>
                    <h3 className="text-lg font-bold text-white leading-tight">Ayarlar</h3>
                    <p className="text-slate-500 text-[10px] mt-1">Sistem Tercihleri</p>
                </div>
            </button>
        </div>

        {/* Footer Info */}
        <div className="mt-12 pt-8 border-t border-white/5 text-center md:text-left flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex flex-col gap-1">
                <p className="text-xs text-slate-500">© 2025 ScholarSphere Nexus. All rights reserved.</p>
                <p className="text-xs text-slate-600 max-w-2xl leading-relaxed">
                    ScholarSphere Nexus was developed by <strong className="text-slate-400 font-bold">Cemal YÜKSEL</strong> as the final project for <strong className="text-slate-400 font-bold">Build with LLMs Bootcamp</strong>
                </p>
            </div>
            <div className="flex items-center gap-2 text-[10px] text-teal-800/60 bg-teal-900/10 px-3 py-1 rounded-full border border-teal-900/20">
                <span className="w-1.5 h-1.5 bg-teal-500 rounded-full animate-pulse"></span>
                Systems Operational
            </div>
        </div>

      </div>
    </div>
  );
};

export default LandingView;
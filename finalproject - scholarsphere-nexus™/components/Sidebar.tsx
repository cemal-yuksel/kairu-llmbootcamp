

import React from 'react';
import { 
  MessageSquare, 
  Library, 
  Users, 
  Settings, 
  BookOpen,
  X,
  ScanEye,
  Radar,
  Home,
  GraduationCap,
  Briefcase, // Updated Icon
  Headphones
} from 'lucide-react';
import { SidebarProps, View } from '../types';

const Sidebar: React.FC<SidebarProps> = ({ currentView, setCurrentView, isMobileOpen, setIsMobileOpen }) => {
  
  // Modules List (Excluding Home, Library, Settings)
  const moduleItems = [
    { id: View.CHAT, label: 'NexusGPT™', icon: <MessageSquare size={20} /> },
    { id: View.PAPER_XRAY, label: 'PaperX-Ray™', icon: <ScanEye size={20} /> },
    { id: View.RESEARCH_GAPS, label: 'Research Gaps™', icon: <Radar size={20} /> },
    { id: View.ACADEMIC_TOOLS, label: 'Academic Tools™', icon: <Briefcase size={20} /> }, // Updated
    { id: View.PODCAST, label: 'Audio Overview™', icon: <Headphones size={20} /> },
    { id: View.CO_AUTHOR, label: 'Co-Author Studio™', icon: <Users size={20} /> },
  ];

  const baseClasses = "fixed inset-y-0 left-0 z-50 w-64 bg-[#0B0C15] text-slate-400 transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 border-r border-white/10";
  const mobileClasses = isMobileOpen ? "translate-x-0" : "-translate-x-full";

  return (
    <>
      {/* Mobile Overlay */}
      {isMobileOpen && (
        <div 
          className="fixed inset-0 bg-black/60 z-40 lg:hidden backdrop-blur-sm"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      <div className={`${baseClasses} ${mobileClasses} flex flex-col`}>
        {/* Header */}
        <div className="h-16 flex items-center justify-between px-6 border-b border-white/10 bg-[#0B0C15]/50 backdrop-blur-md">
          <button 
            onClick={() => setCurrentView(View.LANDING)}
            className="flex items-center gap-2 text-teal-400 font-bold text-lg tracking-tight hover:text-white transition-colors"
          >
            <BookOpen size={24} />
            <span>ScholarSphere</span>
          </button>
          <button 
            onClick={() => setIsMobileOpen(false)}
            className="lg:hidden text-slate-400 hover:text-white"
          >
            <X size={24} />
          </button>
        </div>

        {/* Nav Items */}
        <nav className="flex-1 py-6 px-3 space-y-1 overflow-y-auto scrollbar-hide">
          
          {/* --- SYSTEM GROUP --- */}
          <button
             onClick={() => {
                setCurrentView(View.LANDING);
                setIsMobileOpen(false);
             }}
             className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 mb-1
                ${currentView === View.LANDING 
                  ? 'bg-white/5 text-white border border-white/10' 
                  : 'hover:bg-white/5 hover:text-white text-slate-400 border border-transparent'
                }`}
          >
             <Home size={20} />
             Ana Sayfa
          </button>

          <button
             onClick={() => {
                setCurrentView(View.LIBRARY);
                setIsMobileOpen(false);
             }}
             className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 mb-1
                ${currentView === View.LIBRARY
                  ? 'bg-emerald-900/20 text-emerald-400 border border-emerald-500/20' 
                  : 'hover:bg-white/5 hover:text-white text-slate-400 border border-transparent'
                }`}
          >
             <Library size={20} />
             Kütüphane
          </button>

          <button
             onClick={() => {
                setCurrentView(View.SETTINGS);
                setIsMobileOpen(false);
             }}
             className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 mb-4
                ${currentView === View.SETTINGS
                  ? 'bg-white/10 text-white border border-white/10' 
                  : 'hover:bg-white/5 hover:text-white text-slate-400 border border-transparent'
                }`}
          >
             <Settings size={20} />
             Ayarlar
          </button>

          {/* --- MODULES GROUP --- */}
          <p className="px-4 text-[10px] font-bold text-slate-600 uppercase tracking-wider mb-2 mt-6 pl-4 border-t border-white/5 pt-4">Araştırma Modülleri</p>
          
          {moduleItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setCurrentView(item.id);
                setIsMobileOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200
                ${currentView === item.id 
                  ? 'bg-teal-900/20 text-teal-400 border border-teal-500/20' 
                  : 'hover:bg-white/5 hover:text-white text-slate-400 border border-transparent'
                }`}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        {/* User Profile Stub */}
        <div className="p-4 border-t border-white/10 bg-[#0F111A]">
          <div className="flex items-center gap-3 px-2 py-2 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-teal-600 to-teal-800 flex items-center justify-center shadow-lg shadow-teal-900/20 border border-teal-500/30 group-hover:border-teal-400/50 transition-all">
                <GraduationCap size={20} className="text-white" />
            </div>
            <div className="min-w-0">
              <p className="text-sm font-bold text-white leading-tight truncate">Arş. Gör. Cemal YÜKSEL</p>
              <p className="text-[10px] font-medium text-teal-400 tracking-wide uppercase mt-0.5">Akademik Pro Lisans</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;
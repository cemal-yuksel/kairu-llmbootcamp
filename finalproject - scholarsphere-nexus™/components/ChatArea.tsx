
import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, BrainCircuit, Zap, MessageSquareText, Loader2, Bot, 
  FileText, Search, Copy, Check, ChevronDown, 
  ChevronUp, X, Eraser, Edit2, ArrowUpAZ, Calendar, Quote, Plus, Trash2, MessageSquare, ShieldCheck
} from 'lucide-react';
import { ChatMode, Message, PdfFile, ToastType, ChatSession, RagChunk } from '../types';
import { callGeminiAPI, getEmbeddings } from '../services/geminiService';
import { findMostRelevantChunks } from '../services/vectorService';
import { getAllSessions, saveSession, deleteSession } from '../services/db';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatAreaProps {
  activeFiles: PdfFile[]; // Available files from library
  addToast: (msg: string, type: ToastType) => void;
}

const ChatArea: React.FC<ChatAreaProps> = ({ activeFiles, addToast }) => {
  // Session Management
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isSessionSidebarOpen, setIsSessionSidebarOpen] = useState(true);

  // Title Editing State
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitleInput, setEditTitleInput] = useState('');

  // Chat State (derived from current session)
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<ChatMode>(ChatMode.STANDARD);
  const [isRetrieving, setIsRetrieving] = useState(false);
  
  // Context Selection State (Modal)
  const [isContextModalOpen, setIsContextModalOpen] = useState(false);
  const [sessionSelectedFileIds, setSessionSelectedFileIds] = useState<Set<string>>(new Set());
  const [contextSearchTerm, setContextSearchTerm] = useState(''); 
  const [contextSort, setContextSort] = useState<'date' | 'name'>('date');

  // UI State for Retrieval Details
  const [expandedRetrievals, setExpandedRetrievals] = useState<Set<string>>(new Set());

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // --- 1. Load Sessions on Mount & Hydrate ---
  useEffect(() => {
    const loadSessions = async () => {
        try {
            const loaded = await getAllSessions();
            
            // Robust Hydration: Ensure Dates are Date objects, not strings from JSON
            const hydrated: ChatSession[] = loaded.map(s => ({
                ...s,
                createdAt: new Date(s.createdAt),
                lastModified: new Date(s.lastModified),
                messages: s.messages.map(m => ({
                    ...m,
                    timestamp: new Date(m.timestamp)
                })),
                selectedFileIds: s.selectedFileIds || [] // Ensure array exists
            }));

            // Sort by last modified desc
            const sorted = hydrated.sort((a, b) => b.lastModified.getTime() - a.lastModified.getTime());
            setSessions(sorted);
            
            // Restore last active session from LocalStorage
            const lastActiveId = localStorage.getItem('scholar_last_session_id');
            const targetSession = sorted.find(s => s.id === lastActiveId);

            if (targetSession) {
                selectSession(targetSession);
            } else if (sorted.length > 0) {
                selectSession(sorted[0]);
            } else {
                createNewSession();
            }
        } catch (e) {
            console.error("Session load failed", e);
            createNewSession();
        }
    };
    loadSessions();
  }, []);

  // --- 2. Create New Session ---
  const createNewSession = async () => {
      const newSession: ChatSession = {
          id: Date.now().toString(),
          title: 'Yeni Sohbet',
          messages: [{
            id: 'intro',
            role: 'model',
            content: "ScholarSphere Nexus 2.0'a hoş geldiniz. Lütfen çalışmak istediğiniz kaynakları yukarıdan seçin ve sohbete başlayın.",
            timestamp: new Date()
          }],
          selectedFileIds: [],
          createdAt: new Date(),
          lastModified: new Date()
      };
      
      setSessions(prev => [newSession, ...prev]);
      await saveSession(newSession);
      selectSession(newSession);
  };

  // --- 3. Select Session ---
  const selectSession = (session: ChatSession) => {
      // Prevent selecting if we are just clicking inside the edit input
      if (editingSessionId === session.id) return;

      setCurrentSessionId(session.id);
      setMessages(session.messages);
      setSessionSelectedFileIds(new Set(session.selectedFileIds));
      
      // Persist active view to LocalStorage
      localStorage.setItem('scholar_last_session_id', session.id);
  };

  // --- 4. Delete Session ---
  const handleDeleteSession = async (e: React.MouseEvent, id: string) => {
      e.stopPropagation();
      if (!window.confirm("Sohbeti silmek istediğinize emin misiniz?")) return;

      setSessions(prev => prev.filter(s => s.id !== id));
      await deleteSession(id);
      
      if (currentSessionId === id) {
          const remaining = sessions.filter(s => s.id !== id);
          if (remaining.length > 0) {
              selectSession(remaining[0]);
          } else {
              createNewSession();
          }
      }
  };

  // --- 4b. Edit Session Title Handlers ---
  const startEditing = (e: React.MouseEvent, session: ChatSession) => {
      e.stopPropagation();
      setEditingSessionId(session.id);
      setEditTitleInput(session.title);
  };

  const saveTitle = async (e?: React.MouseEvent) => {
      if (e) e.stopPropagation();
      if (!editingSessionId) return;

      const trimmedTitle = editTitleInput.trim() || 'İsimsiz Sohbet';
      
      const sessionToUpdate = sessions.find(s => s.id === editingSessionId);
      if (sessionToUpdate) {
          const updatedSession = { 
              ...sessionToUpdate, 
              title: trimmedTitle, 
              lastModified: new Date() 
          };
          
          setSessions(prev => prev.map(s => s.id === editingSessionId ? updatedSession : s));
          await saveSession(updatedSession);
          addToast("Başlık güncellendi.", 'success');
      }
      setEditingSessionId(null);
  };

  const cancelEdit = (e: React.MouseEvent) => {
      e.stopPropagation();
      setEditingSessionId(null);
  };

  const handleTitleKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
          e.preventDefault();
          saveTitle();
      } else if (e.key === 'Escape') {
          e.preventDefault();
          setEditingSessionId(null);
      }
  };

  // --- 5. Clear Chat History (Keep Session) ---
  const handleClearHistory = async () => {
      if (!currentSessionId) return;
      if (!window.confirm("Bu sohbetin mesaj geçmişini temizlemek istediğinize emin misiniz? Başlık ve seçili dosyalar korunacaktır.")) return;

      const introMsg: Message = {
        id: 'intro',
        role: 'model',
        content: "Sohbet geçmişi temizlendi. Kaldığımız yerden devam edebilir veya yeni bir konu başlatabilirsiniz.",
        timestamp: new Date()
      };

      setMessages([introMsg]);

      const sessionToUpdate = sessions.find(s => s.id === currentSessionId);
      if (sessionToUpdate) {
          const newSession = { 
              ...sessionToUpdate, 
              messages: [introMsg], 
              lastModified: new Date() 
          };
          setSessions(prev => prev.map(s => s.id === currentSessionId ? newSession : s));
          await saveSession(newSession);
          addToast("Sohbet temizlendi.", 'success');
      }
  };

  // --- 6. Context Management Logic ---
  
  // Filter files for the modal view
  const getFilteredFiles = () => {
      let files = activeFiles.filter(f => f.title.toLowerCase().includes(contextSearchTerm.toLowerCase()));
      if (contextSort === 'date') {
          files.sort((a, b) => new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime());
      } else {
          files.sort((a, b) => a.title.localeCompare(b.title));
      }
      return files;
  };

  const toggleFileForSession = async (fileId: string) => {
      if (!currentSessionId) return;

      const newSet = new Set(sessionSelectedFileIds);
      if (newSet.has(fileId)) newSet.delete(fileId);
      else newSet.add(fileId);
      
      setSessionSelectedFileIds(newSet);

      // Update Session persistence
      const updatedSession = sessions.find(s => s.id === currentSessionId);
      if (updatedSession) {
          const newSession = { ...updatedSession, selectedFileIds: Array.from(newSet) };
          setSessions(prev => prev.map(s => s.id === currentSessionId ? newSession : s));
          await saveSession(newSession);
      }
  };

  const selectAllFiles = async () => {
    if (!currentSessionId) return;
    const allIds = activeFiles.map(f => f.id);
    const newSet = new Set(allIds);
    setSessionSelectedFileIds(newSet);
    
    const updatedSession = sessions.find(s => s.id === currentSessionId);
    if (updatedSession) {
        const newSession = { ...updatedSession, selectedFileIds: allIds };
        setSessions(prev => prev.map(s => s.id === currentSessionId ? newSession : s));
        await saveSession(newSession);
    }
  };

  const clearFileSelection = async () => {
    if (!currentSessionId) return;
    setSessionSelectedFileIds(new Set());
    
    const updatedSession = sessions.find(s => s.id === currentSessionId);
    if (updatedSession) {
        const newSession = { ...updatedSession, selectedFileIds: [] };
        setSessions(prev => prev.map(s => s.id === currentSessionId ? newSession : s));
        await saveSession(newSession);
    }
  };

  const toggleRetrievalView = (msgId: string) => {
      const newSet = new Set(expandedRetrievals);
      if (newSet.has(msgId)) newSet.delete(msgId);
      else newSet.add(msgId);
      setExpandedRetrievals(newSet);
  };

  // Scroll logic
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(() => { scrollToBottom(); }, [messages]);

  // --- SEND HANDLER ---
  const handleSend = async () => {
    if (!input.trim() || isLoading || !currentSessionId) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    // Update Local UI State
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setInput('');
    setIsLoading(true);

    // Update Session Title if it's the first real message (Auto-Generation)
    let updatedTitle = sessions.find(s => s.id === currentSessionId)?.title || 'Yeni Sohbet';
    
    if (updatedMessages.length === 2 && (updatedTitle === 'Yeni Sohbet' || updatedTitle === '')) { 
        updatedTitle = userMsg.content.substring(0, 30) + (userMsg.content.length > 30 ? '...' : '');
    }

    const sessionToUpdate = sessions.find(s => s.id === currentSessionId);
    if (sessionToUpdate) {
        const newSession = { 
            ...sessionToUpdate, 
            messages: updatedMessages, 
            title: updatedTitle,
            lastModified: new Date() 
        };
        setSessions(prev => prev.map(s => s.id === currentSessionId ? newSession : s));
        await saveSession(newSession);
    }

    // Optimistic UI
    const tempId = 'thinking-' + Date.now();
    const thinkingMsg: Message = { 
        id: tempId, 
        role: 'model', 
        content: '', 
        timestamp: new Date(), 
        isThinking: true 
    };
    setMessages(prev => [...prev, thinkingMsg]);

    try {
        // RAG Logic with Session Specific Files
        let contextString = "";
        const sessionFiles = activeFiles.filter(f => sessionSelectedFileIds.has(f.id));
        let chunksUsed: RagChunk[] = [];
        
        if (sessionFiles.length > 0) {
            setIsRetrieving(true);
            try {
                const queryEmbedding = await getEmbeddings(userMsg.content);
                const allChunks = sessionFiles.flatMap(f => f.chunks || []);
                const relevantChunks = findMostRelevantChunks(queryEmbedding, allChunks, 10);
                
                chunksUsed = relevantChunks.map(c => {
                    const file = sessionFiles.find(f => f.id === c.fileId);
                    return { ...c, sourceTitle: file?.title || 'Bilinmeyen Kaynak' };
                });

                if (chunksUsed.length > 0) {
                    contextString = chunksUsed.map(chunk => {
                        return `[Kaynak: ${chunk.sourceTitle} | Sayfa: ${chunk.pageNumber}]\n${chunk.content}`;
                    }).join('\n\n---\n\n');
                } else {
                    contextString = sessionFiles.slice(0, 1).map(f => f.text?.substring(0, 20000)).join('\n');
                }

            } catch (vecError) {
                console.error("Vector Search Failed", vecError);
                addToast("Vektör arama başarısız, standart mod.", 'warning');
                contextString = sessionFiles.map(f => f.text || '').join('\n').substring(0, 30000);
            } finally {
                setIsRetrieving(false);
            }
        }

        const history = updatedMessages.filter(m => m.id !== 'intro');
        const responseText = await callGeminiAPI(history, userMsg.content, mode, contextString);
      
        const finalModelMsg: Message = {
            id: (Date.now() + 1).toString(),
            role: 'model',
            content: responseText,
            timestamp: new Date(),
            retrievedChunks: chunksUsed
        };

        const finalMessages = [...updatedMessages, finalModelMsg];
        setMessages(finalMessages);

        if (sessionToUpdate) {
            const completedSession = { 
                ...sessionToUpdate, 
                messages: finalMessages,
                title: updatedTitle,
                lastModified: new Date()
            };
            setSessions(prev => prev.map(s => s.id === currentSessionId ? completedSession : s));
            await saveSession(completedSession);
        }

    } catch (error: any) {
      console.error(error);
      setMessages(prev => prev.filter(m => m.id !== tempId));
      addToast(error.message || "Bir hata oluştu.", 'error');
      
      const errorMsg: Message = {
          id: (Date.now() + 1).toString(),
          role: 'model',
          content: "⚠️ Bir hata oluştu: " + (error.message || "Bilinmeyen hata."),
          timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
      setIsRetrieving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    addToast("Metin kopyalandı", 'success');
  };

  return (
    <div className="flex h-full bg-[#0B0C15] relative font-sans overflow-hidden text-slate-200">
      
      {/* LEFT SIDEBAR: CHAT SESSIONS */}
      <div className={`bg-[#0F111A] border-r border-white/10 flex flex-col transition-all duration-300 shrink-0 
          ${isSessionSidebarOpen ? 'w-64' : 'w-0 overflow-hidden'}
      `}>
          <div className="p-4 border-b border-white/10 flex items-center justify-between">
              <h3 className="font-bold text-slate-200">Sohbetler</h3>
              <button onClick={createNewSession} className="p-1.5 bg-teal-500/10 text-teal-400 rounded-lg hover:bg-teal-500/20 transition-colors">
                  <Plus size={18} />
              </button>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-hide">
              {sessions.map(session => (
                  <div 
                    key={session.id}
                    onClick={() => selectSession(session)}
                    className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all min-h-[44px]
                        ${currentSessionId === session.id 
                            ? 'bg-teal-500/10 border border-teal-500/20 text-teal-100' 
                            : 'hover:bg-white/5 text-slate-400 border border-transparent'}
                    `}
                  >
                      {editingSessionId === session.id ? (
                          <div className="flex items-center gap-1 w-full" onClick={(e) => e.stopPropagation()}>
                              <input 
                                autoFocus
                                type="text" 
                                value={editTitleInput}
                                onChange={(e) => setEditTitleInput(e.target.value)}
                                onKeyDown={handleTitleKeyDown}
                                className="flex-1 min-w-0 text-sm px-2 py-1 border border-teal-500 rounded bg-slate-900 text-white focus:outline-none focus:ring-1 focus:ring-teal-400"
                              />
                              <button onClick={saveTitle} className="p-1 text-emerald-400 hover:bg-emerald-900/30 rounded"><Check size={14} /></button>
                              <button onClick={cancelEdit} className="p-1 text-red-400 hover:bg-red-900/30 rounded"><X size={14} /></button>
                          </div>
                      ) : (
                          <>
                            <div className="flex items-center gap-3 overflow-hidden flex-1">
                                <MessageSquare size={16} className={`shrink-0 ${currentSessionId === session.id ? 'text-teal-400' : 'text-slate-500'}`} />
                                <span className={`text-sm truncate ${currentSessionId === session.id ? 'font-medium' : ''}`}>
                                    {session.title}
                                </span>
                            </div>
                            <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <button onClick={(e) => startEditing(e, session)} className="p-1.5 text-slate-400 hover:text-indigo-400 hover:bg-indigo-500/10 rounded transition-colors"><Edit2 size={14} /></button>
                                <button onClick={(e) => handleDeleteSession(e, session.id)} className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"><Trash2 size={14} /></button>
                            </div>
                          </>
                      )}
                  </div>
              ))}
          </div>
      </div>

      {/* MAIN CHAT AREA */}
      <div className="flex-1 flex flex-col h-full relative">
        
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-[0.02] pointer-events-none bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] mix-blend-overlay"></div>

        {/* Header */}
        <div className="flex flex-col bg-[#0B0C15]/90 backdrop-blur-md border-b border-white/10 shadow-sm z-10">
            <div className="flex items-center justify-between py-3 px-4">
                <div className="flex items-center gap-3">
                    <button onClick={() => setIsSessionSidebarOpen(!isSessionSidebarOpen)} className="text-slate-400 hover:text-white">
                        <MessageSquareText size={20} />
                    </button>
                    <button onClick={createNewSession} className="p-1.5 text-slate-400 hover:text-teal-400 hover:bg-teal-500/10 rounded-lg transition-colors" title="Yeni Sohbet">
                        <Plus size={20} />
                    </button>
                    <button onClick={handleClearHistory} className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors" title="Geçmişi Temizle">
                        <Eraser size={20} />
                    </button>

                    <div className="flex flex-col">
                        <span className="text-sm font-bold text-white">NexusGPT™</span>
                        <div className="flex items-center gap-1 text-[10px] text-slate-400">
                            <span className={`w-1.5 h-1.5 rounded-full ${sessionSelectedFileIds.size > 0 ? 'bg-green-500 animate-pulse' : 'bg-slate-600'}`}></span>
                            {sessionSelectedFileIds.size} Kaynak Aktif
                        </div>
                    </div>
                </div>
                
                {/* Context Selector Button */}
                <button 
                    onClick={() => setIsContextModalOpen(true)}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all shadow-sm
                        ${sessionSelectedFileIds.size > 0 
                            ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30 hover:bg-indigo-500/30' 
                            : 'bg-white/5 text-slate-400 border-white/10 hover:bg-white/10'}
                    `}
                >
                    <FileText size={14} />
                    <span>
                        {sessionSelectedFileIds.size > 0 
                            ? `${sessionSelectedFileIds.size} Kaynak Seçili` 
                            : 'Kaynak Seçimi'}
                    </span>
                    <ChevronDown size={12} />
                </button>
            </div>

            {/* Mode Selector Strip */}
            <div className="flex bg-[#0F111A] border-t border-white/5 px-4 py-1 gap-2 overflow-x-auto scrollbar-hide">
                {[
                    { m: ChatMode.FAST, icon: <Zap size={14} />, label: 'Hızlı' },
                    { m: ChatMode.STANDARD, icon: <Bot size={14} />, label: 'Standart' },
                    { m: ChatMode.DEEP_THINK, icon: <BrainCircuit size={14} />, label: 'Derin Düşün' },
                    { m: ChatMode.DEEP_SEARCH, icon: <ShieldCheck size={14} />, label: 'Doğrulama' }
                ].map(opt => (
                    <button
                        key={opt.m}
                        onClick={() => setMode(opt.m)}
                        className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wide whitespace-nowrap transition-all
                            ${mode === opt.m 
                                ? 'bg-teal-500/10 text-teal-400 ring-1 ring-teal-500/30' 
                                : 'text-slate-500 hover:bg-white/5 hover:text-slate-300'}
                        `}
                    >
                        {opt.icon}
                        {opt.label}
                    </button>
                ))}
            </div>

            {/* Selected Files Chips Strip */}
            {sessionSelectedFileIds.size > 0 && (
                <div className="flex items-center gap-2 px-4 py-2 bg-indigo-900/20 border-t border-indigo-500/20 overflow-x-auto scrollbar-hide shrink-0 animate-in slide-in-from-top-1">
                    <span className="text-[10px] font-bold text-indigo-300 uppercase tracking-wider shrink-0 flex items-center gap-1">
                        <FileText size={10} />
                        Bağlam:
                    </span>
                    {activeFiles.filter(f => sessionSelectedFileIds.has(f.id)).map(f => (
                        <div key={f.id} className="flex items-center gap-1 pl-2 pr-1 py-1 bg-[#0F111A] border border-indigo-500/30 rounded-md text-[10px] font-medium text-indigo-300 shrink-0 shadow-sm group">
                            <span className="truncate max-w-[120px]">{f.title}</span>
                            <button
                                onClick={(e) => { e.stopPropagation(); toggleFileForSession(f.id); }}
                                className="p-0.5 hover:bg-red-500/20 text-indigo-400 hover:text-red-400 rounded transition-colors"
                            >
                                <X size={10} />
                            </button>
                        </div>
                    ))}
                    <button onClick={clearFileSelection} className="ml-auto px-2 py-1 text-[10px] text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded">Temizle</button>
                </div>
            )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6 z-0 scrollbar-hide bg-[#0B0C15]">
            {messages.map((msg) => (
            <div key={msg.id} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-slide-up`}>
                <div className={`flex max-w-[90%] md:max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'} items-end gap-3`}>
                    {msg.role === 'model' && (
                        <div className="w-8 h-8 rounded-full bg-[#0F111A] border border-white/10 shadow-sm flex items-center justify-center shrink-0 text-teal-400 mb-4">
                            {msg.isThinking ? <BrainCircuit size={16} className="animate-pulse" /> : <Bot size={18} />}
                        </div>
                    )}
                    <div className="flex flex-col gap-1 min-w-0 w-full">
                        <div className={`relative px-5 py-4 shadow-sm text-sm leading-relaxed group 
                            ${msg.role === 'user' 
                                ? 'bg-teal-600/20 text-teal-50 border border-teal-500/20 rounded-2xl rounded-tr-none' 
                                : 'bg-[#0F111A] text-slate-300 border border-white/5 rounded-2xl rounded-tl-none'}
                        `}>
                            {msg.isThinking ? (
                                <div className="flex flex-col gap-2 min-w-[220px]">
                                    <div className="flex items-center gap-2 text-teal-400">
                                        <Loader2 size={16} className="animate-spin" />
                                        <span className="font-semibold">{isRetrieving ? 'Vektör taranıyor...' : 'ScholarSphere analiz ediyor...'}</span>
                                    </div>
                                </div>
                            ) : (
                                <div className={`markdown-body ${msg.role === 'user' ? 'text-teal-50' : 'text-slate-300'}`}>
                                    {msg.role === 'user' ? <div className="whitespace-pre-wrap">{msg.content}</div> : (
                                        <>
                                            <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
                                                p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
                                                a: ({node, ...props}) => <a className="text-teal-400 hover:underline" {...props} />,
                                                code: ({node, inline, className, children, ...props}: any) => inline 
                                                    ? <code className="bg-slate-800 text-slate-200 px-1 py-0.5 rounded text-xs border border-white/10" {...props}>{children}</code>
                                                    : <code className="block bg-[#1A1D26] text-slate-300 p-3 rounded-lg text-xs overflow-x-auto my-2 border border-white/10" {...props}>{children}</code>,
                                                th: ({node, ...props}) => <th className="bg-slate-800 border-slate-700 text-slate-200 p-2" {...props} />,
                                                td: ({node, ...props}) => <td className="border-slate-700 p-2" {...props} />,
                                                strong: ({node, ...props}) => <strong className="text-white font-bold" {...props} />
                                            }}>{msg.content}</ReactMarkdown>
                                            {msg.retrievedChunks && msg.retrievedChunks.length > 0 && (
                                                <div className="mt-6 pt-4 border-t border-white/10">
                                                    <button onClick={() => toggleRetrievalView(msg.id)} className="flex items-center gap-2 text-xs font-bold text-indigo-400 hover:text-indigo-300 mb-3 bg-indigo-500/10 border border-indigo-500/20 px-3 py-1.5 rounded-lg w-fit">
                                                        <Quote size={12} /> Kullanılan Kaynaklar ({msg.retrievedChunks.length}) {expandedRetrievals.has(msg.id) ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                                                    </button>
                                                    {expandedRetrievals.has(msg.id) && (
                                                        <div className="flex flex-col gap-2 animate-in fade-in slide-in-from-top-2">
                                                            {msg.retrievedChunks.map((chunk, i) => (
                                                                <div key={i} className="p-3 bg-[#15171F] rounded-lg border border-white/5 hover:border-indigo-500/30 transition-colors">
                                                                    <div className="flex justify-between mb-1.5"><span className="text-[11px] font-bold text-indigo-300 truncate">{chunk.sourceTitle}</span><span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 rounded">s.{chunk.pageNumber}</span></div>
                                                                    <p className="text-[11px] text-slate-500 italic line-clamp-2">"{chunk.content.substring(0, 150)}..."</p>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </>
                                    )}
                                </div>
                            )}
                            {!msg.isThinking && msg.role === 'model' && (
                                <div className="absolute -bottom-6 left-0 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button onClick={() => handleCopy(msg.content)} className="text-slate-500 hover:text-teal-400 p-1"><Copy size={14} /></button>
                                </div>
                            )}
                        </div>
                        {!msg.isThinking && <div className={`text-[10px] pr-1 opacity-60 ${msg.role === 'user' ? 'text-right text-slate-500' : 'text-left text-slate-600'}`}>{msg.timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>}
                    </div>
                </div>
            </div>
            ))}
            <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 bg-[#0B0C15] border-t border-white/10">
            <div className="max-w-4xl mx-auto relative flex items-end gap-2 bg-[#0F111A] p-2 rounded-xl border border-white/10 shadow-sm focus-within:border-teal-500/50 focus-within:ring-1 focus-within:ring-teal-500/20 transition-all">
            <textarea value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder="Sohbete başla..." className="flex-1 max-h-32 min-h-[44px] py-2.5 px-3 bg-transparent border-none focus:ring-0 resize-none text-sm text-slate-200 placeholder:text-slate-500" rows={1} />
            <button onClick={handleSend} disabled={isLoading || !input.trim()} className={`p-2.5 rounded-lg mb-0.5 transition-all duration-200 shrink-0 ${input.trim() && !isLoading ? 'bg-teal-600 text-white shadow-md hover:bg-teal-500 hover:scale-105' : 'bg-white/5 text-slate-500 cursor-not-allowed'}`}>{isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}</button>
            </div>
        </div>

        {/* --- CONTEXT SELECTION MODAL --- */}
        {isContextModalOpen && (
            <div className="absolute inset-0 z-[60] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-200">
                <div className="bg-[#0F111A] rounded-2xl shadow-2xl w-full max-w-3xl max-h-[80vh] flex flex-col overflow-hidden border border-white/10 animate-in zoom-in-95 duration-200">
                    {/* Modal Header */}
                    <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between bg-[#0B0C15]">
                        <div>
                            <h2 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                                <FileText size={20} className="text-indigo-400" />
                                Sohbet Bağlamı Yönetimi
                            </h2>
                            <p className="text-xs text-slate-500 mt-0.5">Yapay zekanın cevap verirken kullanacağı kaynakları seçin.</p>
                        </div>
                        <button onClick={() => setIsContextModalOpen(false)} className="p-2 hover:bg-white/10 rounded-full text-slate-400 transition-colors">
                            <X size={20} />
                        </button>
                    </div>

                    {/* Toolbar */}
                    <div className="px-6 py-3 border-b border-white/10 flex flex-col sm:flex-row sm:items-center gap-3 bg-[#0F111A]">
                        <div className="relative flex-1">
                            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                            <input 
                                type="text" 
                                value={contextSearchTerm}
                                onChange={(e) => setContextSearchTerm(e.target.value)}
                                placeholder="Dosya adı ile ara..." 
                                className="w-full pl-9 pr-4 py-2 text-sm bg-[#0B0C15] border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500/50 transition-all text-slate-200 placeholder:text-slate-600"
                                autoFocus
                            />
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                             <button 
                                onClick={() => setContextSort(contextSort === 'date' ? 'name' : 'date')}
                                className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-slate-400 bg-[#0B0C15] border border-white/10 rounded-lg hover:bg-white/5 transition-colors"
                             >
                                {contextSort === 'date' ? <Calendar size={14} /> : <ArrowUpAZ size={14} />}
                                {contextSort === 'date' ? 'Tarihe Göre' : 'İsme Göre'}
                             </button>
                             <div className="w-px h-6 bg-white/10 mx-1"></div>
                             <button onClick={selectAllFiles} className="px-3 py-2 text-xs font-medium text-indigo-400 hover:bg-indigo-500/10 rounded-lg transition-colors">Tümünü Seç</button>
                             <button onClick={clearFileSelection} className="px-3 py-2 text-xs font-medium text-red-400 hover:bg-red-500/10 rounded-lg transition-colors">Temizle</button>
                        </div>
                    </div>

                    {/* File Grid */}
                    <div className="flex-1 overflow-y-auto p-6 bg-[#0B0C15]">
                        {getFilteredFiles().length === 0 ? (
                             <div className="h-full flex flex-col items-center justify-center text-slate-600 opacity-60">
                                <Search size={48} className="mb-4" />
                                <p>Eşleşen dosya bulunamadı.</p>
                             </div>
                        ) : (
                             <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                {getFilteredFiles().map(file => {
                                    const isSelected = sessionSelectedFileIds.has(file.id);
                                    return (
                                        <div 
                                            key={file.id}
                                            onClick={() => toggleFileForSession(file.id)}
                                            className={`relative group p-3 rounded-xl border transition-all duration-200 cursor-pointer flex items-start gap-3
                                                ${isSelected 
                                                    ? 'bg-indigo-500/10 border-indigo-500/30 shadow-sm ring-1 ring-indigo-500/20' 
                                                    : 'bg-[#0F111A] border-white/5 hover:border-indigo-500/20 hover:bg-[#151720]'}
                                            `}
                                        >
                                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 transition-colors
                                                ${isSelected ? 'bg-indigo-500/20 text-indigo-400' : 'bg-[#0B0C15] text-slate-500'}
                                            `}>
                                                <FileText size={20} />
                                            </div>
                                            
                                            <div className="flex-1 min-w-0">
                                                <h4 className={`text-sm font-semibold truncate ${isSelected ? 'text-indigo-300' : 'text-slate-300'}`}>{file.title}</h4>
                                                <div className="flex items-center gap-2 mt-1 text-[10px] text-slate-500">
                                                    <span>{new Date(file.uploadDate).toLocaleDateString()}</span>
                                                    <span>•</span>
                                                    <span>{file.pageCount || '?'} sayfa</span>
                                                </div>
                                            </div>

                                            <div className={`w-5 h-5 rounded-full border flex items-center justify-center transition-all
                                                ${isSelected ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-transparent border-slate-600 group-hover:border-indigo-400'}
                                            `}>
                                                {isSelected && <Check size={12} />}
                                            </div>
                                        </div>
                                    );
                                })}
                             </div>
                        )}
                    </div>
                    
                    {/* Footer */}
                    <div className="p-4 border-t border-white/10 bg-[#0F111A] flex items-center justify-between">
                        <div className="text-xs text-slate-400">
                            <span className="font-bold text-slate-200">{sessionSelectedFileIds.size}</span> dosya seçili
                        </div>
                        <button 
                            onClick={() => setIsContextModalOpen(false)}
                            className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-semibold text-sm shadow-md shadow-indigo-900/20 transition-all hover:shadow-lg"
                        >
                            Seçimi Tamamla
                        </button>
                    </div>
                </div>
            </div>
        )}
      </div>
    </div>
  );
};

export default ChatArea;
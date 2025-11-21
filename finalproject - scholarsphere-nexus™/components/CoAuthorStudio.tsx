
import React, { useState, useEffect, useRef } from 'react';
import { Send, Check, Play, RotateCcw, FileText, Bot, User, PenTool, ChevronRight, Save, Download, Loader2 } from 'lucide-react';
import { PdfFile } from '../types';
import { generateCoAuthorStep } from '../services/geminiService';
import { generateResearchReport } from '../services/pdfService';

interface CoAuthorStudioProps {
  activeFiles: PdfFile[];
}

// The roadmap of an academic paper
const SECTIONS = [
  'Araştırma Konusu & Başlık',
  'Giriş (Introduction)',
  'Literatür Taraması',
  'Metodoloji',
  'Bulgular ve Tartışma',
  'Sonuç ve Öneriler'
];

interface ChatMsg {
  id: string;
  role: 'ai' | 'user';
  text: string;
  isDraftProposal?: boolean; // If true, this message contains a draft text to be approved
}

const CoAuthorStudio: React.FC<CoAuthorStudioProps> = ({ activeFiles }) => {
  // State for the Writing Process
  const [currentStep, setCurrentStep] = useState(0); // 0 = Topic, 1 = Intro, etc.
  const [researchTopic, setResearchTopic] = useState("");
  const [fullDraft, setFullDraft] = useState("");
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [isInitialized, setIsInitialized] = useState(false);
  
  // Chat State
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // --- PERSISTENCE: Hydrate ---
  useEffect(() => {
      const savedState = localStorage.getItem('scholar_coauthor_state');
      if (savedState) {
          try {
              const parsed = JSON.parse(savedState);
              setResearchTopic(parsed.researchTopic || "");
              setFullDraft(parsed.fullDraft || "");
              setCurrentStep(parsed.currentStep || 0);
              setMessages(parsed.messages || []);
          } catch (e) {
              console.error("CoAuthor hydration failed", e);
              setInitialMsg();
          }
      } else {
          setInitialMsg();
      }
      setIsInitialized(true);
  }, []);

  const setInitialMsg = () => {
      setMessages([{
        id: '1',
        role: 'ai',
        text: "Merhaba! Co-Author Studio™'ya hoş geldiniz. Ben akademik yazım ortağınızım. \n\nKütüphanenizdeki kaynakları kullanarak adım adım makalenizi yazacağız. Öncelikle, çalışmak istediğiniz araştırma konusunu veya sorusunu yazar mısınız? (Örn: 'Sosyal medyanın tüketici aktivizmi üzerindeki etkisi')"
      }]);
  };

  // --- PERSISTENCE: Save ---
  useEffect(() => {
      if (isInitialized) {
          const stateToSave = {
              researchTopic,
              fullDraft,
              currentStep,
              messages
          };
          localStorage.setItem('scholar_coauthor_state', JSON.stringify(stateToSave));
      }
  }, [researchTopic, fullDraft, currentStep, messages, isInitialized]);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Helper to build context from active files
  const getContextString = () => {
    return activeFiles.map(f => 
       `KAYNAK: ${f.title}\n${f.chunks.slice(0, 5).map(c => c.content).join('\n')}...`
    ).join('\n\n');
  };

  const handleExportPDF = async () => {
    if (!fullDraft || isExporting) return;
    setIsExporting(true);
    
    try {
        // Naive section splitting by markdown headers for better PDF structure
        const sectionsRaw = fullDraft.split('## ');
        const sections = sectionsRaw.map(sec => {
            const [title, ...content] = sec.split('\n');
            return {
                title: title ? title.trim() : "Giriş",
                content: content.join('\n').trim()
            };
        }).filter(s => s.content); // Remove empty sections

        await generateResearchReport(
            researchTopic || "Akademik Taslak",
            "Co-Author Studio",
            sections
        );
    } catch (e) {
        console.error(e);
        alert("PDF oluşturulamadı.");
    } finally {
        setIsExporting(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userText = input;
    setInput("");
    
    // Add User Message
    const newMsgs: ChatMsg[] = [...messages, { id: Date.now().toString(), role: 'user', text: userText }];
    setMessages(newMsgs);
    setIsLoading(true);

    try {
        // LOGIC FLOW
        
        // STEP 0: Setting the Topic
        if (currentStep === 0) {
            setResearchTopic(userText);
            setMessages(prev => [...prev, { 
                id: 'thinking', role: 'ai', text: 'Konu analizi yapılıyor ve kaynaklar taranıyor...' 
            }]);

            // Immediately trigger the first section (Introduction) generation
            const context = getContextString();
            if (!context && activeFiles.length === 0) {
                 setMessages(prev => prev.filter(m => m.id !== 'thinking').concat({
                    id: Date.now().toString(),
                    role: 'ai',
                    text: "Uyarı: Kütüphanede seçili belge yok. Lütfen önce 'Kütüphane' sekmesinden PDF yükleyin ve seçin, böylece atıf yapabilirim."
                 }));
                 setIsLoading(false);
                 return;
            }

            const draftText = await generateCoAuthorStep(userText, SECTIONS[1], context, "İlk taslağı oluştur.");
            
            setMessages(prev => prev.filter(m => m.id !== 'thinking').concat({
                id: Date.now().toString(),
                role: 'ai',
                text: draftText,
                isDraftProposal: true
            }));
            
            setCurrentStep(1); // Move state to Intro
        } 
        // STEPS 1+: Reviewing/Revising/Moving Next
        else {
            // Detect intent: Approval or Revision?
            // Simple keyword check for prototype (in production, use LLM classifier)
            const isApproval = userText.toLowerCase().includes('tamam') || 
                               userText.toLowerCase().includes('evet') || 
                               userText.toLowerCase().includes('onay') ||
                               userText.toLowerCase().includes('devam');

            if (isApproval) {
                // 1. Find the last proposed draft in chat history
                const lastProposal = [...messages].reverse().find(m => m.isDraftProposal);
                
                if (lastProposal) {
                    // 2. Append to Full Draft
                    const sectionTitle = SECTIONS[currentStep];
                    const newDraftContent = `\n\n## ${sectionTitle}\n\n${lastProposal.text}`;
                    setFullDraft(prev => prev + newDraftContent);
                    
                    // 3. Move to next step
                    const nextStepIndex = currentStep + 1;
                    if (nextStepIndex < SECTIONS.length) {
                         setCurrentStep(nextStepIndex);
                         setMessages(prev => [...prev, { 
                            id: 'thinking', role: 'ai', text: `${SECTIONS[nextStepIndex]} bölümü yazılıyor...` 
                         }]);
                         
                         const context = getContextString();
                         const nextDraft = await generateCoAuthorStep(researchTopic, SECTIONS[nextStepIndex], context, "Önceki bölüm onaylandı. Sıradaki bölümü yaz.");
                         
                         setMessages(prev => prev.filter(m => m.id !== 'thinking').concat({
                            id: Date.now().toString(),
                            role: 'ai',
                            text: nextDraft,
                            isDraftProposal: true
                         }));
                    } else {
                         setMessages(prev => [...prev, {
                             id: 'done', role: 'ai', text: "Tebrikler! Makale taslağı tamamlandı. Sağ panelden kopyalayabilirsiniz."
                         }]);
                    }
                }
            } else {
                // Revision Request
                setMessages(prev => [...prev, { 
                    id: 'thinking', role: 'ai', text: `"${SECTIONS[currentStep]}" bölümü revize ediliyor...` 
                }]);
                
                const context = getContextString();
                const revisedDraft = await generateCoAuthorStep(researchTopic, SECTIONS[currentStep], context, `Kullanıcı revize istedi: ${userText}. Lütfen bölümü tekrar yaz.`);
                
                setMessages(prev => prev.filter(m => m.id !== 'thinking').concat({
                    id: Date.now().toString(),
                    role: 'ai',
                    text: revisedDraft,
                    isDraftProposal: true
                }));
            }
        }

    } catch (error) {
        console.error(error);
        setMessages(prev => [...prev, { id: 'err', role: 'ai', text: "Bir hata oluştu. Lütfen tekrar deneyin." }]);
    } finally {
        setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-full overflow-hidden bg-[#0B0C15] font-sans text-slate-300">
      
      {/* LEFT: Writing Assistant Chat (45%) */}
      <div className="w-full lg:w-[45%] flex flex-col bg-[#0F111A] border-r border-white/10">
         
         {/* Header */}
         <div className="h-16 px-6 border-b border-white/10 flex items-center justify-between bg-[#0B0C15] text-white">
            <div className="flex items-center gap-3">
               <PenTool size={20} className="text-teal-400" />
               <div>
                  <h2 className="font-bold text-sm">Co-Author Studio™</h2>
                  <p className="text-[10px] text-slate-500 flex items-center gap-1">
                    {activeFiles.length > 0 ? (
                        <span className="text-green-400 flex items-center gap-1"><Check size={10} /> {activeFiles.length} Kaynak Bağlı</span>
                    ) : (
                        <span className="text-red-400">Kaynak Seçilmedi</span>
                    )}
                  </p>
               </div>
            </div>
            <div className="px-3 py-1 bg-teal-900/30 rounded text-xs font-medium text-teal-400 border border-teal-500/30">
                Adım {currentStep + 1}/{SECTIONS.length}: {SECTIONS[currentStep].split(' ')[0]}
            </div>
         </div>

         {/* Chat Flow */}
         <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-[#0F111A]">
            {messages.map((msg) => (
               <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 shadow-sm
                      ${msg.role === 'ai' ? 'bg-[#151720] text-teal-400 border border-white/5' : 'bg-teal-600 text-white'}
                  `}>
                      {msg.role === 'ai' ? <Bot size={18} /> : <User size={18} />}
                  </div>
                  
                  <div className={`flex flex-col gap-2 max-w-[85%]`}>
                      <div className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm
                          ${msg.role === 'user' 
                             ? 'bg-teal-600/20 text-teal-100 border border-teal-500/30 rounded-tr-none' 
                             : 'bg-[#151720] text-slate-300 border border-white/5 rounded-tl-none'}
                      `}>
                          {msg.text.startsWith('#') ? (
                             /* Simple Markdown-ish render for draft proposals */
                             <div className="prose prose-invert prose-sm prose-p:my-1 prose-headings:text-teal-400 prose-headings:font-bold">
                                 <div className="whitespace-pre-wrap font-serif">{msg.text}</div>
                             </div>
                          ) : (
                             <div className="whitespace-pre-wrap">{msg.text}</div>
                          )}
                      </div>

                      {/* Action Buttons for AI Draft Proposals */}
                      {msg.role === 'ai' && msg.isDraftProposal && (
                          <div className="flex items-center gap-2 mt-1 animate-in fade-in slide-in-from-top-2">
                              <span className="text-xs text-slate-500">Nasıl devam edelim?</span>
                              <button 
                                onClick={() => { setInput("Tamam, harika. Devam edelim."); handleSend(); }}
                                className="flex items-center gap-1 px-3 py-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 rounded-md text-xs font-bold transition-colors border border-emerald-500/20"
                              >
                                <Check size={12} /> Onayla & Devam Et
                              </button>
                              <button 
                                onClick={() => setInput("Şurası olmamış, daha fazla detay ver...")}
                                className="flex items-center gap-1 px-3 py-1.5 bg-[#151720] border border-white/10 hover:bg-white/5 text-slate-400 rounded-md text-xs font-medium transition-colors"
                              >
                                <RotateCcw size={12} /> Revize İste
                              </button>
                          </div>
                      )}
                  </div>
               </div>
            ))}
            <div ref={messagesEndRef} />
         </div>

         {/* Input Area */}
         <div className="p-4 bg-[#0B0C15] border-t border-white/10">
            <div className="relative flex items-end gap-2">
               <textarea
                 value={input}
                 onChange={(e) => setInput(e.target.value)}
                 onKeyDown={handleKeyDown}
                 placeholder={currentStep === 0 ? "Araştırma konunuzu yazın..." : "Cevabınızı yazın (örn: 'Tamam devam et' veya 'Genişlet')..."}
                 className="flex-1 bg-[#151720] border-transparent focus:bg-[#0F111A] focus:border-teal-500/50 focus:ring-0 rounded-xl py-3 px-4 min-h-[50px] max-h-32 resize-none text-sm text-slate-200 placeholder:text-slate-600"
               />
               <button 
                 onClick={handleSend}
                 disabled={isLoading || !input.trim()}
                 className="p-3 bg-teal-700 hover:bg-teal-600 text-white rounded-xl shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all"
               >
                 {isLoading ? <div className="animate-spin w-5 h-5 border-2 border-white/30 border-t-white rounded-full" /> : <Send size={20} />}
               </button>
            </div>
         </div>
      </div>

      {/* RIGHT: Live Draft Preview (55%) */}
      <div className="hidden lg:flex flex-1 flex-col bg-[#0B0C15] h-full relative">
          
          {/* Roadmap / Progress Bar */}
          <div className="h-16 bg-[#0B0C15] border-b border-white/10 px-6 flex items-center justify-between overflow-hidden">
              <div className="flex items-center gap-1 overflow-x-auto no-scrollbar">
                {SECTIONS.map((sec, idx) => (
                    <div key={idx} className="flex items-center shrink-0">
                        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-bold whitespace-nowrap transition-all
                            ${idx < currentStep 
                                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' 
                                : idx === currentStep 
                                    ? 'bg-teal-500/10 border-teal-500/30 text-teal-400 ring-1 ring-teal-500/20' 
                                    : 'bg-transparent border-transparent text-slate-600'}
                        `}>
                            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px]
                                ${idx < currentStep ? 'bg-emerald-500/20' : idx === currentStep ? 'bg-teal-500/20' : 'bg-slate-800 text-slate-500'}
                            `}>
                                {idx < currentStep ? <Check size={12} /> : idx + 1}
                            </span>
                            {sec.split(' ')[0]}
                        </div>
                        {idx < SECTIONS.length - 1 && <ChevronRight size={16} className="text-slate-700 mx-1" />}
                    </div>
                ))}
              </div>

              {fullDraft && (
                  <button 
                    onClick={handleExportPDF}
                    disabled={isExporting}
                    className="flex items-center gap-2 px-3 py-1.5 bg-teal-700 text-white rounded-lg shadow-sm hover:bg-teal-600 text-xs font-bold transition-all disabled:opacity-70"
                  >
                      {isExporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                      PDF Olarak İndir
                  </button>
              )}
          </div>

          {/* Document Preview */}
          <div className="flex-1 p-8 overflow-y-auto bg-[#0B0C15]">
              <div className="max-w-3xl mx-auto bg-[#151720] min-h-[800px] shadow-2xl shadow-black/50 border border-white/5 p-12 relative animate-in fade-in duration-500">
                  
                  {fullDraft ? (
                      <div className="prose prose-invert max-w-none prose-headings:font-serif prose-headings:font-bold prose-headings:text-slate-200 prose-p:font-serif prose-p:leading-loose prose-p:text-lg prose-p:text-slate-300">
                          {researchTopic && (
                              <div className="text-center mb-12 border-b border-white/10 pb-8">
                                  <h1 className="text-3xl font-bold mb-4 capitalize text-white">{researchTopic}</h1>
                                  <p className="text-slate-500 italic">Taslak Sürüm 1.0</p>
                              </div>
                          )}
                          <div className="whitespace-pre-wrap">
                            {fullDraft}
                          </div>
                          
                          {/* Pending Section Indicator */}
                          <div className="mt-8 p-4 bg-[#0B0C15] border border-dashed border-slate-700 rounded-lg flex items-center justify-center text-slate-500 text-sm italic">
                              {SECTIONS[currentStep]} bölümü yazılıyor...
                          </div>
                      </div>
                  ) : (
                      <div className="flex flex-col items-center justify-center h-full text-slate-600 space-y-4">
                          <FileText size={64} className="opacity-30" />
                          <p className="font-medium text-slate-500">Makale taslağınız burada oluşacak.</p>
                          <p className="text-xs max-w-xs text-center text-slate-600">Sol panelden konuyu belirleyin ve Co-Author asistanının bölümleri yazmasını bekleyin.</p>
                      </div>
                  )}
              </div>
          </div>
      </div>

    </div>
  );
};

export default CoAuthorStudio;
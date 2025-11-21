
import React, { useState } from 'react';
import { Briefcase, PenTool, Sparkles, FileCheck, Mail, Beaker, ShieldAlert, Loader2, ChevronRight, ArrowLeft, Copy, Check } from 'lucide-react';
import { GoogleGenAI } from "@google/genai";

interface Tool {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  promptTemplate: string;
  inputLabel: string;
  placeholder: string;
}

// Tool Configurations
const TOOLS: Tool[] = [
  {
    id: 'abstract',
    title: 'Abstract Architect',
    description: 'Makaleniz için yapılandırılmış (structured) ve profesyonel bir özet oluşturur.',
    icon: <PenTool size={24} />,
    color: 'from-teal-500 to-emerald-600',
    inputLabel: 'Makale Metni veya Taslağı:',
    placeholder: 'Makalenizin tamamını veya giriş/metod/sonuç kısımlarını buraya yapıştırın...',
    promptTemplate: `Sen uzman bir akademik editörsün. Aşağıdaki metni kullanarak uluslararası standartlarda (örneğin Elsevier veya Springer formatında) bir 'Özet' (Abstract) yaz. 
    Özet şu yapıda olmalı: Amaç, Metodoloji, Bulgular, Sonuç ve Özgün Değer. Dil: Akademik Türkçe. Maksimum 250 kelime.`
  },
  {
    id: 'title',
    title: 'Title Forge',
    description: 'Çalışmanız için çarpıcı, akademik ve SEO uyumlu 5 alternatif başlık üretir.',
    icon: <Sparkles size={24} />,
    color: 'from-indigo-500 to-blue-600',
    inputLabel: 'Makale Özeti (Abstract):',
    placeholder: 'Özetinizi buraya yapıştırın...',
    promptTemplate: `Aşağıdaki özeti analiz et ve bu çalışma için 5 farklı akademik başlık önerisi sun.
    1. Tanımlayıcı Başlık
    2. Sonuç Odaklı Başlık
    3. Soru Biçiminde Başlık
    4. İki Parçalı (Colon) Başlık
    5. Yaratıcı/Metaforik Başlık`
  },
  {
    id: 'reference',
    title: 'Reference Fixer',
    description: 'Bozuk, eksik veya karışık referansları tam APA 7 formatına dönüştürür.',
    icon: <FileCheck size={24} />,
    color: 'from-purple-500 to-pink-600',
    inputLabel: 'Bozuk Referanslar:',
    placeholder: 'Örn: Yilmaz 2020, AI in Education, Journal of Ed Tech...',
    promptTemplate: `Aşağıdaki referans bilgilerini al ve bunları hatasız bir APA 7 kaynakça girdisine dönüştür. Eksik bilgi varsa [EKSİK: Bilgi Türü] şeklinde belirt. Çıktıyı sadece düzeltilmiş referans listesi olarak ver.`
  },
  {
    id: 'email',
    title: 'Email Composer',
    description: 'Editörlere, danışmanlara veya jüriye gönderilecek resmi e-postalar tasarlar.',
    icon: <Mail size={24} />,
    color: 'from-amber-500 to-orange-600',
    inputLabel: 'E-posta Konusu ve Detaylar:',
    placeholder: 'Örn: Dergi editörüne revizyon süresi için ek süre talebi...',
    promptTemplate: `Aşağıdaki durum için son derece profesyonel, nazik ve akademik bir e-posta taslağı yaz. Konu satırını da ekle.
    Durum: `
  },
  {
    id: 'hypothesis',
    title: 'Hypothesis Generator',
    description: 'Araştırma sorunuzdan H0 (Sıfır) ve H1 (Alternatif) hipotez setleri oluşturur.',
    icon: <Beaker size={24} />,
    color: 'from-rose-500 to-red-600',
    inputLabel: 'Araştırma Sorusu:',
    placeholder: 'Örn: Uzaktan çalışmanın çalışan tükenmişliği üzerindeki etkisi nedir?',
    promptTemplate: `Aşağıdaki araştırma sorusu için bilimsel hipotezler (H0 ve H1) oluştur. Ayrıca bu hipotezleri test etmek için uygun olabilecek istatistiksel analiz yöntemini de (örn: t-test, regresyon) öner.`
  },
  {
    id: 'reviewer',
    title: 'Review Responder',
    description: 'Hakemlerden gelen sert eleştirilere, akademik nezaket çerçevesinde yanıt mektubu hazırlar.',
    icon: <ShieldAlert size={24} />,
    color: 'from-cyan-500 to-blue-500',
    inputLabel: 'Hakem Eleştirisi:',
    placeholder: 'Örn: Metodoloji kısmı çok zayıf ve örneklem yetersiz...',
    promptTemplate: `Aşağıdaki hakem eleştirisine, 'Response to Reviewers' mektubunda kullanılmak üzere profesyonel, yapıcı ve nazik bir yanıt paragrafı yaz. Eleştiriyi kabul et ve (hayali bir) düzeltme yapıldığını belirterek teşekkür et.`
  }
];

const AcademicToolsView: React.FC = () => {
  const [activeTool, setActiveTool] = useState<Tool | null>(null);
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleToolClick = (tool: Tool) => {
    setActiveTool(tool);
    setInputText("");
    setResult("");
    setCopied(false);
  };

  const handleBack = () => {
    setActiveTool(null);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(result);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleGenerate = async () => {
    if (!inputText.trim() || !activeTool) return;
    
    setIsLoading(true);
    setResult("");

    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
        const prompt = `${activeTool.promptTemplate}\n\nGİRDİ:\n"""${inputText}"""`;
        
        const response = await ai.models.generateContent({
            model: 'gemini-3-pro-preview',
            contents: [{ role: 'user', parts: [{ text: prompt }] }],
        });

        setResult(response.text || "Sonuç üretilemedi.");
    } catch (error: any) {
        setResult("Hata oluştu: " + error.message);
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#0B0C15] font-sans overflow-hidden text-slate-200 relative">
       
       {/* Header */}
       <div className="px-8 py-6 border-b border-white/10 bg-[#0B0C15] flex items-center gap-3 z-10 shrink-0">
           <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
               <Briefcase size={24} />
           </div>
           <div>
               <h1 className="text-xl font-bold text-white">Academic Tools™</h1>
               <p className="text-xs text-slate-400">Araştırmacıların günlük görevleri için 6 özel yardımcı araç.</p>
           </div>
       </div>

       {/* Main Content */}
       <div className="flex-1 overflow-y-auto p-8 relative">
           
           {/* Tool Grid (Visible when no tool selected) */}
           {!activeTool && (
               <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
                   {TOOLS.map((tool) => (
                       <button
                           key={tool.id}
                           onClick={() => handleToolClick(tool)}
                           className="group relative bg-[#0F111A] border border-white/10 rounded-2xl p-6 text-left hover:border-white/20 hover:bg-[#151720] transition-all duration-300 overflow-hidden shadow-sm hover:shadow-xl"
                       >
                           {/* Gradient Blob */}
                           <div className={`absolute top-0 right-0 w-32 h-32 bg-gradient-to-br ${tool.color} opacity-10 blur-2xl rounded-full group-hover:opacity-20 transition-opacity`}></div>
                           
                           <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${tool.color} flex items-center justify-center text-white shadow-lg mb-4 group-hover:scale-110 transition-transform duration-300`}>
                               {tool.icon}
                           </div>
                           
                           <h3 className="text-lg font-bold text-white mb-2 group-hover:text-blue-200 transition-colors">{tool.title}</h3>
                           <p className="text-sm text-slate-400 leading-relaxed mb-4 h-10 line-clamp-2">{tool.description}</p>
                           
                           <div className="flex items-center text-xs font-bold text-slate-500 group-hover:text-white transition-colors mt-auto">
                               ARACI AÇ <ChevronRight size={14} className="ml-1" />
                           </div>
                       </button>
                   ))}
               </div>
           )}

           {/* Active Tool Interface */}
           {activeTool && (
               <div className="max-w-4xl mx-auto animate-in zoom-in-95 duration-300 flex flex-col h-full">
                   <button onClick={handleBack} className="flex items-center gap-2 text-slate-400 hover:text-white mb-6 transition-colors w-fit">
                       <ArrowLeft size={18} />
                       Geri Dön
                   </button>

                   <div className="bg-[#0F111A] border border-white/10 rounded-2xl flex-1 flex flex-col overflow-hidden shadow-2xl">
                       {/* Tool Header */}
                       <div className="p-6 border-b border-white/10 bg-[#13151C] flex items-center gap-4">
                           <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${activeTool.color} flex items-center justify-center text-white shadow-md`}>
                               {activeTool.icon}
                           </div>
                           <div>
                               <h2 className="text-lg font-bold text-white">{activeTool.title}</h2>
                               <p className="text-xs text-slate-400">{activeTool.description}</p>
                           </div>
                       </div>

                       {/* Split View: Input & Output */}
                       <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
                           {/* Input Area */}
                           <div className="flex-1 p-6 border-b lg:border-b-0 lg:border-r border-white/10 flex flex-col bg-[#0F111A]">
                               <label className="text-xs font-bold text-slate-400 uppercase mb-3 block ml-1">{activeTool.inputLabel}</label>
                               <textarea 
                                   value={inputText}
                                   onChange={(e) => setInputText(e.target.value)}
                                   placeholder={activeTool.placeholder}
                                   className="flex-1 w-full bg-[#0B0C15] border border-white/10 rounded-xl p-4 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500/50 resize-none transition-all"
                               />
                               <div className="mt-4 flex justify-end">
                                   <button 
                                       onClick={handleGenerate}
                                       disabled={isLoading || !inputText.trim()}
                                       className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-white shadow-lg transition-all active:scale-95
                                           ${isLoading ? 'bg-slate-700 cursor-wait' : `bg-gradient-to-r ${activeTool.color} hover:opacity-90`}
                                       `}
                                   >
                                       {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Sparkles size={18} />}
                                       {isLoading ? 'Oluşturuluyor...' : 'Oluştur'}
                                   </button>
                               </div>
                           </div>

                           {/* Output Area */}
                           <div className="flex-1 p-6 bg-[#151720] flex flex-col relative">
                               <div className="flex items-center justify-between mb-3">
                                   <label className="text-xs font-bold text-slate-400 uppercase ml-1">Sonuç:</label>
                                   {result && (
                                       <button onClick={handleCopy} className="flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-white transition-colors px-2 py-1 bg-white/5 rounded-md hover:bg-white/10">
                                           {copied ? <Check size={12} className="text-green-400"/> : <Copy size={12} />}
                                           {copied ? 'Kopyalandı' : 'Kopyala'}
                                       </button>
                                   )}
                               </div>
                               
                               <div className="flex-1 bg-[#0B0C15] border border-white/5 rounded-xl p-6 overflow-y-auto text-sm leading-relaxed text-slate-300 font-mono shadow-inner">
                                   {isLoading ? (
                                       <div className="h-full flex flex-col items-center justify-center text-slate-600 gap-3">
                                           <Loader2 size={32} className="animate-spin text-blue-500/50" />
                                           <p className="text-xs animate-pulse">Yapay zeka düşünüyor...</p>
                                       </div>
                                   ) : result ? (
                                       <div className="whitespace-pre-wrap">{result}</div>
                                   ) : (
                                       <div className="h-full flex flex-col items-center justify-center text-slate-600 opacity-50">
                                           <Sparkles size={48} className="mb-4" />
                                           <p>Girdi sağlayın ve 'Oluştur' butonuna basın.</p>
                                       </div>
                                   )}
                               </div>
                           </div>
                       </div>
                   </div>
               </div>
           )}
       </div>
    </div>
  );
};

export default AcademicToolsView;

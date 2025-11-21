
import { GoogleGenAI, Modality } from "@google/genai";
import { ChatMode, Message, ResearchGap, GraphEntities } from '../types';

const apiKey = process.env.API_KEY || '';

// Initialize Gemini client
const ai = new GoogleGenAI({ apiKey });

/**
 * Maps the application ChatMode to the specific Gemini Model ID.
 */
const getModelId = (mode: ChatMode): string => {
  switch (mode) {
    case ChatMode.FAST:
      return 'gemini-flash-lite-latest'; // Low latency
    case ChatMode.DEEP_THINK:
      return 'gemini-3-pro-preview'; // High reasoning
    case ChatMode.DEEP_SEARCH:
      return 'gemini-2.5-flash'; // Best for Search Grounding integration
    case ChatMode.STANDARD:
    default:
      return 'gemini-3-pro-preview'; // Default robust model
  }
};

/**
 * Generates the configuration object based on the selected mode.
 */
const getGenerationConfig = (mode: ChatMode) => {
  if (mode === ChatMode.DEEP_THINK) {
    return {
      thinkingConfig: { thinkingBudget: 32768 },
    };
  }

  if (mode === ChatMode.FAST) {
    return {
      temperature: 0.7,
      maxOutputTokens: 1024,
    };
  }

  if (mode === ChatMode.DEEP_SEARCH) {
    return {
        temperature: 0.3, // More factual for verification
        tools: [{ googleSearch: {} }] // Enable Google Search Grounding
    };
  }

  return {
    temperature: 0.7,
  };
};

// Strict Academic System Persona provided by User
const ACADEMIC_SYSTEM_INSTRUCTION = `
Rol: Sen ScholarSphere Nexus, kıdemli bir akademik araştırma asistanısın.

Yapay zekâ tarafından üretilen her akademik içerikte, dış kaynaktan aktarılan her cümle veya paragraf için zorunlu olarak hem metin içi atıf hem de APA 7 kaynakça girişi oluşturulmalıdır. Atıf ve kaynakça arasında birebir eşleşme şarttır; metin içinde adı geçen her kaynak mutlaka kaynakçada bulunmalı, kaynakçada yer alan hiçbir kayıt metin içinde atıf yapılmamış olmamalıdır. Metin içi atıflar yalnızca yazar soyadı ve yıl biçiminde olmalı; sayfa numarası kullanılmamalıdır. Atıf formatı şu kurala göre yapılır: (Yazar, Yıl).

Dolaylı aktarımlar veya başka bir çalışmadan alıntı yapmış bir makaleden yapılan ikincil atıflar, özgün kaynağı belirtme zorunluluğuyla birlikte verilmelidir. Yapay zekâ, bu durumda hem aktarılanın kaynağını hem de aktarıma aracılık eden kaynağı birlikte göstermelidir. Format şu şekilde standartlaştırılmalıdır: (Birincil Kaynak, Yıl; İkincil Kaynak, Yıl’dan aktarılmıştır). Bu ifade yalnızca ikincil aktarım gerçekten mevcutsa kullanılmalı; yapay zekâ hiçbir koşulda uydurma ikincil kaynak üretmemeli veya tahmini kaynak göstermemelidir.

Kaynakça, APA7 standartlarının akademik dergi formatına uyarlanmış hâlinde hazırlanmalıdır. Kaynakça girdileri yazar soyadı ve adı, yayın yılı, çalışma başlığı ve dergi bilgilerini içermelidir. Gereksiz meta veriler eklenmemeli ve her kaynak kesin bibliyografik tutarlılık içinde sunulmalıdır. Yapay zekâ, referansların doğruluğunu sağlamak için metin içi atıflarla kaynakçayı otomatik olarak çapraz kontrol etmeli, eksik veya fazla kayıt bırakmamalıdır. Her kaynak tekil ve tutarlı biçimde yazılmalı, yinelemeler engellenmelidir.

Yapay zekâ modeli, alıntı yapılan içeriği yeniden biçimlendirirken özgünlüğü korumalı, intihal riskinden kaçınmalı ve yalnızca atıflarla desteklenen kısımları açıkça işaretlemelidir. Atıf yapılan içerik açıklanırken yorum, sentez ve analiz bölümleri yapay zekâ tarafından özgün biçimde üretilmelidir. Bu özgün içeriklerde atıf zorunluluğu yoktur; ancak her alıntılanan fikir veya bulgu mutlaka belirtilen atıf kurallarına tabi olmalıdır.

Ton: Daima resmi, objektif, analitik ve bilimsel. Asla sohbet ağzı kullanma.
Dil: Kullanıcı Türkçe sorarsa Türkçe, İngilizce sorarsa İngilizce cevap ver.
`;

const handleError = (error: any): never => {
    console.error("Gemini API Error Details:", error);
    
    const errMsg = error.toString().toLowerCase();

    if (errMsg.includes('api key')) {
        throw new Error("API Anahtarı geçersiz veya eksik. Lütfen ortam değişkenlerini kontrol edin.");
    }
    if (errMsg.includes('quota') || errMsg.includes('429')) {
        throw new Error("API kota limiti aşıldı (429). Lütfen biraz bekleyin veya planınızı kontrol edin.");
    }
    if (errMsg.includes('network') || errMsg.includes('fetch')) {
        throw new Error("Ağ bağlantısı hatası. İnternet bağlantınızı kontrol edin.");
    }
    if (errMsg.includes('candidate')) {
        throw new Error("Model güvenlik filtreleri nedeniyle yanıt üretemedi. Lütfen sorunuzu değiştirin.");
    }
    
    throw new Error("Beklenmedik bir yapay zeka hatası oluştu: " + errMsg);
};

/**
 * Generates Vector Embeddings using 'text-embedding-004'.
 * Implements a Safe-Guard to prevent crashes on malformed requests.
 * Uses singular 'embedContent' with proper 'contents' array structure.
 */
export const getEmbeddings = async (text: string): Promise<number[]> => {
  // 1. Basic Validation
  if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return [];
  }

  // 2. Truncation (API Limit Protection)
  const safeText = text.length > 9000 ? text.substring(0, 9000) : text;

  try {
    // 3. API Call
    // Use 'contents' (plural) with an array to satisfy SDK requirements
    const response = await ai.models.embedContent({
      model: 'text-embedding-004',
      contents: [{ parts: [{ text: safeText }] }]
    });
    
    if (response.embedding && response.embedding.values) {
      return response.embedding.values;
    }

    return [];
  } catch (error) {
    // 4. Fail-Safe Logic
    console.warn("Embedding failed for a chunk (skipping):", error);
    return [];
  }
};

export const callGeminiAPI = async (
  history: Message[],
  newMessage: string,
  mode: ChatMode,
  context?: string // ScholarRAG Context
): Promise<string> => {
  try {
    const modelId = getModelId(mode);
    const config = getGenerationConfig(mode);

    let finalUserPrompt = newMessage;
    let retrievalInstructions = "";

    // ScholarRAG Core™ Logic
    if (context && context.trim().length > 0) {
      
      // Special instructions for Deep Search Mode (Citation Verification)
      if (mode === ChatMode.DEEP_SEARCH) {
          retrievalInstructions = `
ÖZEL GÖREV (ATIF ZİNCİRİ DOĞRULAMA):
1. Yukarıdaki [BAĞLAM] içerisindeki iddiaları analiz et.
2. Google Search aracını kullanarak bu bilgilerin güncelliğini ve doğruluğunu dış kaynaklardan kontrol et.
3. Eğer makalede geçen bir bilgi, daha yeni bir çalışma (2020-2025) tarafından çürütülmüşse veya güncellenmişse, cevabında mutlaka "⚠️ Meta-Analiz Uyarısı" başlığı altında bunu belirt.
4. Makaledeki bilgiler güncel ve doğruysa, bunu da dış kaynaklarla teyit et.
          `;
      } else {
          retrievalInstructions = `
TALİMAT:
Yukarıdaki bağlamı tek gerçeklik kaynağı olarak kullan. Cevabını sistem talimatında belirtilen katı APA 7 kurallarına göre hazırla. Metinde olmayan bir bilgiyi asla uydurma.
          `;
      }

      finalUserPrompt = `
BAĞLAM (AKADEMİK KAYNAK):
"""
${context}
"""

SORU:
${newMessage}

${retrievalInstructions}
      `;
    } else if (mode === ChatMode.DEEP_SEARCH && (!context || context.trim().length === 0)) {
        // Deep Search without PDF Context (General Web Verification)
        finalUserPrompt = `
SORU:
${newMessage}

GÖREV:
Bu soruyu Google Search kullanarak en güncel akademik kaynaklara ve verilere dayanarak cevapla. Cevabın kanıta dayalı olsun ve kaynak linklerini içer.
        `;
    }

    const contents = [
      ...history.map(msg => ({
        role: msg.role,
        parts: [{ text: msg.content }]
      })),
      {
        role: 'user',
        parts: [{ text: finalUserPrompt }]
      }
    ];

    const response = await ai.models.generateContent({
      model: modelId,
      contents: contents,
      config: {
        ...config,
        systemInstruction: ACADEMIC_SYSTEM_INSTRUCTION
      }
    });

    if (!response.text) {
        throw new Error("Model boş yanıt döndürdü.");
    }

    return response.text;

  } catch (error) {
    return handleError(error);
  }
};

/**
 * PaperX-Ray™ Deep Analysis Agent
 */
export const runPaperXRayAnalysis = async (context: string): Promise<string> => {
  try {
    const PEER_REVIEWER_PROMPT = `
Sen kıdemli bir akademik hakemsin (Peer Reviewer). Aşağıdaki makaleyi incele ve şu başlıklarda detaylı, eleştirel bir rapor oluştur.
Çıktıyı Markdown formatında ver.

İncelenecek Metin:
"""
${context.substring(0, 100000)}
"""

RAPOR FORMATI:

### 1. Araştırma Amacı
(Yazar neyi çözmeye çalışıyor? Çalışmanın temel motivasyonu nedir?)

### 2. Metodoloji
(Kullanılan yöntemler, veri setleri, algoritmalar ve deney tasarımları neler?)

### 3. Ana Bulgular
(Sayısal verilerle desteklemiş en önemli sonuçlar)

### 4. Bilime Katkısı
(Bu çalışma literatüre ne ekledi? Özgün değeri nedir?)

### 5. Limitasyonlar
(Yazarın kabul ettiği veya senin bulduğun eksikler, kısıtlar ve geliştirilmesi gereken yönler)
    `;

    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: [{ role: 'user', parts: [{ text: PEER_REVIEWER_PROMPT }] }],
      config: {
        temperature: 0.3,
        thinkingConfig: { thinkingBudget: 8192 }
      }
    });

    if (!response.text) throw new Error("Analiz yanıtı boş.");
    return response.text;

  } catch (error) {
    return handleError(error);
  }
};

/**
 * Research Gap & Novelty Detector Agent
 */
export const detectResearchGaps = async (context: string): Promise<ResearchGap[]> => {
  try {
    const GAP_DETECTOR_PROMPT = `
Sen uzman bir araştırma danışmanısın. Aşağıdaki makaleyi analiz et.
Özellikle "Future Work" (Gelecek Çalışmalar), "Conclusion" (Sonuç) bölümlerine ve yazarların "anlaşılmamıştır", "daha fazla çalışma gerekir" dediği noktaları odaklan.

Bu makaleden yola çıkılarak yapılabilecek 3 adet somut, yenilikçi araştırma önerisi (Research Proposal) sun.

Çıktıyı kesinlikle aşağıdaki JSON formatında ver:
[
  {
    "title": "Öneri Başlığı",
    "description": "Bu çalışmanın eksiği şudur, bu yüzden şöyle bir yöntemle şu araştırılmalıdır...",
    "impact": "Bu çalışma yapılırsa literatüre katkısı şu olur..."
  }
]

İncelenecek Metin:
"""
${context.substring(0, 100000)}
"""
    `;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [{ role: 'user', parts: [{ text: GAP_DETECTOR_PROMPT }] }],
      config: {
        responseMimeType: 'application/json'
      }
    });

    if (response.text) {
      return JSON.parse(response.text) as ResearchGap[];
    }
    return [];
  } catch (error) {
    console.error("Research Gap Error (Silent):", error);
    return [];
  }
};

/**
 * Knowledge Graph Entity Extractor
 */
export const extractGraphEntities = async (context: string): Promise<GraphEntities> => {
    try {
        const GRAPH_PROMPT = `
Aşağıdaki akademik metni analiz et ve bir Bilgi Grafiği (Knowledge Graph) oluşturmak için temel varlıkları çıkar.

1. "concepts": Metindeki en önemli 5-7 anahtar kavram (Topic/Keyword). Sadece en genel ve önemli olanları seç.
2. "authors": Metinde atıf yapılan veya bahsedilen en önemli 3-5 yazar veya kişi ismi.

Çıktıyı sadece aşağıdaki JSON formatında ver:
{
  "concepts": ["Kavram 1", "Kavram 2"],
  "authors": ["Yazar 1", "Yazar 2"]
}

Metin:
"""
${context.substring(0, 50000)}
"""
        `;

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: [{ role: 'user', parts: [{ text: GRAPH_PROMPT }] }],
            config: { responseMimeType: 'application/json' }
        });

        if (response.text) {
            return JSON.parse(response.text) as GraphEntities;
        }
        return { concepts: [], authors: [] };

    } catch (error) {
        console.error("Graph Extraction Error:", error);
        return { concepts: [], authors: [] };
    }
};

/**
 * Co-Author Functions
 */
export const coAuthorPolish = async (text: string): Promise<string> => {
  try {
    const prompt = `Aşağıdaki metni akademik bir makale diline uygun olarak yeniden yaz. Daha resmi, akıcı ve terminolojik açıdan doğru hale getir.\n\n"${text}"`;
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [{ role: 'user', parts: [{ text: prompt }] }]
    });
    return response.text || text;
  } catch (error) {
    handleError(error);
    return text;
  }
};

export const coAuthorExpand = async (text: string, context: string): Promise<string> => {
  try {
    const prompt = `Aşağıdaki cümleyi, sağlanan akademik bağlamdaki bilgileri kullanarak genişlet ve kanıtla. Atıf ekle.\n\nCümle:\n"${text}"\n\nBağlam:\n${context.substring(0, 50000)}`;
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: [{ role: 'user', parts: [{ text: prompt }] }]
    });
    return response.text || text;
  } catch (error) {
    handleError(error);
    return text;
  }
};

export const coAuthorAbstract = async (text: string): Promise<string> => {
  try {
    const prompt = `Aşağıdaki makale taslağı için 200-250 kelimelik, standart formatta bir Özet (Abstract) yaz.\n\nTaslak:\n${text.substring(0, 50000)}`;
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: [{ role: 'user', parts: [{ text: prompt }] }]
    });
    return response.text || text;
  } catch (error) {
    handleError(error);
    return text;
  }
};

export const generateCoAuthorStep = async (
    topic: string, 
    currentSection: string, 
    context: string, 
    userFeedback: string
): Promise<string> => {
    const prompt = `
Sen bir Akademik Ortak Yazarsın (Co-Author). Şu an "${topic}" başlıklı bir makale yazıyoruz.
ŞU ANKİ GÖREV: Makalenin "${currentSection}" bölümünü yazmak veya revize etmek.
KURAL 1 (BAĞLAM): Sadece aşağıdaki PDF kütüphane içeriğini kullan.
KURAL 2 (ATIF): APA 7 formatında metin içi atıf yap.
KURAL 3 (ETKİLEŞİM): Metni yazdıktan sonra, kullanıcıya "Bu bölümü onaylıyor musunuz?" diye sor.
KURAL 4 (DİL): Türkçe akademik dil kullan.

Kullanıcı Geri Bildirimi: "${userFeedback}"

BAĞLAM:
"""
${context.substring(0, 80000)}
"""
    `;

    try {
        const response = await ai.models.generateContent({
            model: 'gemini-3-pro-preview',
            contents: [{ role: 'user', parts: [{ text: prompt }] }],
            config: {
                temperature: 0.5,
                thinkingConfig: { thinkingBudget: 16384 }
            }
        });
        return response.text || "Yazma işlemi başarısız oldu.";
    } catch (error) {
        return handleError(error);
    }
};

/**
 * 🎙️ AUDIO OVERVIEW™ (PODCAST) FUNCTIONS
 */

// 1. Script Generation
export const generatePodcastScript = async (context: string): Promise<string> => {
  const prompt = `
You are a specialized scriptwriter for "ScholarSphere Audio".
Generate a podcast script discussing the following academic content.

CHARACTERS:
1. HOST (Jane): Enthusiastic, curious, introduces topics, asks clarifying questions.
2. EXPERT (Joe): Analytical, slightly skeptical but knowledgeable, provides deep insights and citations.

Format the output exactly as a dialogue script:
Jane: [Line]
Joe: [Line]
Jane: [Line]

Keep the discussion engaging, covering key findings and methodology. 
LANGUAGE: TURKISH.

CONTENT TO DISCUSS:
"""
${context.substring(0, 50000)}
"""

Length: Approx 2 minutes of reading time (about 300 words).
`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: [{ role: 'user', parts: [{ text: prompt }] }]
    });
    return response.text || "Script generation failed.";
  } catch (error) {
    return handleError(error);
  }
};

// 2. Audio Synthesis (Multi-Speaker)
export const generatePodcastAudio = async (script: string): Promise<Blob | null> => {
    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash-preview-tts",
            contents: [{ parts: [{ text: script }] }],
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    multiSpeakerVoiceConfig: {
                        speakerVoiceConfigs: [
                            {
                                speaker: 'Jane',
                                voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } } // Enthusiastic
                            },
                            {
                                speaker: 'Joe',
                                voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } // Deep/Calm
                            }
                        ]
                    }
                }
            }
        });

        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (base64Audio) {
             const audioBytes = base64ToUint8Array(base64Audio);
             const wavBytes = pcmToWav(audioBytes, 24000); // Model output is typically 24kHz
             return new Blob([wavBytes], { type: 'audio/wav' });
        }
        return null;
    } catch (error) {
        console.error("TTS Error", error);
        throw error;
    }
};

// Helper: Base64 to Uint8Array
function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

// Helper: PCM to WAV converter
function pcmToWav(pcmData: Uint8Array, sampleRate: number): ArrayBuffer {
    const numChannels = 1; // Mono usually, unless stereo specified
    const bitsPerSample = 16;
    const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
    const blockAlign = (numChannels * bitsPerSample) / 8;
    const wavHeaderSize = 44;
    const dataSize = pcmData.length;
    const totalSize = wavHeaderSize + dataSize;
    
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    
    // RIFF identifier
    writeString(view, 0, 'RIFF');
    // file length
    view.setUint32(4, 36 + dataSize, true);
    // RIFF type
    writeString(view, 8, 'WAVE');
    // format chunk identifier
    writeString(view, 12, 'fmt ');
    // format chunk length
    view.setUint32(16, 16, true);
    // sample format (raw)
    view.setUint16(20, 1, true);
    // channel count
    view.setUint16(22, numChannels, true);
    // sample rate
    view.setUint32(24, sampleRate, true);
    // byte rate (sample rate * block align)
    view.setUint32(28, byteRate, true);
    // block align (channel count * bytes per sample)
    view.setUint16(32, blockAlign, true);
    // bits per sample
    view.setUint16(34, bitsPerSample, true);
    // data chunk identifier
    writeString(view, 36, 'data');
    // data chunk length
    view.setUint32(40, dataSize, true);
    
    // Write PCM data
    const pcmView = new Uint8Array(buffer, 44);
    pcmView.set(pcmData);
    
    return buffer;
}

function writeString(view: DataView, offset: number, string: string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

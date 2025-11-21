
import React, { useState, useEffect, useRef } from 'react';
import { Share2, Loader2, RefreshCw, Info, MousePointerClick } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';
import { PdfFile } from '../types';
import { extractGraphEntities } from '../services/geminiService';

interface KnowledgeGraphViewProps {
  files: PdfFile[];
  onUpdateFile: (file: PdfFile) => void;
  addToast: (msg: string, type: any) => void;
}

const KnowledgeGraphView: React.FC<KnowledgeGraphViewProps> = ({ files, onUpdateFile, addToast }) => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ w: 800, h: 600 });

  // Responsive Graph sizing
  useEffect(() => {
    const updateDims = () => {
        if (containerRef.current) {
            setDimensions({
                w: containerRef.current.clientWidth,
                h: containerRef.current.clientHeight
            });
        }
    };
    window.addEventListener('resize', updateDims);
    updateDims();
    return () => window.removeEventListener('resize', updateDims);
  }, []);

  // Build Graph Data
  useEffect(() => {
    const nodes: any[] = [];
    const links: any[] = [];
    const nodeIds = new Set();

    files.forEach(file => {
        // 1. Document Nodes
        if (!nodeIds.has(file.id)) {
            nodes.push({
                id: file.id,
                name: file.title,
                group: 'document',
                val: 20
            });
            nodeIds.add(file.id);
        }

        // 2. Entity Nodes (Concepts & Authors)
        if (file.graphData) {
            // Concepts
            file.graphData.concepts.forEach(concept => {
                const cId = `concept-${concept}`;
                if (!nodeIds.has(cId)) {
                    nodes.push({ id: cId, name: concept, group: 'concept', val: 8 });
                    nodeIds.add(cId);
                }
                links.push({ source: file.id, target: cId });
            });

            // Authors
            file.graphData.authors.forEach(author => {
                const aId = `author-${author}`;
                if (!nodeIds.has(aId)) {
                    nodes.push({ id: aId, name: author, group: 'author', val: 5 });
                    nodeIds.add(aId);
                }
                links.push({ source: file.id, target: aId });
            });
        }
    });

    setGraphData({ nodes, links });
  }, [files]);

  const handleGenerateGraph = async () => {
      const unprocessedFiles = files.filter(f => !f.graphData);
      if (unprocessedFiles.length === 0) {
          addToast("Tüm dosyalar zaten ağa eklenmiş.", "info");
          return;
      }

      setIsProcessing(true);
      let count = 0;
      
      for (const file of unprocessedFiles) {
          try {
              // Use chunked context to avoid token limits, analyze first 30k chars
              const context = file.text?.substring(0, 30000) || "";
              const entities = await extractGraphEntities(context);
              
              const updatedFile = { ...file, graphData: entities };
              onUpdateFile(updatedFile);
              
              count++;
              setProgress(Math.round((count / unprocessedFiles.length) * 100));
          } catch (e) {
              console.error("Graph generation failed for file", file.title, e);
          }
      }
      
      setIsProcessing(false);
      setProgress(0);
      addToast("Bilgi Grafiği başarıyla güncellendi.", "success");
  };

  return (
    <div className="h-full flex flex-col bg-slate-950 font-sans overflow-hidden relative">
        
        {/* Header Overlay */}
        <div className="absolute top-0 left-0 w-full p-6 z-10 flex items-start justify-between pointer-events-none">
            <div className="pointer-events-auto bg-slate-900/80 backdrop-blur border border-slate-700 p-4 rounded-2xl shadow-xl">
                <div className="flex items-center gap-2 text-scholar-teal-400 mb-1">
                    <Share2 size={20} />
                    <h2 className="font-bold text-lg">Nexus Graph™</h2>
                </div>
                <p className="text-xs text-slate-400 max-w-xs">
                    Kütüphanenizdeki belgeler arasındaki anlamsal ilişkileri, ortak kavramları ve atıf ağlarını görselleştirin.
                </p>

                <div className="mt-4 flex items-center gap-3">
                    <div className="flex items-center gap-2 text-xs">
                        <span className="w-3 h-3 rounded-full bg-scholar-teal-500 block"></span>
                        <span className="text-slate-300">Belgeler</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                        <span className="w-3 h-3 rounded-full bg-purple-500 block"></span>
                        <span className="text-slate-300">Kavramlar</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                        <span className="w-3 h-3 rounded-full bg-amber-500 block"></span>
                        <span className="text-slate-300">Yazarlar</span>
                    </div>
                </div>
            </div>

            <div className="pointer-events-auto">
                <button 
                    onClick={handleGenerateGraph}
                    disabled={isProcessing}
                    className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl shadow-lg shadow-indigo-900/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isProcessing ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
                    {isProcessing ? `Analiz Ediliyor %${progress}` : 'Ağı Güncelle'}
                </button>
            </div>
        </div>

        {/* Graph Canvas */}
        <div ref={containerRef} className="flex-1 cursor-move">
            {files.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-600">
                    <Share2 size={64} className="mb-4 opacity-20" />
                    <p>Kütüphane boş. Grafik oluşturmak için belge yükleyin.</p>
                </div>
            ) : (
                <ForceGraph2D
                    width={dimensions.w}
                    height={dimensions.h}
                    graphData={graphData}
                    nodeLabel="name"
                    nodeColor={(node: any) => {
                        if (node.group === 'document') return '#2dd4bf'; // Teal
                        if (node.group === 'concept') return '#a855f7'; // Purple
                        if (node.group === 'author') return '#f59e0b'; // Amber
                        return '#ccc';
                    }}
                    nodeRelSize={6}
                    linkColor={() => 'rgba(255,255,255,0.1)'}
                    backgroundColor="#020617" // Slate 950
                    d3VelocityDecay={0.1}
                    cooldownTicks={100}
                    onNodeClick={(node: any) => {
                        if (node.group === 'document') {
                            addToast(`Seçildi: ${node.name}`, 'info');
                        }
                    }}
                />
            )}
        </div>
    </div>
  );
};

export default KnowledgeGraphView;

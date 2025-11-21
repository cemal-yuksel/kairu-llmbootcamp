
import React, { useState, useEffect } from 'react';
import { Settings, Trash2, Database, Shield, Server, HardDrive } from 'lucide-react';
import { getStorageEstimate, clearAllData } from '../services/db';

const SettingsView: React.FC = () => {
  const [storageUsage, setStorageUsage] = useState<string>('Hesaplanıyor...');
  const [quota, setQuota] = useState<string>('');

  useEffect(() => {
    calculateStorage();
  }, []);

  const calculateStorage = async () => {
    const estimate = await getStorageEstimate();
    if (estimate) {
      const usedMB = (estimate.usage / (1024 * 1024)).toFixed(2);
      const quotaGB = (estimate.quota / (1024 * 1024 * 1024)).toFixed(2);
      setStorageUsage(`${usedMB} MB`);
      setQuota(` / ${quotaGB} GB`);
    } else {
      // Fallback for browsers not supporting estimate (rare now)
      setStorageUsage('Bilinmiyor');
    }
  };

  const handleClearSession = async () => {
    if (window.confirm("DİKKAT: Tüm sohbet geçmişi, kütüphane dosyaları ve taslaklar kalıcı olarak silinecek. Onaylıyor musunuz?")) {
      await clearAllData();
      localStorage.clear(); // Also clear legacy localstorage
      window.location.reload();
    }
  };

  return (
    <div className="h-full bg-[#0B0C15] font-sans overflow-y-auto p-8 text-slate-200">
      <div className="max-w-3xl mx-auto space-y-8">
        
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Settings className="text-teal-500" />
            Sistem Ayarları
          </h1>
          <p className="text-slate-400 mt-1">
            Uygulama tercihleri, veri yönetimi ve oturum güvenliği.
          </p>
        </div>

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-[#0F111A] p-4 rounded-xl border border-white/10 shadow-sm">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 bg-green-500/10 text-green-400 rounded-lg">
                        <Server size={20} />
                    </div>
                    <span className="font-bold text-slate-200">Sunucu Durumu</span>
                </div>
                <div className="text-sm text-slate-400">
                    <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                    Serverless (Aktif)
                </div>
            </div>
            
            <div className="bg-[#0F111A] p-4 rounded-xl border border-white/10 shadow-sm">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 bg-blue-500/10 text-blue-400 rounded-lg">
                        <Shield size={20} />
                    </div>
                    <span className="font-bold text-slate-200">API Bağlantısı</span>
                </div>
                <div className="text-sm text-slate-400">
                    Gemini 3 & 1.5 Pro & Flash Lite
                </div>
            </div>

            <div className="bg-[#0F111A] p-4 rounded-xl border border-white/10 shadow-sm">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 bg-purple-500/10 text-purple-400 rounded-lg">
                        <HardDrive size={20} />
                    </div>
                    <span className="font-bold text-slate-200">Depolama (IDB)</span>
                </div>
                <div className="text-sm text-slate-400">
                    {storageUsage} <span className="text-slate-500">{quota}</span>
                </div>
            </div>
        </div>

        {/* Data Privacy Notice */}
        <div className="bg-blue-900/20 border border-blue-500/30 rounded-xl p-6 flex gap-4">
            <Database className="text-blue-400 shrink-0 mt-1" size={24} />
            <div>
                <h3 className="font-bold text-blue-300">Genişletilmiş Veritabanı Mimarisi</h3>
                <p className="text-sm text-blue-200/80 mt-1 leading-relaxed">
                    ScholarSphere Nexus, yüksek kapasiteli <strong>IndexedDB</strong> teknolojisini kullanır. 
                    Bu sayede yüzlerce PDF makalesini, kota sorununa takılmadan tarayıcınızda güvenle saklayabilirsiniz. 
                    Verileriniz yine cihazınızda kalır, sunucuya gitmez.
                </p>
            </div>
        </div>

        {/* Danger Zone */}
        <div className="border border-red-500/30 rounded-xl overflow-hidden bg-[#0F111A]">
            <div className="bg-red-900/20 px-6 py-3 border-b border-red-500/20 flex items-center gap-2">
                <span className="text-red-400 font-bold text-sm uppercase tracking-wide">Tehlikeli Bölge</span>
            </div>
            <div className="p-6 flex items-center justify-between">
                <div>
                    <h3 className="font-bold text-slate-200">Tüm Verileri Temizle ve Sıfırla</h3>
                    <p className="text-sm text-slate-400 mt-1">
                        Sohbet geçmişi, kütüphane dosyaları ve kaydedilen taslaklar kalıcı olarak silinecektir. Bu işlem geri alınamaz.
                    </p>
                </div>
                <button 
                    onClick={handleClearSession}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-bold shadow-sm transition-colors flex items-center gap-2"
                >
                    <Trash2 size={18} />
                    Verileri Sil
                </button>
            </div>
        </div>
        
        <div className="text-center pt-8 pb-4">
             <p className="text-xs text-slate-500">ScholarSphere Nexus™ 2.0 | Build v2.4.3</p>
        </div>

      </div>
    </div>
  );
};

export default SettingsView;
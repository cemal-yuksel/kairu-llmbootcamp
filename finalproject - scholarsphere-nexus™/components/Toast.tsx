
import React, { useEffect } from 'react';
import { CheckCircle, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';
import { ToastMessage, ToastType } from '../types';

interface ToastProps {
  toasts: ToastMessage[];
  removeToast: (id: string) => void;
}

const Toast: React.FC<ToastProps> = ({ toasts, removeToast }) => {
  return (
    <div className="fixed top-4 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={() => removeToast(toast.id)} />
      ))}
    </div>
  );
};

const ToastItem: React.FC<{ toast: ToastMessage; onRemove: () => void }> = ({ toast, onRemove }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onRemove();
    }, 5000); // Auto dismiss after 5 seconds
    return () => clearTimeout(timer);
  }, [onRemove]);

  const getIcon = () => {
    switch (toast.type) {
      case 'success': return <CheckCircle size={20} className="text-emerald-500" />;
      case 'error': return <AlertCircle size={20} className="text-red-500" />;
      case 'warning': return <AlertTriangle size={20} className="text-amber-500" />;
      default: return <Info size={20} className="text-blue-500" />;
    }
  };

  const getStyles = () => {
    switch (toast.type) {
      case 'success': return 'bg-white border-emerald-100 shadow-emerald-100';
      case 'error': return 'bg-white border-red-100 shadow-red-100';
      case 'warning': return 'bg-white border-amber-100 shadow-amber-100';
      default: return 'bg-white border-blue-100 shadow-blue-100';
    }
  };

  return (
    <div className={`
      pointer-events-auto flex items-start gap-3 p-4 rounded-xl border shadow-lg min-w-[300px] max-w-md 
      transform transition-all duration-300 animate-in slide-in-from-right fade-in
      ${getStyles()}
    `}>
      <div className="mt-0.5">{getIcon()}</div>
      <p className="flex-1 text-sm font-medium text-slate-700 leading-relaxed">
        {toast.message}
      </p>
      <button 
        onClick={onRemove}
        className="text-slate-400 hover:text-slate-600 transition-colors p-0.5"
      >
        <X size={16} />
      </button>
    </div>
  );
};

export default Toast;

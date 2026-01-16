'use client';

import { useEffect, useState } from 'react';

interface AIStatus {
  checking: boolean;
  connected: boolean;
  model?: string;
  error?: string;
}

export default function AIStatusIndicator() {
  const [status, setStatus] = useState<AIStatus>({
    checking: true,
    connected: false
  });

  useEffect(() => {
    checkAIStatus();
    
    // Recheck every 30 seconds
    const interval = setInterval(checkAIStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkAIStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/agent/health');
      const data = await response.json();
      
      setStatus({
        checking: false,
        connected: data.api_key_configured,
        model: data.model
      });
    } catch (error) {
      setStatus({
        checking: false,
        connected: false,
        error: 'Backend offline'
      });
    }
  };

  if (status.checking) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 rounded-lg border border-[#2a3a27] bg-black/40">
        <span className="material-symbols-outlined text-[#38ff14]/60 text-sm animate-spin">autorenew</span>
        <span className="text-[10px] text-[#38ff14]/60 uppercase tracking-widest">Initializing AI...</span>
      </div>
    );
  }

  if (!status.connected) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 rounded-lg border border-red-500/30 bg-red-900/10">
        <span className="material-symbols-outlined text-red-500 text-sm">error</span>
        <span className="text-[10px] text-red-500 uppercase tracking-widest">AI Offline</span>
      </div>
    );
  }

  return (
    <div className="relative group">
      {/* Main Status Badge */}
      <div className="flex items-center gap-2 px-4 py-2 rounded-lg border border-[#38ff14]/50 bg-[#38ff14]/10 cursor-pointer hover:bg-[#38ff14]/20 transition-all">
        <div className="relative">
          <span className="material-symbols-outlined text-[#38ff14] text-sm">neurology</span>
          <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-[#38ff14] rounded-full animate-pulse shadow-[0_0_8px_#38ff14]"></span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] text-[#38ff14] font-bold uppercase tracking-widest leading-none">AI Active</span>
          <span className="text-[8px] text-[#38ff14]/60 uppercase tracking-wider">{status.model}</span>
        </div>
      </div>

      {/* Hover Tooltip */}
      <div className="absolute top-full left-0 mt-2 w-64 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
        <div className="p-4 rounded-lg border border-[#38ff14]/30 bg-black/95 backdrop-blur-sm shadow-xl">
          <div className="flex items-center gap-2 mb-3">
            <span className="material-symbols-outlined text-[#38ff14] text-lg">psychology</span>
            <span className="text-[#38ff14] text-xs font-bold uppercase tracking-widest">AI System Status</span>
          </div>
          
          <div className="space-y-2 text-[10px]">
            <div className="flex justify-between items-center">
              <span className="text-[#38ff14]/60">Model:</span>
              <span className="text-[#38ff14] font-bold">{status.model}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#38ff14]/60">Status:</span>
              <span className="text-[#38ff14] font-bold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-[#38ff14] animate-pulse"></span>
                Online
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#38ff14]/60">Capabilities:</span>
              <span className="text-[#38ff14] font-bold">Full</span>
            </div>
          </div>

          <div className="mt-3 pt-3 border-t border-[#38ff14]/20">
            <div className="flex items-start gap-2">
              <span className="material-symbols-outlined text-[#38ff14] text-xs mt-0.5">info</span>
              <p className="text-[8px] text-[#38ff14]/60 uppercase leading-relaxed">
                GPT-4o is ready to analyze errors, generate fixes, and exorcise bugs from your codebase.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
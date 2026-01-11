'use client';

import React, { useEffect, useRef } from 'react';

interface TerminalViewerProps {
    logs?: string[];
}

export default function TerminalViewer({ logs = [] }: TerminalViewerProps) {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom when logs change
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    // Default demo logs if none provided
    const demoLogs = [
        "[22:04:11] SYSTEM :: Booting kernel version 4.1.2-PRIME...",
        "[22:04:11] SYSTEM :: Initializing hardware abstraction layer...",
        "[22:04:12] AUTH :: Login success. Welcome back, ROOT_USER.",
        "[22:04:12] UPLINK :: Scanning for available bridge nodes...",
        "[22:04:13] UPLINK :: Found Node-74 (Beijing), Node-12 (Amsterdam), Node-01 (San Francisco).",
        "[22:04:14] UPLINK :: Connecting to Node-01...",
        "[22:04:15] UPLINK :: ESTABLISHED. ENCRYPTION AES-256 ACTIVE.",
        "[22:05:01] PROCESS :: Running decryption script 'hydra_v4.sh'",
        "[22:05:01] DECRYPTING PACKETS... 74% - CRC Checksum Pending..."
    ];

    const currentLogs = logs.length > 0 ? logs : demoLogs;

    return (
        <div className="min-h-screen bg-black text-[#38ff14] font-mono relative selection:bg-[#38ff14] selection:text-black">

            {/* CRT Overlay */}
            <div className="crt-overlay fixed inset-0 pointer-events-none z-50"></div>

            <div className="flex bg-black h-screen w-full">
                {/* Sidebar */}
                <aside className="w-64 border-r border-[#2a3a27] bg-black/80 backdrop-blur-sm z-10 hidden lg:flex flex-col">
                    <div className="p-6 flex flex-col gap-8 h-full">
                        <div className="flex flex-col">
                            <h1 className="text-[#38ff14] text-sm font-bold tracking-widest uppercase neon-glow font-display">Root Console</h1>
                            <p className="text-[#38ff14]/50 text-xs font-mono mt-1">Uplink: [ACTIVE]</p>
                        </div>

                        <nav className="flex flex-col gap-2">
                            {[
                                { name: 'Main Frame', icon: 'terminal' },
                                { name: 'System Logs', icon: 'monitoring', active: true },
                                { name: 'Nodes', icon: 'language' },
                                { name: 'Encryptions', icon: 'security' }
                            ].map((item, idx) => (
                                <div key={idx} className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors cursor-pointer ${item.active ? 'bg-[#38ff14]/20 text-[#38ff14] border border-[#38ff14]/30' : 'text-[#38ff14]/60 hover:text-[#38ff14] hover:bg-[#38ff14]/10'}`}>
                                    <span className="material-symbols-outlined text-[20px]">{item.icon}</span>
                                    <p className="text-sm font-medium">{item.name}</p>
                                </div>
                            ))}
                        </nav>

                        <div className="mt-auto pt-6 border-t border-[#2a3a27]">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="size-8 rounded-full bg-[#38ff14]/20 border border-[#38ff14]/40 flex items-center justify-center">
                                    <span className="material-symbols-outlined text-[#38ff14] text-sm">person</span>
                                </div>
                                <div className="flex flex-col">
                                    <p className="text-white text-xs font-bold leading-none">ADMIN_X</p>
                                    <p className="text-[#38ff14]/40 text-[10px] uppercase tracking-tighter">Level 4 Access</p>
                                </div>
                            </div>
                            <button className="w-full py-2 bg-[#38ff14]/10 border border-[#38ff14]/20 text-[#38ff14] text-xs font-bold rounded-lg hover:bg-[#38ff14]/20 transition-all uppercase">
                                Secure Logout
                            </button>
                        </div>
                    </div>
                </aside>

                {/* Main Content */}
                <main className="flex-1 flex flex-col relative overflow-hidden">
                    {/* Header */}
                    <header className="h-14 border-b border-[#2a3a27] bg-black/90 flex items-center justify-between px-6 z-20 shrink-0">
                        <div className="flex items-center gap-6">
                            <div className="flex gap-2">
                                <div className="w-3 h-3 rounded-full bg-red-500/80 shadow-[0_0_5px_rgba(239,68,68,0.5)]"></div>
                                <div className="w-3 h-3 rounded-full bg-yellow-500/80 shadow-[0_0_5px_rgba(234,179,8,0.5)]"></div>
                                <div className="w-3 h-3 rounded-full bg-green-500/80 shadow-[0_0_5px_rgba(34,197,94,0.5)]"></div>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="material-symbols-outlined text-[#38ff14]/70 text-lg">terminal</span>
                                <h2 className="text-[#38ff14] text-sm font-bold tracking-widest uppercase neon-glow font-display">Terminal Logs — Session 0x821</h2>
                            </div>
                        </div>
                        <div className="flex gap-3">
                            {['search', 'file_download', 'settings'].map((icon, idx) => (
                                <button key={idx} className="w-8 h-8 flex items-center justify-center rounded border border-[#2a3a27] hover:border-[#38ff14]/50 text-[#38ff14]/70 hover:text-[#38ff14] transition-colors">
                                    <span className="material-symbols-outlined text-[18px]">{icon}</span>
                                </button>
                            ))}
                        </div>
                    </header>

                    {/* Stats Bar */}
                    <div className="flex flex-wrap gap-4 p-4 bg-black/40 border-b border-[#2a3a27]/50 shrink-0">
                        {[
                            { label: 'CPU Load', value: '14.02%', icon: 'bar_chart' },
                            { label: 'Memory Usage', value: '2.4 GB', icon: 'memory' },
                            { label: 'Network Latency', value: '12 ms', icon: 'speed' },
                            { label: 'Uptime', value: '102:14:02', icon: 'schedule' },
                        ].map((stat, idx) => (
                            <div key={idx} className="flex-1 min-w-[120px] p-3 rounded border border-[#2a3a27] bg-black/60">
                                <p className="text-[#38ff14]/40 text-[10px] font-bold uppercase tracking-widest mb-1">{stat.label}</p>
                                <div className="flex items-center justify-between">
                                    <p className="text-[#38ff14] text-xl font-mono neon-glow">{stat.value}</p>
                                    {idx === 0 ? (
                                        <div className="h-1 w-12 bg-[#38ff14]/20 rounded-full overflow-hidden">
                                            <div className="h-full bg-[#38ff14] w-[14%] shadow-[0_0_5px_#38ff14]"></div>
                                        </div>
                                    ) : (
                                        <span className="material-symbols-outlined text-[#38ff14]/40 text-sm">{stat.icon}</span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Terminal Area */}
                    <div className="flex-1 overflow-y-auto p-6 font-mono terminal-scroll bg-black/20" ref={scrollRef}>
                        <div className="max-w-4xl mx-auto space-y-1">
                            {currentLogs.map((log, index) => (
                                <div key={index} className="flex gap-4 group hover:bg-[#38ff14]/5 pl-2 -ml-2 rounded transition-colors">
                                    {/* If log content is a string, displayed as is. 
                                Assuming logs might vary in format, but for now treating as text.
                            */}
                                    <p className="text-[#38ff14]/80 break-words w-full">{log}</p>
                                </div>
                            ))}

                            {/* Input Line */}
                            <div className="py-2"></div>
                            <div className="flex gap-2 items-center">
                                <span className="text-[#38ff14] font-bold">root@terminal:~$</span>
                                <span className="text-[#38ff14] neon-glow">./init_payload --force</span>
                                <span className="cursor block w-2 h-4 bg-[#38ff14]"></span>
                            </div>
                        </div>
                    </div>

                    {/* Footer */}
                    <footer className="h-8 border-t border-[#2a3a27] bg-black/90 flex items-center justify-between px-4 text-[10px] font-mono shrink-0">
                        <div className="flex gap-4 text-[#38ff14]/40 uppercase tracking-widest">
                            <span>IP: 192.168.1.102</span>
                            <span>VPN: HOLLAND-04 (SECURED)</span>
                        </div>
                        <div className="flex gap-4 items-center">
                            <div className="flex items-center gap-1 text-[#38ff14]">
                                <span className="w-1.5 h-1.5 rounded-full bg-[#38ff14] animate-pulse"></span>
                                <span>LIVE STREAM</span>
                            </div>
                            <span className="text-[#38ff14]/40">UTF-8</span>
                        </div>
                    </footer>
                </main>
            </div>
        </div>
    );
}

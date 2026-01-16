'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import AIStatusIndicator from './AIStatusIndicator';

interface LayoutProps {
    children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [isSearchOpen, setIsSearchOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const pathname = usePathname();
    const router = useRouter();

    const handleDownload = () => {
        const data = {
            system: 'BUG_EXORCIST',
            status: 'Operational',
            timestamp: new Date().toISOString(),
            logs: [
                'Uplink established',
                'Core modules initialized',
                'Security protocols active'
            ]
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `bug_exorcist_logs_${new Date().getTime()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const menuItems = [
        { name: 'Dashboard', href: '/', icon: 'dashboard' },
        { name: 'Active Bugs', href: '/bugs', icon: 'bug_report' },
        { name: 'History', href: '/history', icon: 'history' },
        { name: 'Settings', href: '/settings', icon: 'settings' },
    ];

    return (
        <div className="min-h-screen bg-black text-[#38ff14] font-mono selection:bg-[#38ff14] selection:text-black">
            {/* CRT Overlay */}
            <div className="crt-overlay fixed inset-0 pointer-events-none z-50"></div>

            {/* Sidebar */}
            <aside 
                className={`fixed top-0 left-0 z-40 h-screen transition-transform ${
                    isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
                } w-64 border-r border-[#2a3a27] bg-black/95 backdrop-blur-sm`}
            >
                <div className="flex flex-col h-full px-4 py-6">
                    {/* Logo */}
                    <div className="flex items-center gap-3 mb-10 px-2">
                        <div className="size-10 rounded bg-[#38ff14]/20 border border-[#38ff14]/50 flex items-center justify-center animate-pulse">
                            <span className="material-symbols-outlined text-[#38ff14] text-2xl">pest_control</span>
                        </div>
                        <h1 className="text-xl font-bold tracking-tighter neon-glow">BUG_EXORCIST</h1>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 space-y-2">
                        {menuItems.map((item) => {
                            const isActive = pathname === item.href;
                            return (
                                <Link 
                                    key={item.name}
                                    href={item.href}
                                    className={`flex items-center gap-3 px-4 py-3 rounded-lg border transition-all duration-200 group ${
                                        isActive 
                                            ? 'bg-[#38ff14]/10 border-[#38ff14]/50 text-[#38ff14]' 
                                            : 'border-transparent text-[#38ff14]/60 hover:bg-[#38ff14]/5 hover:text-[#38ff14]'
                                    }`}
                                >
                                    <span className={`material-symbols-outlined text-xl ${isActive ? 'neon-glow' : 'group-hover:neon-glow'}`}>
                                        {item.icon}
                                    </span>
                                    <span className="text-sm font-bold tracking-widest uppercase">{item.name}</span>
                                    {isActive && (
                                        <div className="ml-auto w-1.5 h-1.5 rounded-full bg-[#38ff14] shadow-[0_0_8px_#38ff14]"></div>
                                    )}
                                </Link>
                            );
                        })}
                    </nav>

                    {/* Footer / User Profile */}
                    <div className="mt-auto pt-6 border-t border-[#2a3a27]">
                        <div className="flex items-center gap-3 p-2">
                            <div className="size-10 rounded-full bg-[#38ff14]/20 border border-[#38ff14]/40 flex items-center justify-center">
                                <span className="material-symbols-outlined text-[#38ff14]">person</span>
                            </div>
                            <div className="flex flex-col">
                                <p className="text-white text-xs font-bold leading-none">ADMIN_X</p>
                                <p className="text-[#38ff14]/40 text-[10px] uppercase tracking-tighter">Level 4 Access</p>
                            </div>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <div className={`transition-all duration-300 ${isSidebarOpen ? 'pl-64' : 'pl-0'}`}>
                {/* Top Navbar */}
                <header className="h-16 border-b border-[#2a3a27] bg-black/90 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-30">
                    <div className="flex items-center gap-4">
                        <button 
                            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                            className="p-2 rounded border border-[#2a3a27] hover:border-[#38ff14]/50 text-[#38ff14]/70 hover:text-[#38ff14] transition-colors"
                        >
                            <span className="material-symbols-outlined">
                                {isSidebarOpen ? 'menu_open' : 'menu'}
                            </span>
                        </button>
                        
                        {/* AI Status Indicator */}
                        <AIStatusIndicator />
                        
                        {/* Search Input (Expandable) */}
                        <div className={`flex items-center gap-2 transition-all duration-300 ${isSearchOpen ? 'w-64 opacity-100' : 'w-0 opacity-0 overflow-hidden'}`}>
                            <input 
                                type="text"
                                placeholder="SEARCH_DATABASE..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="bg-black/50 border border-[#38ff14]/30 rounded px-3 py-1 text-xs focus:border-[#38ff14] outline-none text-[#38ff14] w-full"
                                autoFocus={isSearchOpen}
                            />
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        {/* Search Toggle */}
                        <button 
                            onClick={() => setIsSearchOpen(!isSearchOpen)}
                            className={`w-10 h-10 flex items-center justify-center rounded border transition-all ${
                                isSearchOpen 
                                    ? 'bg-[#38ff14]/20 border-[#38ff14] text-[#38ff14]' 
                                    : 'border-[#2a3a27] hover:border-[#38ff14]/50 text-[#38ff14]/70 hover:text-[#38ff14]'
                            } group`}
                            title="Search"
                        >
                            <span className="material-symbols-outlined text-[22px] group-hover:neon-glow">search</span>
                        </button>

                        {/* Download Button */}
                        <button 
                            onClick={handleDownload}
                            className="w-10 h-10 flex items-center justify-center rounded border border-[#2a3a27] hover:border-[#38ff14]/50 text-[#38ff14]/70 hover:text-[#38ff14] transition-all group"
                            title="Download System Logs"
                        >
                            <span className="material-symbols-outlined text-[22px] group-hover:neon-glow">download</span>
                        </button>

                        {/* Settings Button */}
                        <button 
                            onClick={() => router.push('/settings')}
                            className={`w-10 h-10 flex items-center justify-center rounded border transition-all ${
                                pathname === '/settings'
                                    ? 'bg-[#38ff14]/20 border-[#38ff14] text-[#38ff14]'
                                    : 'border-[#2a3a27] hover:border-[#38ff14]/50 text-[#38ff14]/70 hover:text-[#38ff14]'
                            } group`}
                            title="Settings"
                        >
                            <span className="material-symbols-outlined text-[22px] group-hover:neon-glow">settings</span>
                        </button>
                    </div>
                </header>

                {/* Page Content */}
                <main className="p-6">
                    {children}
                </main>
            </div>
        </div>
    );
}
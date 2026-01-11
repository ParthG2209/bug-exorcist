import TerminalViewer from "../components/TerminalViewer";

export default function Home() {
    return (
        <main className="flex min-h-screen flex-col items-center justify-between">
            <div className="w-full">
                <TerminalViewer />
            </div>
        </main>
    );
}

"use client";
// This component will be used once the frontend is initialized with Next.js

import { useEffect, useState } from "react";

export default function SettingsPage() {
  const [mounted, setMounted] = useState(false);
  const [openaiKey, setOpenaiKey] = useState("");
  const [githubRepo, setGithubRepo] = useState("");

  useEffect(() => {
    setMounted(true);
    setOpenaiKey(localStorage.getItem("openai_api_key") || "");
    setGithubRepo(localStorage.getItem("github_repo_url") || "");
  }, []);

  const saveSettings = () => {
    if (!openaiKey.trim() || !githubRepo.trim()) {
      alert("Please fill in both the OpenAI API Key and GitHub Repository URL.");
      return;
    }
    localStorage.setItem("openai_api_key", openaiKey);
    localStorage.setItem("github_repo_url", githubRepo);
    alert("Settings saved successfully!");
  };

  if (!mounted) {
    return null;
  }

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Settings</h1>

      <div style={{ marginTop: "1rem" }}>
        <label>OpenAI API Key</label>
        <br />
        <input
          type="password"
          value={openaiKey}
          onChange={(e) => setOpenaiKey(e.target.value)}
        />
      </div>

      <div style={{ marginTop: "1rem" }}>
        <label>GitHub Repository URL</label>
        <br />
        <input
          type="text"
          value={githubRepo}
          onChange={(e) => setGithubRepo(e.target.value)}
        />
      </div>

      <button style={{ marginTop: "1.5rem" }} onClick={saveSettings}>
        Save
      </button>
    </div>
  );
}

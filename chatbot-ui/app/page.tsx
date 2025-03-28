"use client";

import { useState, useEffect, useRef } from "react";
import Navbar from "../components/Navbar";
import ChatWindow from "../components/ChatWindow";
import InputArea from "../components/InputArea";
import SettingsPanel from "../components/SettingsPanel";
import { FaArrowDown } from "react-icons/fa";

type ChatMessage = {
  role: "user" | "bot";
  content: string;
  thought: string;
};

function mergeToken(current: string, token: string): string {

  if (current.endsWith("`") && token.startsWith("`")) {
    return current + token;
  }

  let maxOverlap = 0;
  const len = Math.min(current.length, token.length);
  for (let i = 1; i <= len; i++) {
    if (current.slice(-i) === token.slice(0, i)) {
      maxOverlap = i;
    }
  }
  return current + token.slice(maxOverlap);
}


export default function ChatPage() {
  const [maxTokens, setMaxTokens] = useState<number>(512);
  const [temperature, setTemperature] = useState<number>(0.0);
  const [topK, setTopK] = useState<number>(1);
  const [topP, setTopP] = useState<number>(0.95);
  const [repetitionPenalty, setRepetitionPenalty] = useState<number>(1.0);
  const [frequencyPenalty, setFrequencyPenalty] = useState<number>(0.0);
  const [presencePenalty, setPresencePenalty] = useState<number>(0.0);

  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [prompt, setPrompt] = useState("");
  const [conversations, setConversations] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const pendingBotMessageRef = useRef<string>("");
  const updateScheduledRef = useRef<boolean>(false);
  const userInterruptedScroll = useRef<boolean>(false);
  const generationTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [tokenCount, setTokenCount] = useState<number>(0);
  const [tokensPerSecond, setTokensPerSecond] = useState<number>(0);
  const startTimeRef = useRef<number | null>(null);
  

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);

  const profiles = [
    {
      name: "Less helu",
      maxTokens: 8192,
      temperature: 0.1,
      topK: 40,
      topP: 0.9,
      repetitionPenalty: 1.1,
      frequencyPenalty: 0.05,
      presencePenalty: 0.05,
    },
    {
      name: "Default",
      maxTokens: 1024,
      temperature: 0.0,
      topK: 40,
      topP: 0.95,
      repetitionPenalty: 1.0,
      frequencyPenalty: 0.0,
      presencePenalty: 0.0,
    },
    {
      name: "Creative",
      maxTokens: 1024,
      temperature: 0.55,
      topK: 50,
      topP: 0.8,
      repetitionPenalty: 1.1,
      frequencyPenalty: 0.1,
      presencePenalty: 0.05,
    },
  ];

  const [selectedProfileIndex, setSelectedProfileIndex] = useState(0);

  const switchToProfile = (index: number) => {
    const p = profiles[index];
    setMaxTokens(p.maxTokens);
    setTemperature(p.temperature);
    setTopK(p.topK);
    setTopP(p.topP);
    setRepetitionPenalty(p.repetitionPenalty);
    setFrequencyPenalty(p.frequencyPenalty);
    setPresencePenalty(p.presencePenalty);
    setSelectedProfileIndex(index);
  };

  useEffect(() => {
    switchToProfile(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  

  useEffect(() => {
    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    let wsPort = "";
    if (window.location.hostname === "localhost") {
      wsPort = ":7000";
    }
    const ws = new WebSocket(`${wsProtocol}://${window.location.hostname}${wsPort}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.log("WebSocket connection opened");
    };

    ws.onmessage = (event) => {
      const token = event.data;
    
      setTokenCount((prevTokenCount) => {
        const newTokenCount = prevTokenCount + 1;
    
        if (!startTimeRef.current) {
          startTimeRef.current = Date.now();
        }
    
        const elapsedSeconds = (Date.now() - startTimeRef.current) / 1000;
        setTokensPerSecond(newTokenCount / elapsedSeconds);
    
        return newTokenCount;
      });
    
      pendingBotMessageRef.current = mergeToken(pendingBotMessageRef.current, token);
    
      if (generationTimeoutRef.current) {
        clearTimeout(generationTimeoutRef.current);
      }
    
      setIsGenerating(true);
    
      generationTimeoutRef.current = setTimeout(() => {
        setIsGenerating(false);
        generationTimeoutRef.current = null;
        startTimeRef.current = null;
        setTokenCount(0);
      }, 1000); 
    
      if (!updateScheduledRef.current) {
        updateScheduledRef.current = true;
        setTimeout(() => {
          setConversations((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
    
            if (updated[lastIndex].role === "bot") {
              const fullMessage = pendingBotMessageRef.current;
    
              const thinkStart = fullMessage.indexOf("<think>");
              const thinkEnd = fullMessage.indexOf("</think>");
    
              if (thinkStart !== -1 && thinkEnd === -1) {
                updated[lastIndex].thought = fullMessage.substring(thinkStart + 7).trim();
                updated[lastIndex].content = fullMessage.substring(0, thinkStart).trim();
              } else if (thinkStart !== -1 && thinkEnd !== -1) {
                updated[lastIndex].thought = fullMessage.substring(thinkStart + 7, thinkEnd).trim();
                updated[lastIndex].content = (fullMessage.substring(0, thinkStart) + fullMessage.substring(thinkEnd + 8)).trim();
              } else if (thinkStart === -1 && thinkEnd === -1 && updated[lastIndex].thought) {
                updated[lastIndex].thought += token;
              } else {
                updated[lastIndex].content = fullMessage;
              }
            }
    
            return updated;
          });
          updateScheduledRef.current = false;
        }, 20);
      }
    };
    
    
    
    
    

    ws.onclose = () => {
      setIsConnected(false);
      console.log("WebSocket connection closed");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error: ", error);
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setConversations((prev) => [
      ...prev,
      { role: "user", content: prompt },
      { role: "bot", content: "" },
    ]);
    pendingBotMessageRef.current = "";
    userInterruptedScroll.current = false;
    setAutoScroll(true);
    setIsGenerating(true);

    const payload = {
      prompt,
      max_tokens: maxTokens,
      temperature,
      top_k: topK,
      top_p: topP,
      repetition_penalty: repetitionPenalty,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
    };

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    } else {
      console.error("WebSocket connection is not open.");
    }

    setPrompt("");
  };

  // New: Handle the Search bubble click.
  const handleSearch = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setConversations((prev) => [
      ...prev,
      { role: "user", content: prompt },
      { role: "bot", content: "" },
    ]);
    pendingBotMessageRef.current = "";
    userInterruptedScroll.current = false;
    setAutoScroll(true);
    setIsGenerating(true);

    const payload = {
      prompt,
      search: true, // flag for search
      max_tokens: maxTokens,
      temperature,
      top_k: topK,
      top_p: topP,
      repetition_penalty: repetitionPenalty,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
    };

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    } else {
      console.error("WebSocket connection is not open.");
    }

    setPrompt("");
  };

  const handleScroll = () => {
    if (!chatContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop <= clientHeight + 5;
    if (!isAtBottom) {
      setAutoScroll(false);
      userInterruptedScroll.current = true;
    }
  };

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: "smooth",
      });
      setAutoScroll(true);
      userInterruptedScroll.current = false;
    }
  };

  return (
    <div
      className={`min-h-screen flex flex-col ${
        theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-gray-900"
      }`}
    >
      <Navbar
        theme={theme}
        toggleTheme={toggleTheme}
        toggleSettings={() => setShowSettings((prev) => !prev)}
        profiles={profiles}
        selectedProfileIndex={selectedProfileIndex}
        onSelectProfile={switchToProfile}
      />

      {showSettings && (
        <div className="fixed inset-0 bg-gray-900 bg-opacity-50 z-40 flex items-start justify-center">
          <SettingsPanel
            maxTokens={maxTokens}
            setMaxTokens={setMaxTokens}
            temperature={temperature}
            setTemperature={setTemperature}
            topK={topK}
            setTopK={setTopK}
            topP={topP}
            setTopP={setTopP}
            repetitionPenalty={repetitionPenalty}
            setRepetitionPenalty={setRepetitionPenalty}
            frequencyPenalty={frequencyPenalty}
            setFrequencyPenalty={setFrequencyPenalty}
            presencePenalty={presencePenalty}
            setPresencePenalty={setPresencePenalty}
            closePanel={() => setShowSettings(false)}
          />
        </div>
      )}

      <div
        className="flex-1 overflow-auto mt-16"
        ref={chatContainerRef}
        onScroll={handleScroll}
      >
        <ChatWindow
          conversations={conversations}
          containerRef={chatContainerRef}
          theme={theme}
          isGenerating={isGenerating}
          onScroll={() => {}}
        />
      </div>

      <div className="sticky bottom-0">
        <InputArea
          prompt={prompt}
          setPrompt={setPrompt}
          handleSubmit={handleSubmit}
          handleSearch={handleSearch}  // pass the new search handler
          isBusy={!isConnected || isGenerating}
        />
      </div>
      <div className="fixed bottom-20 right-4 bg-gray-800 text-white px-3 py-1 rounded-full shadow-lg z-50">
  Tokens: {tokenCount} | TPS: {tokensPerSecond.toFixed(2)}
</div>

    </div>
    
  );
}

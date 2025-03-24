// ChatWindow.tsx
import MarkdownRender from "./MarkdownRender";
import ThoughtProcess from "./ThoughtProcess";

type ChatMessage = {
  role: "user" | "bot";
  content: string;
  thought?: string;
};

type ChatWindowProps = {
  conversations: ChatMessage[];
  containerRef: React.RefObject<HTMLDivElement>;
  theme: "dark" | "light";
  isGenerating: boolean;
};

export default function ChatWindow({ conversations, containerRef, theme, isGenerating }) {
  return (
    <main className="flex-1 overflow-auto p-4" ref={containerRef}>
      <div className="max-w-5xl mx-auto space-y-4">
        {conversations.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`rounded-lg p-4 max-w-md md:max-w-4xl break-words shadow-lg ${
              msg.role === "user"
                ? "bg-blue-600 text-white"
                : theme === "dark"
                  ? "bg-gray-700 text-white"
                  : "bg-gray-100 text-gray-900"
              }`}
            >
              {msg.role === "bot" && msg.thought && (
                <ThoughtProcess thought={msg.thought} isGenerating={isGenerating && idx === conversations.length - 1} />
              )}
              <MarkdownRender content={msg.content} theme={theme} />
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}




// MarkdownRender.tsx
import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/cjs/styles/prism";

function fixMarkdown(content: string): string {
  return content.replace(/(```\s*)+$/g, "```");
}

export default function MarkdownRender({ content, theme }) {
  const sanitizedContent = fixMarkdown(content);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000); // Reset after 2 seconds
  };

  let codeBlockCounter = 0;

  return (
    <ReactMarkdown
      components={{
        code({ inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          if (!inline && match) {
            const currentIndex = codeBlockCounter++;
            return (
              <div className="relative">
                <button
                  className="absolute top-1 right-1 text-xs bg-gray-600 px-1 rounded"
                  onClick={() => handleCopy(String(children), currentIndex)}
                >
                  {copiedIndex === currentIndex ? "Copied!" : "Copy"}
                </button>
                <SyntaxHighlighter
                  style={theme === "dark" ? vscDarkPlus : {}}
                  language={match[1]}
                  PreTag="div"
                >
                  {String(children).replace(/\n$/, "")}
                </SyntaxHighlighter>
              </div>
            );
          } else {
            return (
              <code className="bg-gray-700 px-1 rounded">
                {children}
              </code>
            );
          }
        },
      }}
    >
      {sanitizedContent}
    </ReactMarkdown>
  );
}

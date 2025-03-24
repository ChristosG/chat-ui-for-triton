"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { FaChevronDown, FaChevronUp } from "react-icons/fa";

export default function ThoughtProcess({
  thought,
  isGenerating,
}: {
  thought: string;
  isGenerating: boolean;
}) {
  const [expanded, setExpanded] = useState(true);

  useEffect(() => {
    if (!isGenerating) setExpanded(false);
  }, [isGenerating]);

  return (
    <div className="bg-gray-800 text-xs rounded shadow-md mb-2">
      <button
        className="flex items-center justify-between w-full py-1 px-2 text-gray-400"
        onClick={() => setExpanded((prev) => !prev)}
      >
        🧠 Thought process {expanded ? <FaChevronUp /> : <FaChevronDown />}
      </button>
      {expanded && (
        <div className="p-2 border-t border-gray-700 whitespace-pre-wrap font-mono">
          {thought.split("").map((char, idx) => (
            <motion.span
              key={`${char}-${idx}`}
              initial={{ opacity: 0.3 }}
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{
                duration: 1,
                repeat: isGenerating ? Infinity : 0,
                delay: idx * 0.015,
              }}
            >
              {char}
            </motion.span>
          ))}
        </div>
      )}
    </div>
  );
}

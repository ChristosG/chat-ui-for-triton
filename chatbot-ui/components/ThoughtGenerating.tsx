'use client';
import { motion } from "framer-motion";

export default function ThoughtGenerating() {
  return (
    <motion.div
      className="text-xs italic text-gray-400 mt-2"
      initial={{ opacity: 0.5 }}
      animate={{ opacity: [0.5, 1, 0.5] }}
      transition={{ duration: 1, repeat: Infinity }}
    >
      🧠 Generating thought...
    </motion.div>
  );
}

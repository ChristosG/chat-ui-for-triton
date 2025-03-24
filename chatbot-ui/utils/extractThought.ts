// utils/extractThought.ts
export function extractThought(message: string) {
    const thoughtRegex = /<think>([\s\S]*?)<\/think>/;
    const match = message.match(thoughtRegex);
  
    if (match) {
      const thought = match[1].trim();
      const response = message.replace(thoughtRegex, "").trim();
      return { thought, response };
    }
    return { thought: null, response: message };
  }
  
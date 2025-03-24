import React, { useRef } from "react";
import TextareaAutosize from "react-textarea-autosize";

type InputAreaProps = {
  prompt: string;
  setPrompt: (value: string) => void;
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  // New prop for search action.
  handleSearch: (e: React.MouseEvent<HTMLButtonElement>) => void;
  isBusy: boolean;
};

export default function InputArea({ prompt, setPrompt, handleSubmit, handleSearch, isBusy }: InputAreaProps) {
  const formRef = useRef<HTMLFormElement>(null);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      formRef.current?.requestSubmit();
    }
  };

  return (
    <footer className="p-4 shadow-md">
      <form
        ref={formRef}
        onSubmit={handleSubmit}
        className="max-w-3xl mx-auto flex rounded-lg focus-within:ring-2 focus-within:ring-blue-500"
      >
        <TextareaAutosize
          minRows={2}
          maxRows={7}
          placeholder="Ρώτα με ό,τι θες..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={onKeyDown}
          className="flex-1 p-4 rounded-l-lg border-none outline-none dark:bg-gray-800 dark:text-white text-black resize-none"
        />
        {/* Search bubble button */}
        <button
          type="button"
          onClick={handleSearch}
          disabled={isBusy}
          className="bg-green-600 rounded-lg hover:bg-green-700 text-white px-4 py-2 "
        >
          Search
        </button>
        <button
          type="submit"
          disabled={isBusy}
          className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-4 rounded-r-lg disabled:opacity-50"
        >
          {isBusy ? "Waiting..." : "Send"}
        </button>
      </form>
    </footer>
  );
}

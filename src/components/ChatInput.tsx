import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send, Paperclip } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, disabled }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-2">
      <div className="flex-1 relative">
        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Paste news content, headlines, or claims to fact-check..."
          disabled={disabled}
          className="min-h-[3rem] max-h-32 resize-none pr-10 bg-chat-input-bg border-border/60 focus:border-primary/60 transition-colors"
          rows={1}
        />
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="absolute right-1 bottom-1 h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
          onClick={() => {
            // Future: Handle file attachments for images/PDFs
          }}
        >
          <Paperclip className="h-4 w-4" />
        </Button>
      </div>
      
      <Button
        type="submit"
        disabled={!message.trim() || disabled}
        className="h-12 w-12 rounded-full bg-gradient-primary hover:opacity-90 transition-all duration-200 shadow-lg hover:shadow-xl disabled:opacity-50"
      >
        <Send className="h-4 w-4" />
      </Button>
    </form>
  );
};
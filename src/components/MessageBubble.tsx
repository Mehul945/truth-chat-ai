import React from 'react';
import { Message } from './ChatInterface';
import { cn } from '@/lib/utils';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.sender === 'user';
  const analysis = message.analysis;

  const getAnalysisColors = () => {
    if (!analysis) return '';
    
    switch (analysis.verdict) {
      case 'real':
        return 'border-success/30 bg-success/5';
      case 'fake':
        return 'border-destructive/30 bg-destructive/5';
      default:
        return 'border-warning/30 bg-warning/5';
    }
  };

  const getVerdictIcon = () => {
    if (!analysis) return '';
    
    switch (analysis.verdict) {
      case 'real':
        return '✅';
      case 'fake':
        return '❌';
      default:
        return '⚠️';
    }
  };

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] sm:max-w-[70%] rounded-2xl px-4 py-3 shadow-sm",
          isUser
            ? "bg-gradient-primary text-chat-user-text ml-12"
            : cn(
                "bg-chat-ai-bg text-chat-ai-text mr-12",
                analysis && getAnalysisColors()
              )
        )}
      >
        {!isUser && analysis && (
          <div className="flex items-center gap-2 mb-2 pb-2 border-b border-current/10">
            <span className="text-lg">{getVerdictIcon()}</span>
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-sm">
                  {analysis.verdict.toUpperCase()} NEWS
                </span>
                <span className="text-xs opacity-75">
                  {(analysis.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
              <div className="w-full bg-current/10 rounded-full h-1.5 mt-1">
                <div
                  className={cn(
                    "h-1.5 rounded-full transition-all duration-500",
                    analysis.verdict === 'real' && "bg-success",
                    analysis.verdict === 'fake' && "bg-destructive", 
                    analysis.verdict === 'uncertain' && "bg-warning"
                  )}
                  style={{ width: `${analysis.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        )}

        <div className="whitespace-pre-wrap text-sm leading-relaxed">
          {message.content.split('\n').map((line, index) => {
            // Handle markdown-style formatting
            if (line.startsWith('**') && line.endsWith('**')) {
              return (
                <div key={index} className="font-semibold mb-1">
                  {line.slice(2, -2)}
                </div>
              );
            }
            if (line.startsWith('• ')) {
              return (
                <div key={index} className="ml-2 opacity-90">
                  {line}
                </div>
              );
            }
            return line ? (
              <div key={index}>{line}</div>
            ) : (
              <div key={index} className="h-2" />
            );
          })}
        </div>

        <div className={cn(
          "text-xs mt-2 opacity-60",
          isUser ? "text-right" : "text-left"
        )}>
          {message.timestamp.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
          })}
        </div>
      </div>
    </div>
  );
};
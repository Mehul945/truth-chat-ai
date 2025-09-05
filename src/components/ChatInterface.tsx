import React, { useState, useRef, useEffect } from 'react';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { ScrollArea } from '@/components/ui/scroll-area';

export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  analysis?: {
    verdict: 'real' | 'fake' | 'uncertain';
    confidence: number;
    reasons: string[];
  };
}

export const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Welcome to FakeNews Detective! 🕵️ I\'m here to help you analyze news content for credibility. Simply paste any news article, headline, or claim you\'d like me to fact-check.',
      sender: 'ai',
      timestamp: new Date(),
    }
  ]);
  
  const [isTyping, setIsTyping] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const analyzeNews = (content: string): Message['analysis'] => {
    // Simple mock analysis - in real app this would be AI-powered
    const keywords = content.toLowerCase();
    let verdict: 'real' | 'fake' | 'uncertain' = 'uncertain';
    let confidence = Math.random() * 0.4 + 0.3; // 30-70%
    const reasons: string[] = [];

    if (keywords.includes('breaking') || keywords.includes('exclusive')) {
      confidence += 0.1;
      reasons.push('Uses attention-grabbing language');
    }
    
    if (keywords.includes('scientists say') || keywords.includes('study shows')) {
      verdict = 'real';
      confidence += 0.2;
      reasons.push('References scientific sources');
    }
    
    if (keywords.includes('miracle cure') || keywords.includes('doctors hate')) {
      verdict = 'fake';
      confidence += 0.3;
      reasons.push('Contains sensationalist medical claims');
    }
    
    if (keywords.includes('government') || keywords.includes('conspiracy')) {
      confidence -= 0.1;
      reasons.push('Contains politically sensitive content');
    }

    if (content.length < 50) {
      confidence -= 0.2;
      reasons.push('Limited content for comprehensive analysis');
    }

    if (confidence > 0.7) {
      verdict = verdict === 'uncertain' ? 'real' : verdict;
    } else if (confidence < 0.4) {
      verdict = 'uncertain';
      reasons.push('Insufficient evidence for confident assessment');
    }

    return {
      verdict,
      confidence: Math.min(0.95, Math.max(0.15, confidence)),
      reasons: reasons.length ? reasons : ['Standard credibility assessment completed']
    };
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);

    // Simulate AI processing delay
    setTimeout(() => {
      const analysis = analyzeNews(content);
      let responseContent = '';

      switch (analysis.verdict) {
        case 'real':
          responseContent = `✅ **Likely REAL News** (${(analysis.confidence * 100).toFixed(0)}% confidence)

This content appears to be credible based on my analysis.`;
          break;
        case 'fake':
          responseContent = `❌ **Likely FAKE News** (${(analysis.confidence * 100).toFixed(0)}% confidence)

This content shows signs of misinformation or unreliable claims.`;
          break;
        default:
          responseContent = `⚠️ **Uncertain Classification** (${(analysis.confidence * 100).toFixed(0)}% confidence)

I need more context or sources to make a confident assessment.`;
      }

      responseContent += `\n\n**Analysis factors:**\n${analysis.reasons.map(reason => `• ${reason}`).join('\n')}`;

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: responseContent,
        sender: 'ai',
        timestamp: new Date(),
        analysis,
      };

      setMessages(prev => [...prev, aiMessage]);
      setIsTyping(false);
    }, 1500);
  };

  const examplePrompts = [
    "Breaking: Scientists discover miracle anti-aging pill that doctors don't want you to know!",
    "New study from Harvard Medical School shows benefits of Mediterranean diet for heart health",
    "Local government announces new infrastructure investment for road improvements"
  ];

  return (
    <div className="flex flex-col h-full bg-gradient-bg">
      <header className="flex-shrink-0 p-4 border-b bg-card/80 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            🕵️ FakeNews Detective
            <span className="text-sm font-normal text-muted-foreground ml-2">AI-Powered Fact Checker</span>
          </h1>
        </div>
      </header>

      <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full">
        <ScrollArea ref={scrollAreaRef} className="flex-1 px-4 py-6">
          <div className="space-y-4">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-chat-ai-bg rounded-2xl px-4 py-3 max-w-xs">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {messages.length === 1 && (
          <div className="px-4 pb-4">
            <div className="bg-card/50 rounded-xl p-4 mb-4">
              <p className="text-sm text-muted-foreground mb-3">Try these examples:</p>
              <div className="space-y-2">
                {examplePrompts.map((prompt, index) => (
                  <button
                    key={index}
                    onClick={() => handleSendMessage(prompt)}
                    className="w-full text-left p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors text-sm"
                  >
                    "{prompt}"
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="flex-shrink-0 p-4 border-t bg-card/80 backdrop-blur-sm">
          <ChatInput onSend={handleSendMessage} disabled={isTyping} />
        </div>
      </div>
    </div>
  );
};
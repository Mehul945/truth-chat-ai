import React, { useState, useRef, useEffect } from 'react';
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
    explanation?: Array<{
      word: string;
      weight: number;
      importance: 'positive' | 'negative';
      abs_weight: number;
    }>;
    highlightedText?: string;
  };
}

const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.sender === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-gray-100 text-gray-800'
        }`}
      >
        {message.analysis?.highlightedText ? (
          <div 
            dangerouslySetInnerHTML={{ __html: message.analysis.highlightedText }}
            className="whitespace-pre-wrap"
          />
        ) : (
          <div className="whitespace-pre-wrap">{message.content}</div>
        )}
        
        {message.analysis && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <div
              className={`font-semibold ${
                message.analysis.verdict === "fake" ? "text-red-600" : "text-green-600"
              }`}
            >
              {message.analysis.verdict === "fake" ? "❌ Likely FAKE" : "✅ Likely REAL"}{" "}
              ({(message.analysis.confidence * 100).toFixed(0)}% confidence)
            </div>

            <p className="mt-2 text-gray-700 text-sm leading-relaxed">
              {message.analysis.verdict === "fake"
                ? "This content shows signs of misinformation, exaggerated claims, or unreliable sources."
                : "This content matches credible patterns and appears to come from a reliable source."}
            </p>
          </div>

        )}
      </div>
    </div>
  );
};

const ChatInput: React.FC<{ 
  onSend: (message: string) => void; 
  disabled: boolean;
  value: string;
  onChange: (value: string) => void;
}> = ({ onSend, disabled, value, onChange }) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !disabled) {
        onSend(value.trim());
      }
    }
  };

  const handleCheckClick = () => {
    if (value.trim() && !disabled) {
      onSend(value.trim());
    }
  };

  return (
    <div className="flex gap-3">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Paste news content here for fact-checking..."
        disabled={disabled}
        className="flex-1 min-h-[60px] max-h-[120px] px-4 py-3 rounded-xl border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none resize-none"
        rows={2}
      />
      <button
        onClick={handleCheckClick}
        disabled={disabled || !value.trim()}
        className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium whitespace-nowrap"
      >
        Check News
      </button>
    </div>
  );
};

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
  const [inputValue, setInputValue] = useState('');
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

  const highlightText = (text: string, explanation: any[]) => {
    if (!explanation || explanation.length === 0) return text;

    // Create a map of words to their weights for easier lookup
    const wordWeights = new Map();
    explanation.forEach(item => {
      wordWeights.set(item.word.toLowerCase(), {
        weight: item.weight,
        importance: item.importance,
        abs_weight: item.abs_weight
      });
    });

    // Split text into words while preserving spaces and punctuation
    const words = text.split(/(\s+|[.,!?;:])/);
    
    return words.map((word, index) => {
      const cleanWord = word.toLowerCase().replace(/[.,!?;:]/g, '');
      const wordData = wordWeights.get(cleanWord);
      
      if (wordData && wordData.abs_weight > 0.001) { // Only highlight words with significant weight
        const intensity = Math.min(wordData.abs_weight * 20, 1); // Scale intensity
        const isPositive = wordData.importance === 'positive';
        
        // Positive weights (supporting fake classification) = red background
        // Negative weights (supporting real classification) = green background
        const bgColor = isPositive 
          ? `rgba(239, 68, 68, ${intensity * 0.8})` // Red for fake indicators
          : `rgba(34, 197, 94, ${intensity * 0.8})`; // Green for real indicators
        
        const textColor = intensity > 0.5 ? 'white' : 'inherit';
        const title = `${wordData.importance === 'positive' ? 'Fake indicator' : 'Real indicator'}: ${(wordData.abs_weight * 100).toFixed(2)}% influence`;
        
        return `<span key="${index}" style="background-color: ${bgColor}; color: ${textColor}; padding: 2px 4px; border-radius: 4px; font-weight: bold;" title="${title}">${word}</span>`;
      }
      
      return word;
    }).join('');
  };

  const callPredictionAPI = async (content: string, shouldAddToChat: boolean = false) => {
    if (shouldAddToChat) {
      const userMessage: Message = {
        id: Date.now().toString(),
        content,
        sender: 'user',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, userMessage]);
    }

    setIsTyping(true);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          news_text: content
        })
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const result = await response.json();
      
      const analysis = {
        verdict: result.prediction.toLowerCase() as 'real' | 'fake',
        confidence: result.probability[result.prediction],
        reasons: [`Confidence score: ${(result.probability[result.prediction] * 100).toFixed(1)}%`],
        explanation: result.explanation,
        highlightedText: highlightText(content, result.explanation)
      };

      let responseContent = '';
      let explanation_summary = result.explanation_summary || `Key factors: ${result.explanation?.map((e: any) => e.word).join(', ')}`;
      
      switch (analysis.verdict) {
        case 'real':
          responseContent = `✅ **Likely REAL News** (${(analysis.confidence * 100).toFixed(0)}% confidence)\n\nThis content appears to be credible based on my analysis.\n${explanation_summary}`;
          break;
        case 'fake':
          responseContent = `❌ **Likely FAKE News** (${(analysis.confidence * 100).toFixed(0)}% confidence)\n\nThis content shows signs of misinformation or unreliable claims.\n${explanation_summary}`;
          break;
      }

      if (shouldAddToChat) {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: responseContent,
          sender: 'ai',
          timestamp: new Date(),
          analysis
        };
        setMessages(prev => [...prev, aiMessage]);
      } else {
        // For predict mode, update the user's message with highlighting
        setMessages(prev => prev.map(msg => {
          if (msg.content === content && msg.sender === 'user') {
            return {
              ...msg,
              analysis: {
                ...analysis,
                highlightedText: highlightText(content, result.explanation)
              }
            };
          }
          return msg;
        }));
      }
    } catch (error) {
      console.error('Error calling prediction API:', error);
      // Handle error case
    } finally {
      setIsTyping(false);
    }
  };

  const handleSendMessage = async (content: string) => {
    setInputValue('');
    await callPredictionAPI(content, true);
  };

  const examplePrompts = [
    "Breaking: Scientists discover miracle anti-aging pill that doctors don't want you to know!",
    "New study from Harvard Medical School shows benefits of Mediterranean diet for heart health",
    "Local government announces new infrastructure investment for road improvements"
  ];

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="flex-shrink-0 p-4 border-b bg-white shadow-sm">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            🕵️ FakeNews Detective
            <span className="text-sm font-normal text-gray-600 ml-2">AI-Powered Fact Checker</span>
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
                <div className="bg-gray-100 rounded-2xl px-4 py-3 max-w-xs">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {messages.length === 1 && (
          <div className="px-4 pb-4">
            <div className="bg-white rounded-xl p-4 mb-4 shadow-sm">
              <p className="text-sm text-gray-600 mb-3">Try these examples:</p>
              <div className="space-y-2">
                {examplePrompts.map((prompt, index) => (
                  <button
                    key={index}
                    onClick={() => setInputValue(prompt)}
                    className="w-full text-left p-3 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors text-sm"
                  >
                    "{prompt}"
                  </button>
                ))}
              </div>
            </div>
            
            <div className="bg-blue-50 rounded-xl p-4 mb-4">
              <h3 className="font-semibold text-blue-800 mb-2">How to use:</h3>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Paste your news content and click <strong>"Check News"</strong> to analyze</li>
                <li>• <strong>Word colors:</strong> <span className="bg-red-200 px-2 py-1 rounded">Red = Fake indicators</span> <span className="bg-green-200 px-2 py-1 rounded">Green = Real indicators</span></li>
                <li>• Hover over highlighted words to see their influence on the decision</li>
              </ul>
            </div>
          </div>
        )}

        <div className="flex-shrink-0 p-4 border-t bg-white">
          <ChatInput 
            onSend={handleSendMessage} 
            disabled={isTyping}
            value={inputValue}
            onChange={setInputValue}
          />
        </div>
      </div>
    </div>
  );
};

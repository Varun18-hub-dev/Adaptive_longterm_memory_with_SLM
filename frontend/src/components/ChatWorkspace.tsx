import React, { useRef, useEffect, useState } from 'react';
import {
  Menu,
  SlidersHorizontal,
  Send,
  Sparkles,
  Zap,
  Clock,
  ArrowRight,
  Brain,
  CheckCircle2,
  X,
} from 'lucide-react';
import type { ChatMessage, ChatRequest, HealthResponse } from '../types';

interface ChatWorkspaceProps {
  messages: ChatMessage[];
  onSendMessage: (prompt: string, config: Partial<ChatRequest>) => Promise<void>;
  isLoading: boolean;
  activeSessionId: string;
  isSidebarOpen: boolean;
  onToggleSidebar: () => void;
  isEvidenceOpen: boolean;
  onToggleEvidence: () => void;
  activeTraceId: string | null;
  onSelectTraceId: (traceId: string) => void;
  health: HealthResponse | null;
}

const PRESET_SUGGESTIONS = [
  'I currently work as a senior backend engineer at Stripe.',
  'Where do I work now?',
  'I used to work at OpenAI prior to joining Stripe.',
];

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  messages,
  onSendMessage,
  isLoading,
  activeSessionId,
  isSidebarOpen,
  onToggleSidebar,
  isEvidenceOpen,
  onToggleEvidence,
  activeTraceId,
  onSelectTraceId,
  health,
}) => {
  const [prompt, setPrompt] = useState('');
  const [showSettings, setShowSettings] = useState(false);

  // Settings
  const [storageBudget, setStorageBudget] = useState(150);
  const [maxContextTokens, setMaxContextTokens] = useState(256);
  const [useHybridExtractor, setUseHybridExtractor] = useState(false);
  const [useFakeLlm, setUseFakeLlm] = useState(false);
  const [enableAdmissionGate, setEnableAdmissionGate] = useState(true);

  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    onSendMessage(prompt.trim(), {
      storage_budget: storageBudget,
      max_context_tokens: maxContextTokens,
      use_hybrid_extractor: useHybridExtractor,
      use_fake_llm: useFakeLlm,
      enable_admission_gate: enableAdmissionGate,
    });
    setPrompt('');

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 140)}px`;
  };

  return (
    <main className="chat-workspace">
      {/* Header */}
      <header className="chat-header">
        <div className="chat-header-left">
          {!isSidebarOpen && (
            <button
              onClick={onToggleSidebar}
              className="icon-btn"
              title="Open sidebar"
              aria-label="Open sidebar"
            >
              <Menu size={18} />
            </button>
          )}

          <div className="active-session-badge">
            <span
              className={`status-dot-sm ${
                health?.status === 'healthy' ? '' : 'offline'
              }`}
            />
            <span>{activeSessionId}</span>
          </div>
        </div>

        <div className="chat-header-right">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="icon-btn"
            title="Pipeline settings"
            aria-label="Pipeline settings"
          >
            <SlidersHorizontal size={16} />
          </button>

          <button
            onClick={onToggleEvidence}
            className={`evidence-toggle-btn ${isEvidenceOpen ? 'active' : ''}`}
            title="Toggle Memory Evidence panel"
          >
            <Zap size={13} />
            <span>Memory Evidence</span>
            {activeTraceId && (
              <span
                style={{
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  opacity: 0.75,
                }}
              >
                #{activeTraceId.slice(0, 6)}
              </span>
            )}
          </button>
        </div>
      </header>

      {/* Settings Popover */}
      {showSettings && (
        <div className="settings-popover">
          <div className="settings-header">
            <span>Pipeline Configuration</span>
            <button
              onClick={() => setShowSettings(false)}
              style={{ opacity: 0.6 }}
            >
              <X size={14} />
            </button>
          </div>

          <div className="settings-row">
            <span>Storage Token Budget:</span>
            <input
              type="number"
              min={20}
              max={2000}
              className="settings-input-sm"
              value={storageBudget}
              onChange={(e) => setStorageBudget(Number(e.target.value))}
            />
          </div>

          <div className="settings-row">
            <span>Context Token Limit:</span>
            <input
              type="number"
              min={32}
              max={1024}
              className="settings-input-sm"
              value={maxContextTokens}
              onChange={(e) => setMaxContextTokens(Number(e.target.value))}
            />
          </div>

          <div className="settings-row">
            <span>Hybrid SLM Extractor:</span>
            <input
              type="checkbox"
              checked={useHybridExtractor}
              onChange={(e) => setUseHybridExtractor(e.target.checked)}
            />
          </div>

          <div className="settings-row">
            <span>Admission Quality Gate:</span>
            <input
              type="checkbox"
              checked={enableAdmissionGate}
              onChange={(e) => setEnableAdmissionGate(e.target.checked)}
            />
          </div>

          <div className="settings-row">
            <span>Force Mock LLM (Fast Test):</span>
            <input
              type="checkbox"
              checked={useFakeLlm}
              onChange={(e) => setUseFakeLlm(e.target.checked)}
            />
          </div>
        </div>
      )}

      {/* Messages Scroll Area */}
      <div className="chat-scroll-area" ref={scrollRef}>
        <div className="chat-max-width">
          {messages.length === 0 ? (
            <div className="chat-empty-state">
              <div className="empty-icon-wrapper">
                <Brain size={24} />
              </div>
              <h2 className="empty-title">Adaptive Long-Term Memory</h2>
              <p className="empty-subtitle">
                Test how conversational assertions are dynamically scored,
                admitted, consolidated, and retrieved under token budgets.
              </p>

              <div className="prompt-suggestions">
                {PRESET_SUGGESTIONS.map((suggestion, i) => (
                  <button
                    key={i}
                    className="suggestion-chip"
                    onClick={() => setPrompt(suggestion)}
                  >
                    "{suggestion}"
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`chat-message ${msg.sender}`}>
                {msg.sender === 'user' ? (
                  <div className="message-bubble-user">{msg.content}</div>
                ) : (
                  <div className="message-card-assistant">
                    <div className="assistant-text">{msg.content}</div>

                    <div className="assistant-footer">
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.5rem',
                        }}
                      >
                        {msg.memory_feedback && (
                          <span className="memory-feedback-badge">
                            <CheckCircle2 size={11} />
                            <span>{msg.memory_feedback}</span>
                          </span>
                        )}

                        {msg.duration_ms != null && (
                          <span
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.25rem',
                            }}
                          >
                            <Clock size={11} />
                            <span>{msg.duration_ms.toFixed(1)} ms</span>
                          </span>
                        )}
                      </div>

                      {msg.trace_id && (
                        <button
                          className="view-evidence-link"
                          onClick={() => {
                            onSelectTraceId(msg.trace_id!);
                            if (!isEvidenceOpen) onToggleEvidence();
                          }}
                          title="View 5-stage memory evidence flow"
                        >
                          <span>Evidence Flow</span>
                          <ArrowRight size={12} />
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}

          {isLoading && (
            <div className="chat-message assistant">
              <div
                className="message-card-assistant"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  color: 'var(--text-muted)',
                }}
              >
                <Sparkles size={16} className="animate-spin" />
                <span>Processing prompt through memory pipeline...</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Container */}
      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-box">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            rows={1}
            value={prompt}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder="Type a message or fact to remember..."
            disabled={isLoading}
          />
          <div className="input-actions-bar">
            <span className="input-hint">Enter to send, Shift+Enter for newline</span>
            <button
              type="submit"
              className="send-button"
              disabled={isLoading || !prompt.trim()}
              title="Send message"
            >
              <Send size={14} />
            </button>
          </div>
        </form>
      </div>
    </main>
  );
};

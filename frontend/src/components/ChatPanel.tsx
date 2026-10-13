import React, { useState } from 'react';
import {
  Send,
  SlidersHorizontal,
  Bot,
  Clock,
  Sparkles,
} from 'lucide-react';
import type { ChatMessage, ChatRequest } from '../types';

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (prompt: string, config: Partial<ChatRequest>) => Promise<void>;
  isLoading: boolean;
  activeTraceId: string | null;
  onSelectTraceId: (traceId: string) => void;
  onSelectMemoryId?: (memoryId: string) => void;
}

const PRESET_PROMPTS = [
  'I currently work as a senior backend engineer at Stripe. Please remember this.',
  'Where do I work now?',
  'I transitioned from OpenAI to Stripe in 2024.',
  'Where did I work previously?',
];

export const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  onSendMessage,
  isLoading,
  activeTraceId,
  onSelectTraceId,
  onSelectMemoryId,
}) => {
  const [inputPrompt, setInputPrompt] = useState('');
  const [showConfig, setShowConfig] = useState(false);

  // Advanced pipeline settings
  const [storageBudget, setStorageBudget] = useState(150);
  const [maxContextTokens, setMaxContextTokens] = useState(256);
  const [useHybridExtractor, setUseHybridExtractor] = useState(false);
  const [useFakeLlm, setUseFakeLlm] = useState(false);
  const [enableAdmissionGate, setEnableAdmissionGate] = useState(true);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputPrompt.trim() || isLoading) return;

    onSendMessage(inputPrompt.trim(), {
      storage_budget: storageBudget,
      max_context_tokens: maxContextTokens,
      use_hybrid_extractor: useHybridExtractor,
      use_fake_llm: useFakeLlm,
      enable_admission_gate: enableAdmissionGate,
    });
    setInputPrompt('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="chat-pane">
      <div className="pane-header">
        <h2>
          <Bot size={16} style={{ color: 'var(--accent-primary)' }} />
          <span>Interactive Chat</span>
          {activeTraceId && (
            <span
              className="trace-badge"
              style={{ fontSize: '0.65rem' }}
              title="Active trace"
            >
              Trace: {activeTraceId.slice(0, 8)}...
            </span>
          )}
        </h2>
        <button
          onClick={() => setShowConfig(!showConfig)}
          className="preset-chip"
          style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}
          title="Pipeline parameters and model settings"
        >
          <SlidersHorizontal size={12} />
          <span>Config</span>
        </button>
      </div>

      {showConfig && (
        <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-primary)', borderBottom: '1px solid var(--border-subtle)' }}>
          <div className="config-drawer">
            <div className="config-item">
              <span>Storage Token Budget:</span>
              <input
                type="number"
                min={20}
                max={2000}
                value={storageBudget}
                onChange={(e) => setStorageBudget(Number(e.target.value))}
                style={{ width: '70px', padding: '0.15rem 0.4rem', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'white', borderRadius: '4px' }}
              />
            </div>

            <div className="config-item">
              <span>Prompt Context Tokens:</span>
              <input
                type="number"
                min={32}
                max={1024}
                value={maxContextTokens}
                onChange={(e) => setMaxContextTokens(Number(e.target.value))}
                style={{ width: '70px', padding: '0.15rem 0.4rem', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'white', borderRadius: '4px' }}
              />
            </div>

            <div className="config-item">
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={useHybridExtractor}
                  onChange={(e) => setUseHybridExtractor(e.target.checked)}
                />
                <span>Use Hybrid SLM Extractor</span>
              </label>
            </div>

            <div className="config-item">
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={enableAdmissionGate}
                  onChange={(e) => setEnableAdmissionGate(e.target.checked)}
                />
                <span>Enable Admission Gate</span>
              </label>
            </div>

            <div className="config-item">
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={useFakeLlm}
                  onChange={(e) => setUseFakeLlm(e.target.checked)}
                />
                <span>Force Fake LLM (Fast Test)</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="messages-list">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <Sparkles size={28} style={{ opacity: 0.3 }} />
            <p style={{ fontSize: '0.85rem' }}>Send a prompt to test dynamic memory extraction, allocation, and retrieval.</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`message-item ${msg.sender}`}>
              <div className="message-bubble">{msg.content}</div>

              <div className="message-meta">
                <span>{new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>

                {msg.duration_ms != null && (
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem' }}>
                    <Clock size={10} />
                    {msg.duration_ms.toFixed(1)} ms
                  </span>
                )}

                {msg.trace_id && (
                  <button
                    onClick={() => onSelectTraceId(msg.trace_id!)}
                    className="trace-badge"
                    title="Load execution trace for this turn"
                  >
                    {msg.trace_id.slice(0, 10)}...
                  </button>
                )}
              </div>

              {msg.cited_memory_ids && msg.cited_memory_ids.length > 0 && (
                <div className="citations-list">
                  <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Retrieved memories:</span>
                  {msg.cited_memory_ids.map((cid) => (
                    <button
                      key={cid}
                      onClick={() => onSelectMemoryId && onSelectMemoryId(cid)}
                      className="citation-tag"
                      style={{ cursor: 'pointer' }}
                      title="Inspect memory in explorer"
                    >
                      {cid.slice(0, 8)}...
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="message-item assistant">
            <div className="message-bubble" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="node-status-indicator status-running" />
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                Executing 16-stage memory pipeline...
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Quick Presets */}
      <div className="presets-container">
        {PRESET_PROMPTS.map((prompt, i) => (
          <button
            key={i}
            className="preset-chip"
            onClick={() => setInputPrompt(prompt)}
            disabled={isLoading}
          >
            {prompt.length > 35 ? `${prompt.slice(0, 35)}...` : prompt}
          </button>
        ))}
      </div>

      {/* Input Area */}
      <form className="chat-input-area" onSubmit={handleSubmit}>
        <div className="input-row">
          <textarea
            className="chat-input"
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter conversational prompt (e.g. 'I work at Stripe now')..."
            rows={1}
            disabled={isLoading}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={isLoading || !inputPrompt.trim()}
            title="Send prompt"
          >
            <Send size={15} />
          </button>
        </div>
      </form>
    </div>
  );
};

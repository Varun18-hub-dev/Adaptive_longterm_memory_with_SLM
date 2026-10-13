import React, { useState } from 'react';
import {
  Plus,
  MessageSquare,
  RotateCcw,
  Database,
  ChevronLeft,
  Brain,
  Trash2,
} from 'lucide-react';
import type { HealthResponse } from '../types';

export interface SessionThread {
  sessionId: string;
  title: string;
  lastUpdated: number;
}

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  currentSessionId: string;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  onResetSession: () => void;
  onOpenMemoryExplorer: () => void;
  health: HealthResponse | null;
  sessions: SessionThread[];
  onDeleteSessionThread?: (sessionId: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onToggle,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onResetSession,
  onOpenMemoryExplorer,
  health,
  sessions,
  onDeleteSessionThread,
}) => {
  const [customSessionInput, setCustomSessionInput] = useState('');
  const [showSessionInput, setShowSessionInput] = useState(false);

  const handleCreateCustomSession = (e: React.FormEvent) => {
    e.preventDefault();
    if (customSessionInput.trim()) {
      onSelectSession(customSessionInput.trim());
      setCustomSessionInput('');
      setShowSessionInput(false);
    }
  };

  return (
    <aside
      className={`left-sidebar ${isOpen ? '' : 'collapsed'}`}
      aria-label="Conversation navigation"
    >
      <div className="sidebar-header">
        <div className="app-title-area">
          <div className="app-icon">
            <Brain size={18} />
          </div>
          <div>
            <div className="app-title">Adaptive Memory</div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              {health?.status === 'healthy' ? 'SLM Online' : 'Connecting...'}
            </div>
          </div>
        </div>
        <button
          onClick={onToggle}
          className="icon-btn"
          title="Collapse sidebar"
          aria-label="Collapse sidebar"
        >
          <ChevronLeft size={16} />
        </button>
      </div>

      <button className="new-chat-btn" onClick={onNewChat}>
        <Plus size={16} />
        <span>New Chat</span>
      </button>

      <div className="recent-sessions-list">
        <div className="section-label">Conversations</div>

        {sessions.length === 0 ? (
          <div style={{ padding: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            No recent sessions yet.
          </div>
        ) : (
          sessions.map((thread) => {
            const isActive = thread.sessionId === currentSessionId;
            return (
              <div
                key={thread.sessionId}
                className={`session-item ${isActive ? 'active' : ''}`}
                onClick={() => onSelectSession(thread.sessionId)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') onSelectSession(thread.sessionId);
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', overflow: 'hidden' }}>
                  <MessageSquare size={13} style={{ flexShrink: 0, opacity: isActive ? 1 : 0.6 }} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {thread.title}
                  </span>
                </div>

                {onDeleteSessionThread && sessions.length > 1 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSessionThread(thread.sessionId);
                    }}
                    style={{ padding: '0.2rem', opacity: 0.4 }}
                    title="Remove thread"
                  >
                    <Trash2 size={11} />
                  </button>
                )}
              </div>
            );
          })
        )}

        {showSessionInput ? (
          <form onSubmit={handleCreateCustomSession} style={{ padding: '0.4rem' }}>
            <input
              type="text"
              value={customSessionInput}
              onChange={(e) => setCustomSessionInput(e.target.value)}
              placeholder="session_id"
              style={{
                width: '100%',
                padding: '0.35rem 0.5rem',
                background: 'var(--bg-card)',
                border: '1px solid var(--border-strong)',
                borderRadius: '4px',
                fontSize: '0.75rem',
              }}
              autoFocus
            />
          </form>
        ) : (
          <button
            onClick={() => setShowSessionInput(true)}
            style={{
              padding: '0.35rem 0.5rem',
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
              textAlign: 'left',
            }}
          >
            + Custom Session ID
          </button>
        )}
      </div>

      <div className="sidebar-footer">
        <button className="footer-action-btn" onClick={onOpenMemoryExplorer}>
          <Database size={14} />
          <span>Memory Stores Explorer</span>
        </button>

        <button
          className="footer-action-btn"
          onClick={onResetSession}
          style={{ color: 'var(--status-danger)' }}
          title="Clear memories for active session"
        >
          <RotateCcw size={14} />
          <span>Reset Session Memory</span>
        </button>
      </div>
    </aside>
  );
};

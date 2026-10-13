import React from 'react';
import {
  Brain,
  RotateCcw,
  RefreshCw,
  GitGraph,
  Database,
  BarChart3,
} from 'lucide-react';
import type { HealthResponse } from '../types';

interface NavbarProps {
  activeTab: 'workspace' | 'explorer' | 'research';
  setActiveTab: (tab: 'workspace' | 'explorer' | 'research') => void;
  sessionId: string;
  setSessionId: (id: string) => void;
  health: HealthResponse | null;
  onRefresh: () => void;
  onResetSession: () => void;
  isResetting: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  sessionId,
  setSessionId,
  health,
  onRefresh,
  onResetSession,
  isResetting,
}) => {
  return (
    <header>
      <div className="app-header">
        <div className="header-brand">
          <div className="brand-icon">
            <Brain size={20} />
          </div>
          <div className="brand-titles">
            <h1>Adaptive Long-Term Memory</h1>
            <p>Dynamic Memory-Budget Allocation under Evolving Information</p>
          </div>
        </div>

        <div className="header-controls">
          {/* Health Pill */}
          <div className={`status-pill ${health?.status === 'healthy' ? 'healthy' : 'offline'}`}>
            <span className="status-dot" />
            <span>
              {health?.status === 'healthy'
                ? `Backend Online (${health.active_sessions_count} sessions)`
                : 'Connecting to API...'}
            </span>
            {health?.qwen_model_path_configured && (
              <span style={{ opacity: 0.8, fontSize: '0.7rem' }}>• Qwen SLM</span>
            )}
          </div>

          {/* Session Selector */}
          <div className="session-selector">
            <span style={{ color: 'var(--text-muted)' }}>Session:</span>
            <input
              type="text"
              className="session-input"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value.trim() || 'default_session')}
              placeholder="session_id"
              title="Active session scope for multi-session isolation"
            />
            <button
              onClick={onResetSession}
              disabled={isResetting}
              className="preset-chip"
              style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', padding: '0.15rem 0.4rem' }}
              title="Reset memory stores for current session"
            >
              <RotateCcw size={12} />
              <span>Reset</span>
            </button>
          </div>

          {/* Refresh Action */}
          <button
            onClick={onRefresh}
            className="preset-chip"
            style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}
            title="Refresh memory store and status"
          >
            <RefreshCw size={13} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button
          className={`nav-tab ${activeTab === 'workspace' ? 'active' : ''}`}
          onClick={() => setActiveTab('workspace')}
        >
          <GitGraph size={16} />
          <span>Execution Trace & Workspace</span>
        </button>

        <button
          className={`nav-tab ${activeTab === 'explorer' ? 'active' : ''}`}
          onClick={() => setActiveTab('explorer')}
        >
          <Database size={16} />
          <span>Memory Stores Explorer</span>
        </button>

        <button
          className={`nav-tab ${activeTab === 'research' ? 'active' : ''}`}
          onClick={() => setActiveTab('research')}
        >
          <BarChart3 size={16} />
          <span>Research Observability & Allocation</span>
        </button>
      </nav>
    </header>
  );
};

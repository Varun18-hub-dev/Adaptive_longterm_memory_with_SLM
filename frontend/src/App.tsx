import React, { useEffect, useState } from 'react';
import { Sidebar, type SessionThread } from './components/Sidebar';
import { ChatWorkspace } from './components/ChatWorkspace';
import { MemoryEvidencePanel } from './components/MemoryEvidencePanel';
import { MemoryExplorer } from './components/MemoryExplorer';
import { MemoryDetailModal } from './components/MemoryDetailModal';
import { api } from './services/api';
import type {
  ChatMessage,
  ChatRequest,
  ExecutionTrace,
  HealthResponse,
  MemoriesResponse,
  MemoryItem,
} from './types';
import { X, Database } from 'lucide-react';

const STORAGE_SESSIONS_KEY = 'alm_recent_sessions';
const STORAGE_MESSAGES_KEY_PREFIX = 'alm_messages_';

export const App: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isEvidenceOpen, setIsEvidenceOpen] = useState(true);
  const [sessionId, setSessionId] = useState<string>('default_session');
  const [sessions, setSessions] = useState<SessionThread[]>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_SESSIONS_KEY);
      if (stored) return JSON.parse(stored);
    } catch {
      // fallback
    }
    return [
      {
        sessionId: 'default_session',
        title: 'Default Conversation',
        lastUpdated: Date.now(),
      },
    ];
  });

  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const stored = localStorage.getItem(`${STORAGE_MESSAGES_KEY_PREFIX}default_session`);
      if (stored) return JSON.parse(stored);
    } catch {
      // fallback
    }
    return [];
  });

  const [activeTrace, setActiveTrace] = useState<ExecutionTrace | null>(null);
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const [showMemoryExplorer, setShowMemoryExplorer] = useState(false);
  const [memoriesData, setMemoriesData] = useState<MemoriesResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  // Sync sessions list to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_SESSIONS_KEY, JSON.stringify(sessions));
    } catch {
      // ignore
    }
  }, [sessions]);

  // Sync current messages to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(
        `${STORAGE_MESSAGES_KEY_PREFIX}${sessionId}`,
        JSON.stringify(messages)
      );
    } catch {
      // ignore
    }
  }, [messages, sessionId]);

  // Initial health check and memory loading
  useEffect(() => {
    loadSessionData(sessionId);
    api.checkHealth().then(setHealth).catch(() => setHealth(null));
  }, [sessionId]);

  const loadSessionData = async (sid: string) => {
    try {
      // Load saved messages for this session
      const stored = localStorage.getItem(`${STORAGE_MESSAGES_KEY_PREFIX}${sid}`);
      if (stored) {
        setMessages(JSON.parse(stored));
      } else {
        setMessages([]);
      }

      const mems = await api.getMemories(sid);
      setMemoriesData(mems);
    } catch {
      // Ignore background loading errors
    }
  };

  const handleSendMessage = async (promptText: string, config: Partial<ChatRequest>) => {
    setIsLoading(true);
    setErrorBanner(null);

    const userMessage: ChatMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      content: promptText,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Update session title if first turn
    setSessions((prev) => {
      const existing = prev.find((s) => s.sessionId === sessionId);
      const title =
        existing && existing.title !== 'New Conversation' && existing.title !== 'Default Conversation'
          ? existing.title
          : promptText.slice(0, 30);

      const filtered = prev.filter((s) => s.sessionId !== sessionId);
      return [{ sessionId, title, lastUpdated: Date.now() }, ...filtered];
    });

    try {
      const chatRes = await api.sendChat({
        prompt: promptText,
        session_id: sessionId,
        ...config,
      });

      // Fetch trace for authentic operational evidence
      let traceData: ExecutionTrace | null = null;
      let memoryFeedback = 'Working memory active';

      if (chatRes.trace_id) {
        try {
          traceData = await api.getTrace(chatRes.trace_id);
          setActiveTrace(traceData);

          // Derive authentic short memory feedback
          const gateEv = traceData.events?.find((e) => e.stage === 'admission_gate');
          const conflictEv = traceData.events?.find((e) => e.stage === 'conflict_resolution');
          const retrievalEv = traceData.events?.find((e) => e.stage === 'memory_retrieval');

          const admitted = Number(gateEv?.outputs?.admitted_count ?? 0);
          const conflicts = Number(conflictEv?.outputs?.conflicts_detected ?? 0);
          const retrieved = Number(retrievalEv?.outputs?.retrieved_count ?? 0);

          if (admitted > 0) {
            memoryFeedback = `${admitted} fact${admitted > 1 ? 's' : ''} remembered`;
          } else if (conflicts > 0) {
            memoryFeedback = 'Fact updated (superseded)';
          } else if (retrieved > 0) {
            memoryFeedback = `${retrieved} memor${retrieved > 1 ? 'ies' : 'y'} recalled`;
          }
        } catch {
          // ignore trace fetch failure
        }
      }

      const assistantMessage: ChatMessage = {
        id: `asst_${Date.now()}`,
        sender: 'assistant',
        content: chatRes.response,
        timestamp: Date.now(),
        trace_id: chatRes.trace_id,
        duration_ms: chatRes.duration_ms,
        cited_memory_ids: chatRes.cited_memory_ids,
        status: chatRes.status,
        memory_feedback: memoryFeedback,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Refresh memory stores
      const updatedMems = await api.getMemories(sessionId);
      setMemoriesData(updatedMems);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Pipeline execution error';
      setErrorBanner(msg);

      const errorMessage: ChatMessage = {
        id: `err_${Date.now()}`,
        sender: 'assistant',
        content: `Error: ${msg}`,
        timestamp: Date.now(),
        status: 'failed',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    const newSessionId = `session_${Date.now().toString(36)}`;
    setSessions((prev) => [
      {
        sessionId: newSessionId,
        title: 'New Conversation',
        lastUpdated: Date.now(),
      },
      ...prev,
    ]);
    setSessionId(newSessionId);
    setMessages([]);
    setActiveTrace(null);
  };

  const handleSelectSession = (sid: string) => {
    setSessionId(sid);
    setActiveTrace(null);
  };

  const handleResetSession = async () => {
    if (!window.confirm(`Reset memories and chat history for session '${sessionId}'?`)) {
      return;
    }
    try {
      await api.resetSession(sessionId);
      setMessages([]);
      setActiveTrace(null);
      localStorage.removeItem(`${STORAGE_MESSAGES_KEY_PREFIX}${sessionId}`);
      const emptyMems = await api.getMemories(sessionId);
      setMemoriesData(emptyMems);
    } catch (err: unknown) {
      setErrorBanner(`Failed to reset session: ${err instanceof Error ? err.message : 'Unknown'}`);
    }
  };

  const handleSelectTraceId = async (traceId: string) => {
    try {
      const trace = await api.getTrace(traceId);
      setActiveTrace(trace);
      setIsEvidenceOpen(true);
    } catch (err: unknown) {
      setErrorBanner(`Could not load trace: ${err instanceof Error ? err.message : 'Unknown'}`);
    }
  };

  const handleDeleteSessionThread = (sid: string) => {
    setSessions((prev) => prev.filter((s) => s.sessionId !== sid));
    localStorage.removeItem(`${STORAGE_MESSAGES_KEY_PREFIX}${sid}`);
    if (sessionId === sid) {
      const remaining = sessions.filter((s) => s.sessionId !== sid);
      if (remaining.length > 0) {
        setSessionId(remaining[0].sessionId);
      } else {
        handleNewChat();
      }
    }
  };

  return (
    <div className="app-layout">
      {/* 1. Left Sidebar */}
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        currentSessionId={sessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onResetSession={handleResetSession}
        onOpenMemoryExplorer={() => setShowMemoryExplorer(true)}
        health={health}
        sessions={sessions}
        onDeleteSessionThread={handleDeleteSessionThread}
      />

      {/* 2. Center Chat Workspace */}
      <ChatWorkspace
        messages={messages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        activeSessionId={sessionId}
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        isEvidenceOpen={isEvidenceOpen}
        onToggleEvidence={() => setIsEvidenceOpen(!isEvidenceOpen)}
        activeTraceId={activeTrace?.trace_id || null}
        onSelectTraceId={handleSelectTraceId}
        health={health}
      />

      {/* 3. Right Memory Evidence Panel */}
      <MemoryEvidencePanel
        isOpen={isEvidenceOpen}
        onClose={() => setIsEvidenceOpen(false)}
        trace={activeTrace}
        onSelectMemoryId={(mid) => setSelectedMemoryId(mid)}
      />

      {/* Error banner */}
      {errorBanner && (
        <div
          style={{
            position: 'absolute',
            bottom: '80px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'var(--status-danger-bg)',
            color: 'var(--status-danger)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '8px',
            padding: '0.5rem 1rem',
            fontSize: '0.8rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            zIndex: 100,
            boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
          }}
        >
          <span>{errorBanner}</span>
          <button onClick={() => setErrorBanner(null)} style={{ fontWeight: 'bold' }}>
            ✕
          </button>
        </div>
      )}

      {/* Memory Stores Explorer Modal */}
      {showMemoryExplorer && (
        <div className="modal-overlay" onClick={() => setShowMemoryExplorer(false)}>
          <div
            className="modal-card"
            style={{ maxWidth: '900px', width: '90%' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Database size={16} style={{ color: 'var(--accent-primary)' }} />
                <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>Memory Stores Explorer</h3>
                <span className="badge badge-active">{sessionId}</span>
              </div>
              <button
                onClick={() => setShowMemoryExplorer(false)}
                className="icon-btn"
                title="Close"
              >
                <X size={16} />
              </button>
            </div>
            <div className="modal-body" style={{ maxHeight: '75vh', overflowY: 'auto' }}>
              <MemoryExplorer
                memoriesData={memoriesData}
                onSelectMemory={(m: MemoryItem) => setSelectedMemoryId(m.id)}
                isLoading={isLoading}
              />
            </div>
          </div>
        </div>
      )}

      {/* Memory Detail Modal */}
      <MemoryDetailModal
        memoryId={selectedMemoryId}
        sessionId={sessionId}
        onClose={() => setSelectedMemoryId(null)}
      />
    </div>
  );
};

export default App;

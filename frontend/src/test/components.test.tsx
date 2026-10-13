import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Sidebar } from '../components/Sidebar';
import { ChatWorkspace } from '../components/ChatWorkspace';
import { MemoryEvidencePanel } from '../components/MemoryEvidencePanel';
import { MemoryExplorer } from '../components/MemoryExplorer';
import type { ChatMessage, ExecutionTrace, MemoriesResponse } from '../types';

describe('Redesigned Chat-First UI Component Tests', () => {
  it('renders Sidebar with conversation threads and new chat action', () => {
    const onSelectSession = vi.fn();
    const onNewChat = vi.fn();
    const onResetSession = vi.fn();

    const mockSessions = [
      {
        sessionId: 'session_1',
        title: 'Work Information',
        lastUpdated: Date.now(),
      },
      {
        sessionId: 'session_2',
        title: 'Project Discussion',
        lastUpdated: Date.now(),
      },
    ];

    render(
      <Sidebar
        isOpen={true}
        onToggle={vi.fn()}
        currentSessionId="session_1"
        onSelectSession={onSelectSession}
        onNewChat={onNewChat}
        onResetSession={onResetSession}
        onOpenMemoryExplorer={vi.fn()}
        health={{
          status: 'healthy',
          timestamp: 1000,
          active_sessions_count: 2,
          qwen_model_path_configured: true,
        }}
        sessions={mockSessions}
      />
    );

    expect(screen.getByText('Adaptive Memory')).toBeDefined();
    expect(screen.getByText('New Chat')).toBeDefined();
    expect(screen.getByText('Work Information')).toBeDefined();
    expect(screen.getByText('Project Discussion')).toBeDefined();

    fireEvent.click(screen.getByText('New Chat'));
    expect(onNewChat).toHaveBeenCalled();

    fireEvent.click(screen.getByText('Project Discussion'));
    expect(onSelectSession).toHaveBeenCalledWith('session_2');
  });

  it('renders ChatWorkspace with messages, memory feedback pill, and submit action', () => {
    const onSendMessage = vi.fn();
    const onToggleEvidence = vi.fn();
    const onSelectTraceId = vi.fn();

    const mockMessages: ChatMessage[] = [
      {
        id: 'msg_1',
        sender: 'user',
        content: 'I work at Stripe.',
        timestamp: 1000,
      },
      {
        id: 'msg_2',
        sender: 'assistant',
        content: 'Understood. I will remember that you work at Stripe.',
        timestamp: 1001,
        trace_id: 'trc_abc123',
        duration_ms: 3.4,
        memory_feedback: '1 fact remembered',
      },
    ];

    render(
      <ChatWorkspace
        messages={mockMessages}
        onSendMessage={onSendMessage}
        isLoading={false}
        activeSessionId="session_1"
        isSidebarOpen={true}
        onToggleSidebar={vi.fn()}
        isEvidenceOpen={false}
        onToggleEvidence={onToggleEvidence}
        activeTraceId="trc_abc123"
        onSelectTraceId={onSelectTraceId}
        health={{
          status: 'healthy',
          timestamp: 1000,
          active_sessions_count: 1,
          qwen_model_path_configured: false,
        }}
      />
    );

    expect(screen.getByText('I work at Stripe.')).toBeDefined();
    expect(screen.getByText('Understood. I will remember that you work at Stripe.')).toBeDefined();
    expect(screen.getByText('1 fact remembered')).toBeDefined();
    expect(screen.getByText('3.4 ms')).toBeDefined();

    // Click "Evidence Flow" link on assistant message
    const evidenceLink = screen.getByText('Evidence Flow');
    fireEvent.click(evidenceLink);
    expect(onSelectTraceId).toHaveBeenCalledWith('trc_abc123');
  });

  it('renders MemoryEvidencePanel with 5 connected groups and expands details', () => {
    const mockTrace: ExecutionTrace = {
      schema_version: '1.0.0',
      trace_id: 'trc_xyz789',
      session_id: 'default_session',
      prompt: 'I live in Zurich.',
      status: 'completed',
      start_time: 1000,
      total_duration_ms: 4.8,
      events: [
        {
          event_id: 'evt_1',
          trace_id: 'trc_xyz789',
          sequence_number: 1,
          stage: 'prompt_received',
          status: 'completed',
          start_time: 1000,
          duration_ms: 0.1,
          summary: 'Prompt validated',
          candidate_ids: [],
          memory_ids: [],
          inputs: {},
          outputs: { prompt_char_length: 17, is_question: false },
          metadata: {},
        },
        {
          event_id: 'evt_2',
          trace_id: 'trc_xyz789',
          sequence_number: 2,
          stage: 'candidate_extraction',
          status: 'completed',
          start_time: 1000,
          duration_ms: 1.2,
          summary: 'Extracted candidate',
          candidate_ids: ['cand_1'],
          memory_ids: [],
          inputs: {},
          outputs: {
            candidate_count: 1,
            candidates: [
              {
                id: 'cand_1',
                content: 'I live in Zurich.',
                memory_type: 'episodic',
                token_cost: 6,
              },
            ],
          },
          metadata: {},
        },
        {
          event_id: 'evt_4',
          trace_id: 'trc_xyz789',
          sequence_number: 4,
          stage: 'utility_scoring',
          status: 'completed',
          start_time: 1000,
          duration_ms: 0.2,
          summary: 'Scored utility',
          candidate_ids: ['cand_1'],
          memory_ids: [],
          inputs: {},
          outputs: {
            scored_candidates: [
              {
                id: 'cand_1',
                utility_score: 0.782,
                recency: 1.0,
                stability: 0.8,
                importance: 0.7,
              },
            ],
          },
          metadata: {},
        },
        {
          event_id: 'evt_5',
          trace_id: 'trc_xyz789',
          sequence_number: 5,
          stage: 'admission_gate',
          status: 'completed',
          start_time: 1000,
          duration_ms: 0.1,
          summary: 'Gate evaluated',
          candidate_ids: ['cand_1'],
          memory_ids: [],
          inputs: {},
          outputs: {
            admitted_count: 1,
            rejected_count: 0,
          },
          decision: 'admitted',
          explanation: 'Score exceeds threshold',
          metadata: {},
        },
      ],
      correlations: {},
      metadata: {},
    };

    render(
      <MemoryEvidencePanel
        isOpen={true}
        onClose={vi.fn()}
        trace={mockTrace}
      />
    );

    // Verify 5 plain-language groups appear
    expect(screen.getByText('Understand Prompt')).toBeDefined();
    expect(screen.getByText('Evaluate Candidate Facts')).toBeDefined();
    expect(screen.getByText('Update Memory')).toBeDefined();
    expect(screen.getByText('Retrieve Relevant Memories')).toBeDefined();
    expect(screen.getByText('Generate Response')).toBeDefined();

    // Verify structured candidate evidence is shown
    expect(screen.getByText('"I live in Zurich."')).toBeDefined();
    expect(screen.getByText(/Utility: 0.782/)).toBeDefined();
  });

  it('renders MemoryExplorer with token accounting', () => {
    const mockMemories: MemoriesResponse = {
      session_id: 'user_session',
      storage_budget: 150,
      used_tokens: 15,
      remaining_tokens: 135,
      episodic_store: [
        {
          id: 'mem_1',
          content: 'User works at Google.',
          memory_type: 'episodic',
          status: 'active',
          utility_score: 0.8,
          importance: 0.7,
          recency: 1.0,
          frequency: 1,
          stability: 0.5,
          confidence: 0.9,
          token_cost: 6,
          created_at: 1000,
          last_accessed: 1000,
          provenance: [],
          metadata: {},
        },
      ],
      semantic_store: [],
      archive_store: [],
      rejected_candidates: [],
    };

    render(
      <MemoryExplorer
        memoriesData={mockMemories}
        onSelectMemory={vi.fn()}
        isLoading={false}
      />
    );

    expect(screen.getByText(/Used: 15 tokens/)).toBeDefined();
    expect(screen.getByText('User works at Google.')).toBeDefined();
  });
});

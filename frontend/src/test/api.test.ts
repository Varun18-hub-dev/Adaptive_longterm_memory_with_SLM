import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api } from '../services/api';
import type { ChatResponse, ExecutionTrace, MemoriesResponse } from '../types';

describe('Frontend API Client Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('correctly requests health endpoint', async () => {
    const mockHealth = {
      status: 'healthy',
      timestamp: 1791903000,
      active_sessions_count: 2,
      qwen_model_path_configured: false,
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    } as Response);

    const res = await api.checkHealth();
    expect(res.status).toBe('healthy');
    expect(res.active_sessions_count).toBe(2);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/health'),
      expect.anything()
    );
  });

  it('correctly posts chat request and parses response', async () => {
    const mockChatRes: ChatResponse = {
      response: 'You work at Stripe.',
      trace_id: 'trc_12345',
      session_id: 'test_session',
      status: 'completed',
      duration_ms: 2.5,
      cited_memory_ids: ['mem_01'],
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockChatRes,
    } as Response);

    const res = await api.sendChat({
      prompt: 'Where do I work?',
      session_id: 'test_session',
      storage_budget: 150,
    });

    expect(res.response).toBe('You work at Stripe.');
    expect(res.trace_id).toBe('trc_12345');
    expect(res.cited_memory_ids).toEqual(['mem_01']);
  });

  it('handles FastAPI HTTPException detail parsing on error', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => ({ detail: 'Prompt cannot be empty.' }),
    } as Response);

    await expect(api.sendChat({ prompt: '' })).rejects.toThrow('Prompt cannot be empty.');
  });

  it('fetches complete trace by trace_id', async () => {
    const mockTrace: Partial<ExecutionTrace> = {
      schema_version: '1.0.0',
      trace_id: 'trc_test_999',
      session_id: 'default_session',
      status: 'completed',
      events: [],
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockTrace,
    } as Response);

    const res = await api.getTrace('trc_test_999');
    expect(res.trace_id).toBe('trc_test_999');
    expect(res.status).toBe('completed');
  });

  it('fetches session memories with token accounting', async () => {
    const mockMemories: Partial<MemoriesResponse> = {
      session_id: 'session_a',
      storage_budget: 150,
      used_tokens: 30,
      remaining_tokens: 120,
      episodic_store: [],
      semantic_store: [],
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockMemories,
    } as Response);

    const res = await api.getMemories('session_a');
    expect(res.storage_budget).toBe(150);
    expect(res.used_tokens).toBe(30);
    expect(res.remaining_tokens).toBe(120);
  });
});

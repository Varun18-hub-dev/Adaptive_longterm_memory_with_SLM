/**
 * Adaptive Long-Term Memory for Language Models
 * Phase 7: REST API Client for FastAPI Backend
 */

import type {
  ChatRequest,
  ChatResponse,
  ExecutionTrace,
  HealthResponse,
  MemoriesResponse,
  MemoryDetailResponse,
  TraceEvent,
} from '../types';

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string) || 'http://127.0.0.1:8000';

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  try {
    const res = await fetch(url, { ...options, headers });
    if (!res.ok) {
      let errorMsg = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson && errorJson.detail) {
          errorMsg = typeof errorJson.detail === 'string'
            ? errorJson.detail
            : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Fallback to HTTP text
      }
      throw new ApiError(errorMsg, res.status);
    }
    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    const message = err instanceof Error ? err.message : 'Unknown network error';
    throw new ApiError(`Network error connecting to backend: ${message}`, 0);
  }
}

export const api = {
  getBaseUrl(): string {
    return API_BASE_URL;
  },

  async checkHealth(): Promise<HealthResponse> {
    return request<HealthResponse>('/api/health');
  },

  async sendChat(req: ChatRequest): Promise<ChatResponse> {
    return request<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  async getTrace(traceId: string): Promise<ExecutionTrace> {
    return request<ExecutionTrace>(`/api/traces/${encodeURIComponent(traceId)}`);
  },

  async getTraceEvents(traceId: string): Promise<{ trace_id: string; event_count: number; events: TraceEvent[] }> {
    return request<{ trace_id: string; event_count: number; events: TraceEvent[] }>(
      `/api/traces/${encodeURIComponent(traceId)}/events`
    );
  },

  async getMemories(sessionId: string = 'default_session'): Promise<MemoriesResponse> {
    return request<MemoriesResponse>(
      `/api/memories?session_id=${encodeURIComponent(sessionId)}`
    );
  },

  async getMemoryDetail(
    memoryId: string,
    sessionId: string = 'default_session'
  ): Promise<MemoryDetailResponse> {
    return request<MemoryDetailResponse>(
      `/api/memories/${encodeURIComponent(memoryId)}?session_id=${encodeURIComponent(sessionId)}`
    );
  },

  async resetSession(sessionId: string = 'default_session'): Promise<{ status: string; message: string }> {
    return request<{ status: string; message: string }>('/api/sessions/reset', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId }),
    });
  },
};

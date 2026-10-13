/**
 * Adaptive Long-Term Memory for Language Models
 * Phase 7: Frontend TypeScript Data Models & Contract Definitions
 */

export type TraceStage =
  | 'prompt_received'
  | 'candidate_extraction'
  | 'candidate_normalization'
  | 'utility_scoring'
  | 'admission_gate'
  | 'deduplication'
  | 'conflict_resolution'
  | 'memory_consolidation'
  | 'lifecycle_transitions'
  | 'adaptive_allocation'
  | 'persistence'
  | 'memory_retrieval'
  | 'reranking'
  | 'context_assembly'
  | 'slm_generation'
  | 'response_completed';

export type EventStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'skipped'
  | 'failed'
  | 'warning';

export type OverallStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface StageMetadata {
  id: TraceStage;
  number: number;
  label: string;
  category: 'Ingestion' | 'Lifecycle & Allocation' | 'Retrieval & Inference';
  description: string;
}

export const STAGE_DEFINITIONS: Record<TraceStage, StageMetadata> = {
  prompt_received: {
    id: 'prompt_received',
    number: 1,
    label: 'Prompt Received',
    category: 'Ingestion',
    description: 'User prompt ingested and validated for memory pipeline execution.',
  },
  candidate_extraction: {
    id: 'candidate_extraction',
    number: 2,
    label: 'Candidate Extraction',
    category: 'Ingestion',
    description: 'Extract raw atomic assertion candidates via heuristic or SLM extractor.',
  },
  candidate_normalization: {
    id: 'candidate_normalization',
    number: 3,
    label: 'Candidate Normalization',
    category: 'Ingestion',
    description: 'Deduplicate identical raw tokens, clean text, and initialize metadata.',
  },
  utility_scoring: {
    id: 'utility_scoring',
    number: 4,
    label: 'Utility Scoring',
    category: 'Ingestion',
    description: 'Compute utility scores U(m) using recency, relevance, frequency, and stability weights.',
  },
  admission_gate: {
    id: 'admission_gate',
    number: 5,
    label: 'Admission Gate',
    category: 'Ingestion',
    description: 'Filter candidates using confidence and importance admission thresholds.',
  },
  deduplication: {
    id: 'deduplication',
    number: 6,
    label: 'Deduplication',
    category: 'Lifecycle & Allocation',
    description: 'Soft-merge semantically similar candidates to avoid redundancy.',
  },
  conflict_resolution: {
    id: 'conflict_resolution',
    number: 7,
    label: 'Conflict Resolution',
    category: 'Lifecycle & Allocation',
    description: 'Detect temporal contradictions and supersede obsolete assertions.',
  },
  memory_consolidation: {
    id: 'memory_consolidation',
    number: 8,
    label: 'Memory Consolidation',
    category: 'Lifecycle & Allocation',
    description: 'Promote frequently referenced episodic memories into durable semantic knowledge.',
  },
  lifecycle_transitions: {
    id: 'lifecycle_transitions',
    number: 9,
    label: 'Lifecycle Transitions',
    category: 'Lifecycle & Allocation',
    description: 'Advance TTL counters, transition temporary to active, and archive expired entries.',
  },
  adaptive_allocation: {
    id: 'adaptive_allocation',
    number: 10,
    label: 'Adaptive Allocation',
    category: 'Lifecycle & Allocation',
    description: 'Dynamically rebalance token budgets across episodic and semantic memory pools.',
  },
  persistence: {
    id: 'persistence',
    number: 11,
    label: 'Persistence',
    category: 'Lifecycle & Allocation',
    description: 'Save updated memory stores and audit events to durable storage.',
  },
  memory_retrieval: {
    id: 'memory_retrieval',
    number: 12,
    label: 'Memory Retrieval',
    category: 'Retrieval & Inference',
    description: 'Search episodic and semantic stores with query intent and temporal validity filters.',
  },
  reranking: {
    id: 'reranking',
    number: 13,
    label: 'Reranking',
    category: 'Retrieval & Inference',
    description: 'Score retrieved candidates by relevance, utility, and apply margin pruning.',
  },
  context_assembly: {
    id: 'context_assembly',
    number: 14,
    label: 'Context Assembly',
    category: 'Retrieval & Inference',
    description: 'Fit top-ranked memories within prompt context token budget without duplicates.',
  },
  slm_generation: {
    id: 'slm_generation',
    number: 15,
    label: 'SLM Generation',
    category: 'Retrieval & Inference',
    description: 'Generate response via local Qwen GGUF or configured model adapter.',
  },
  response_completed: {
    id: 'response_completed',
    number: 16,
    label: 'Response Completed',
    category: 'Retrieval & Inference',
    description: 'Finalize output response, audit correlations, and record total elapsed latency.',
  },
};

export interface TraceEvent {
  event_id: string;
  trace_id: string;
  sequence_number: number;
  stage: TraceStage;
  status: EventStatus;
  start_time: number;
  end_time?: number | null;
  duration_ms?: number | null;
  summary: string;
  candidate_ids: string[];
  memory_ids: string[];
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  decision?: string | null;
  explanation?: string | null;
  error_details?: Record<string, unknown> | null;
  metadata: Record<string, unknown>;
}

export interface ExecutionTrace {
  schema_version: string;
  trace_id: string;
  session_id: string;
  prompt: string;
  status: OverallStatus;
  start_time: number;
  end_time?: number | null;
  total_duration_ms?: number | null;
  events: TraceEvent[];
  response?: string | null;
  error?: string | null;
  correlations: {
    candidate_ids?: string[];
    stored_memory_ids?: string[];
    retrieved_memory_ids?: string[];
    context_memory_ids?: string[];
    [key: string]: unknown;
  };
  metadata: Record<string, unknown>;
}

export interface MemoryProvenance {
  session_id: string;
  timestamp: number;
  turn_index: number;
  source_text: string;
}

export interface MemoryItem {
  id: string;
  content: string;
  memory_type: 'episodic' | 'semantic';
  status: 'active' | 'temporary' | 'archived' | 'superseded';
  utility_score: number;
  importance: number;
  recency: number;
  frequency: number;
  stability: number;
  confidence: number;
  token_cost: number;
  created_at: number;
  last_accessed: number;
  provenance: MemoryProvenance[];
  supersedes_id?: string | null;
  superseded_by_id?: string | null;
  valid_from?: number | null;
  valid_until?: number | null;
  metadata: Record<string, unknown>;
}

export interface RejectedCandidateRecord {
  memory: MemoryItem;
  decision: string;
  reason: string;
}

export interface MemoriesResponse {
  session_id: string;
  storage_budget: number;
  used_tokens: number;
  remaining_tokens: number;
  episodic_store: MemoryItem[];
  semantic_store: MemoryItem[];
  archive_store: MemoryItem[];
  rejected_candidates: RejectedCandidateRecord[];
}

export interface AuditLogEntry {
  event_type: string;
  target_memory_id: string;
  related_memory_ids: string[];
  timestamp: number;
  session_id: string;
  details: Record<string, unknown>;
}

export interface MemoryDetailResponse {
  session_id: string;
  store_type: 'episodic' | 'semantic' | 'archive' | 'unknown';
  memory: MemoryItem;
  audit_history: AuditLogEntry[];
}

export interface ChatRequest {
  prompt: string;
  session_id?: string;
  storage_budget?: number;
  max_context_tokens?: number;
  use_fake_llm?: boolean;
  use_hybrid_extractor?: boolean;
  enable_admission_gate?: boolean;
}

export interface ChatResponse {
  response: string;
  trace_id: string;
  session_id: string;
  status: string;
  duration_ms: number;
  cited_memory_ids: string[];
}

export interface HealthResponse {
  status: string;
  timestamp: number;
  active_sessions_count: number;
  qwen_model_path_configured: boolean;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  content: string;
  timestamp: number;
  trace_id?: string;
  duration_ms?: number;
  cited_memory_ids?: string[];
  status?: string;
  memory_feedback?: string;
}

export type EvidenceGroupId =
  | 'understand_prompt'
  | 'evaluate_candidates'
  | 'update_memory'
  | 'retrieve_memories'
  | 'generate_response';

export interface EvidenceGroupDef {
  id: EvidenceGroupId;
  number: number;
  title: string;
  stages: TraceStage[];
  description: string;
}

export const EVIDENCE_GROUPS: EvidenceGroupDef[] = [
  {
    id: 'understand_prompt',
    number: 1,
    title: 'Understand Prompt',
    stages: ['prompt_received'],
    description: 'Validates input prompt and initializes session configuration.',
  },
  {
    id: 'evaluate_candidates',
    number: 2,
    title: 'Evaluate Candidate Facts',
    stages: ['candidate_extraction', 'candidate_normalization', 'utility_scoring', 'admission_gate'],
    description: 'Extracts assertion candidates, scores utility U(m), and applies admission quality gating.',
  },
  {
    id: 'update_memory',
    number: 3,
    title: 'Update Memory',
    stages: ['deduplication', 'conflict_resolution', 'memory_consolidation', 'lifecycle_transitions', 'adaptive_allocation', 'persistence'],
    description: 'Deduplicates, links temporal conflicts, consolidates recurring facts, and dynamically allocates token budgets.',
  },
  {
    id: 'retrieve_memories',
    number: 4,
    title: 'Retrieve Relevant Memories',
    stages: ['memory_retrieval', 'reranking', 'context_assembly'],
    description: 'Searches stores for assertions matching query intent, reranks, and fits within token budget.',
  },
  {
    id: 'generate_response',
    number: 5,
    title: 'Generate Response',
    stages: ['slm_generation', 'response_completed'],
    description: 'Generates assistant response with context-grounded reasoning and records total execution latency.',
  },
];

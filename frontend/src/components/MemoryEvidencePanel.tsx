import React, { useState } from 'react';
import {
  X,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  MinusCircle,
  ChevronDown,
  ChevronRight,
  Code2,
  ExternalLink,
  Layers,
  Sparkles,
} from 'lucide-react';
import {
  EVIDENCE_GROUPS,
  type EvidenceGroupId,
  type ExecutionTrace,
  type TraceEvent,
} from '../types';

interface MemoryEvidencePanelProps {
  isOpen: boolean;
  onClose: () => void;
  trace: ExecutionTrace | null;
  onSelectMemoryId?: (memoryId: string) => void;
}

export const MemoryEvidencePanel: React.FC<MemoryEvidencePanelProps> = ({
  isOpen,
  onClose,
  trace,
  onSelectMemoryId,
}) => {
  const [expandedGroup, setExpandedGroup] = useState<EvidenceGroupId | null>(
    'evaluate_candidates'
  );
  const [showRawJson, setShowRawJson] = useState<Record<string, boolean>>({});

  const toggleGroup = (id: EvidenceGroupId) => {
    setExpandedGroup(expandedGroup === id ? null : id);
  };

  const toggleJson = (id: string) => {
    setShowRawJson((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // Helper to find event
  const getEvent = (stageName: string): TraceEvent | undefined => {
    return trace?.events?.find((e) => e.stage === stageName);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 size={15} style={{ color: 'var(--status-success)' }} />;
      case 'warning':
        return <AlertTriangle size={15} style={{ color: 'var(--status-warning)' }} />;
      case 'failed':
        return <XCircle size={15} style={{ color: 'var(--status-danger)' }} />;
      case 'skipped':
        return <MinusCircle size={15} style={{ color: 'var(--status-neutral)' }} />;
      default:
        return <MinusCircle size={15} style={{ color: 'var(--text-muted)', opacity: 0.4 }} />;
    }
  };

  // Compute summary for each group
  const getGroupSummary = (group: EvidenceGroupId) => {
    if (!trace) return { status: 'skipped', summary: 'No trace available' };

    switch (group) {
      case 'understand_prompt': {
        const ev = getEvent('prompt_received');
        if (!ev) return { status: 'skipped', summary: 'Stage did not execute' };
        const chars = ev.outputs?.prompt_char_length || trace.prompt.length;
        return {
          status: ev.status,
          summary: `Prompt validated (${chars} characters)`,
        };
      }

      case 'evaluate_candidates': {
        const extractEv = getEvent('candidate_extraction');
        const gateEv = getEvent('admission_gate');
        const scoreEv = getEvent('utility_scoring');

        if (!extractEv) return { status: 'skipped', summary: 'No candidates extracted' };

        const candCount = Number(extractEv.outputs?.candidate_count ?? 0);
        const admitted = Number(gateEv?.outputs?.admitted_count ?? 0);
        const rejected = Number(gateEv?.outputs?.rejected_count ?? 0);

        if (candCount === 0) {
          return { status: extractEv.status, summary: '0 candidates extracted (no new facts)' };
        }

        const scoreText =
          scoreEv?.outputs?.scored_candidates &&
          Array.isArray(scoreEv.outputs.scored_candidates) &&
          scoreEv.outputs.scored_candidates[0]?.utility_score != null
            ? ` • Score: ${Number(scoreEv.outputs.scored_candidates[0].utility_score).toFixed(2)}`
            : '';

        const status =
          gateEv?.status === 'failed' || extractEv.status === 'failed'
            ? 'failed'
            : extractEv.status === 'warning'
            ? 'warning'
            : 'completed';

        return {
          status,
          summary: `${candCount} candidate${candCount > 1 ? 's' : ''} • ${admitted} admitted${rejected > 0 ? `, ${rejected} rejected` : ''}${scoreText}`,
        };
      }

      case 'update_memory': {
        const allocEv = getEvent('adaptive_allocation');
        const conflictEv = getEvent('conflict_resolution');
        if (!allocEv) return { status: 'skipped', summary: 'Stores unchanged' };

        const tokens = (allocEv.outputs?.tokens_used as Record<string, number>) || {};
        const budget = (allocEv.outputs?.allocated_budget as Record<string, number>) || {};
        const conflicts = Number(conflictEv?.outputs?.conflicts_detected ?? 0);

        const conflictNote = conflicts > 0 ? ` • ${conflicts} conflict resolved` : '';

        return {
          status: allocEv.status,
          summary: `${tokens.total ?? 0}/${budget.total ?? 150} tokens used${conflictNote}`,
        };
      }

      case 'retrieve_memories': {
        const retrievalEv = getEvent('memory_retrieval');
        const contextEv = getEvent('context_assembly');
        if (!retrievalEv) return { status: 'skipped', summary: 'No memories retrieved' };

        const count = Number(retrievalEv.outputs?.retrieved_count ?? 0);
        const ctxTokens = Number(contextEv?.outputs?.context_token_cost ?? 0);

        if (count === 0) {
          return { status: retrievalEv.status, summary: '0 memories retrieved' };
        }

        return {
          status: retrievalEv.status,
          summary: `${count} memor${count > 1 ? 'ies' : 'y'} retrieved • ${ctxTokens} context tokens`,
        };
      }

      case 'generate_response': {
        const genEv = getEvent('slm_generation');
        const compEv = getEvent('response_completed');
        if (!genEv && !compEv) return { status: 'skipped', summary: 'Pending response' };

        const duration = trace.total_duration_ms != null ? `${trace.total_duration_ms.toFixed(1)} ms` : '-';
        const status = compEv?.status || genEv?.status || 'completed';

        return {
          status,
          summary: `Completed in ${duration}`,
        };
      }
    }
  };

  return (
    <aside
      className={`right-sidebar ${isOpen ? '' : 'collapsed'}`}
      aria-label="Memory evidence details"
    >
      <div className="evidence-header">
        <div className="evidence-title">
          <Sparkles size={16} style={{ color: 'var(--accent-primary)' }} />
          <span>Memory Evidence</span>
          {trace && (
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.65rem',
                color: 'var(--text-muted)',
              }}
            >
              #{trace.trace_id.slice(0, 8)}
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          className="icon-btn"
          title="Close evidence panel"
          aria-label="Close evidence panel"
        >
          <X size={15} />
        </button>
      </div>

      <div className="evidence-scroll-area">
        {!trace ? (
          <div style={{ textAlign: 'center', padding: '2.5rem 1rem', color: 'var(--text-muted)' }}>
            <Layers size={28} style={{ opacity: 0.4, margin: '0 auto 0.5rem' }} />
            <p style={{ fontSize: '0.8rem' }}>No trace active yet.</p>
            <p style={{ fontSize: '0.725rem', marginTop: '0.25rem' }}>
              Send a prompt to inspect live pipeline decisions.
            </p>
          </div>
        ) : (
          <div className="evidence-flow">
            {EVIDENCE_GROUPS.map((group) => {
              const { status, summary } = getGroupSummary(group.id);
              const isExpanded = expandedGroup === group.id;

              return (
                <div
                  key={group.id}
                  className={`evidence-step-card ${isExpanded ? 'active' : ''}`}
                >
                  <div
                    className="step-card-header"
                    onClick={() => toggleGroup(group.id)}
                    style={{ cursor: 'pointer' }}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') toggleGroup(group.id);
                    }}
                  >
                    <div className="step-title-group">
                      <span className="step-number-badge">#{group.number}</span>
                      <span className="step-name">{group.title}</span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      {getStatusIcon(status)}
                      {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    </div>
                  </div>

                  <div className="step-summary-text">{summary}</div>

                  <button
                    className="step-details-btn"
                    onClick={() => toggleGroup(group.id)}
                  >
                    <span>{isExpanded ? 'Hide Details' : 'View Details'}</span>
                  </button>

                  {/* Inline Expanded Structured Evidence */}
                  {isExpanded && (
                    <div className="expanded-evidence-box">
                      {renderGroupStructuredDetails(
                        group.id,
                        trace,
                        onSelectMemoryId,
                        toggleJson,
                        showRawJson
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
};

function renderGroupStructuredDetails(
  groupId: EvidenceGroupId,
  trace: ExecutionTrace,
  onSelectMemoryId?: (id: string) => void,
  toggleJson?: (id: string) => void,
  showRawJson: Record<string, boolean> = {}
) {
  const getEvent = (stageName: string) => trace.events?.find((e) => e.stage === stageName);

  switch (groupId) {
    case 'understand_prompt': {
      const ev = getEvent('prompt_received');
      if (!ev) return <p style={{ color: 'var(--text-muted)' }}>No event recorded.</p>;

      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <div className="evidence-fact-card">
            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Prompt Input:</div>
            <div style={{ color: 'var(--text-primary)' }}>"{trace.prompt}"</div>
          </div>
          <div className="evidence-pill-row">
            <span className="evidence-pill">Chars: {String(ev.outputs?.prompt_char_length ?? trace.prompt.length)}</span>
            <span className="evidence-pill">Question: {Boolean(ev.outputs?.is_question) ? 'Yes' : 'No'}</span>
            <span className="evidence-pill">Session: {trace.session_id}</span>
          </div>
        </div>
      );
    }

    case 'evaluate_candidates': {
      const extractEv = getEvent('candidate_extraction');
      const scoreEv = getEvent('utility_scoring');
      const gateEv = getEvent('admission_gate');

      const candidates = (extractEv?.outputs?.candidates as Array<Record<string, unknown>>) || [];
      const scored = (scoreEv?.outputs?.scored_candidates as Array<Record<string, unknown>>) || [];

      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {Boolean(extractEv?.outputs?.fallback_triggered) ? (
            <div style={{ color: 'var(--status-warning)', fontSize: '0.7rem' }}>
              ⚠️ SLM extractor fallback activated. Used heuristic extractor.
            </div>
          ) : null}

          {candidates.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
              No atomic factual assertions detected in this turn.
            </div>
          ) : (
            candidates.map((cand, idx) => {
              const scoreInfo = scored.find((s) => s.id === cand.id);
              return (
                <div key={idx} className="evidence-fact-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      Candidate Fact
                    </span>
                    <span className="evidence-pill">Cost: {String(cand.token_cost || 0)} tok</span>
                  </div>
                  <div style={{ color: 'var(--text-secondary)' }}>"{String(cand.content || '')}"</div>

                  {scoreInfo && (
                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap', marginTop: '0.2rem' }}>
                      <span className="evidence-pill" style={{ color: 'var(--accent-primary)' }}>
                        Utility: {typeof scoreInfo.utility_score === 'number' ? scoreInfo.utility_score.toFixed(3) : '-'}
                      </span>
                      <span className="evidence-pill">Recency: {String(scoreInfo.recency ?? '-')}</span>
                      <span className="evidence-pill">Stability: {String(scoreInfo.stability ?? '-')}</span>
                      <span className="evidence-pill">Importance: {String(scoreInfo.importance ?? '-')}</span>
                    </div>
                  )}

                  {gateEv && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--status-success)', marginTop: '0.2rem' }}>
                      Gate Decision: {gateEv.decision || 'admitted'} ({gateEv.explanation})
                    </div>
                  )}
                </div>
              );
            })
          )}

          {/* Raw JSON trigger */}
          <button
            onClick={() => toggleJson && toggleJson('eval_candidates')}
            style={{ fontSize: '0.675rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
          >
            <Code2 size={11} />
            <span>{showRawJson['eval_candidates'] ? 'Hide Technical Telemetry' : 'Raw Telemetry'}</span>
          </button>
          {showRawJson['eval_candidates'] && (
            <pre className="json-viewer">{JSON.stringify({ extraction: extractEv, scoring: scoreEv, gate: gateEv }, null, 2)}</pre>
          )}
        </div>
      );
    }

    case 'update_memory': {
      const allocEv = getEvent('adaptive_allocation');
      const conflictEv = getEvent('conflict_resolution');
      const dedupEv = getEvent('deduplication');

      const budget = (allocEv?.outputs?.allocated_budget as Record<string, number>) || {};
      const tokens = (allocEv?.outputs?.tokens_used as Record<string, number>) || {};

      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <div className="evidence-fact-card">
            <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Memory Pool Allocations</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.4rem', marginTop: '0.2rem' }}>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Episodic: </span>
                <span>{tokens.episodic ?? 0} / {budget.episodic ?? '-'} tokens</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Semantic: </span>
                <span>{tokens.semantic ?? 0} / {budget.semantic ?? '-'} tokens</span>
              </div>
            </div>
          </div>

          <div className="evidence-pill-row">
            <span className="evidence-pill">Conflicts: {Number(conflictEv?.outputs?.conflicts_detected ?? 0)}</span>
            <span className="evidence-pill">Distinct Added: {Number(dedupEv?.outputs?.distinct_episodic_added ?? 0)}</span>
            <span className="evidence-pill">Evicted: {Number(allocEv?.outputs?.evicted_count ?? 0)}</span>
          </div>

          <button
            onClick={() => toggleJson && toggleJson('update_mem')}
            style={{ fontSize: '0.675rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
          >
            <Code2 size={11} />
            <span>{showRawJson['update_mem'] ? 'Hide Technical Telemetry' : 'Raw Telemetry'}</span>
          </button>
          {showRawJson['update_mem'] && (
            <pre className="json-viewer">{JSON.stringify({ allocation: allocEv, conflict: conflictEv }, null, 2)}</pre>
          )}
        </div>
      );
    }

    case 'retrieve_memories': {
      const retrievalEv = getEvent('memory_retrieval');
      const contextEv = getEvent('context_assembly');
      const retrieved = (retrievalEv?.outputs?.retrieved_memories as Array<Record<string, unknown>>) || [];

      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {retrieved.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
              No memories were required to answer this prompt.
            </div>
          ) : (
            retrieved.map((m, idx) => (
              <div key={idx} className="evidence-fact-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="evidence-pill" style={{ color: 'var(--accent-primary)' }}>
                    Relevance: {typeof m.relevance_score === 'number' ? m.relevance_score.toFixed(3) : '-'}
                  </span>
                  {Boolean(m.id && onSelectMemoryId) ? (
                    <button
                      onClick={() => onSelectMemoryId?.(String(m.id))}
                      style={{ fontSize: '0.65rem', color: 'var(--accent-primary)', display: 'flex', alignItems: 'center', gap: '0.2rem' }}
                      title="Inspect memory in store"
                    >
                      <span>Store Record</span>
                      <ExternalLink size={10} />
                    </button>
                  ) : null}
                </div>
                <div style={{ color: 'var(--text-secondary)' }}>"{String(m.content || '')}"</div>
              </div>
            ))
          )}

          {contextEv?.outputs?.context_token_cost != null && (
            <div className="evidence-pill-row">
              <span className="evidence-pill">
                Context Tokens: {String(contextEv.outputs.context_token_cost)} / {String(contextEv.outputs.context_budget || 256)}
              </span>
            </div>
          )}

          <button
            onClick={() => toggleJson && toggleJson('retrieval')}
            style={{ fontSize: '0.675rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
          >
            <Code2 size={11} />
            <span>{showRawJson['retrieval'] ? 'Hide Technical Telemetry' : 'Raw Telemetry'}</span>
          </button>
          {showRawJson['retrieval'] && (
            <pre className="json-viewer">{JSON.stringify({ retrieval: retrievalEv, context: contextEv }, null, 2)}</pre>
          )}
        </div>
      );
    }

    case 'generate_response': {
      const genEv = getEvent('slm_generation');
      const compEv = getEvent('response_completed');

      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <div className="evidence-fact-card">
            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Latency Breakdown:</div>
            <div style={{ color: 'var(--text-primary)' }}>
              Total elapsed pipeline duration: {trace.total_duration_ms != null ? `${trace.total_duration_ms.toFixed(2)} ms` : '-'}
            </div>
          </div>
          <div className="evidence-pill-row">
            <span className="evidence-pill">Status: {trace.status}</span>
            <span className="evidence-pill">Adapter: {String(genEv?.inputs?.adapter || 'Configured LLM')}</span>
          </div>

          <button
            onClick={() => toggleJson && toggleJson('gen_response')}
            style={{ fontSize: '0.675rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
          >
            <Code2 size={11} />
            <span>{showRawJson['gen_response'] ? 'Hide Technical Telemetry' : 'Raw Telemetry'}</span>
          </button>
          {showRawJson['gen_response'] && (
            <pre className="json-viewer">{JSON.stringify({ generation: genEv, completed: compEv }, null, 2)}</pre>
          )}
        </div>
      );
    }
  }
}

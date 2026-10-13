import React, { useState } from 'react';
import {
  STAGE_DEFINITIONS,
  type ExecutionTrace,
  type TraceEvent,
  type TraceStage,
} from '../types';
import {
  Layers,
  AlertTriangle,
  Code2,
  ChevronDown,
  ChevronRight,
  Info,
  Target,
  FileText,
} from 'lucide-react';

interface StageInspectorProps {
  trace: ExecutionTrace | null;
  stageId: TraceStage | null;
  onSelectMemoryId?: (memoryId: string) => void;
}

export const StageInspector: React.FC<StageInspectorProps> = ({
  trace,
  stageId,
  onSelectMemoryId,
}) => {
  const [showRawJson, setShowRawJson] = useState(false);

  if (!stageId) {
    return (
      <div className="inspector-pane">
        <div className="inspector-header">
          <h2 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Stage Evidence Inspector</h2>
        </div>
        <div className="empty-chat">
          <Layers size={32} style={{ opacity: 0.4 }} />
          <p style={{ fontSize: '0.85rem' }}>Select any pipeline stage node to inspect runtime evidence.</p>
        </div>
      </div>
    );
  }

  const stageDef = STAGE_DEFINITIONS[stageId];
  const event: TraceEvent | undefined = trace?.events?.find((e) => e.stage === stageId);

  return (
    <div className="inspector-pane">
      <div className="inspector-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="node-number">#{stageDef.number}</span>
          <h2 style={{ fontSize: '0.95rem', fontWeight: 600 }}>{stageDef.label}</h2>
        </div>

        {event && (
          <span
            className={`badge ${
              event.status === 'completed'
                ? 'badge-active'
                : event.status === 'warning'
                ? 'badge-temporary'
                : event.status === 'failed'
                ? 'badge-superseded'
                : 'badge-archived'
            }`}
          >
            {event.status}
          </span>
        )}
      </div>

      <div className="inspector-content">
        {/* Stage Overview Card */}
        <div className="card-panel">
          <div className="card-panel-title">
            <Info size={14} />
            <span>Stage Overview</span>
          </div>

          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            {stageDef.description}
          </p>

          <div className="metric-grid">
            <div className="metric-box">
              <span className="metric-label">Execution Duration</span>
              <span className="metric-val">
                {event?.duration_ms != null ? `${event.duration_ms.toFixed(2)} ms` : event ? '0.00 ms' : 'Not recorded'}
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Category</span>
              <span className="metric-val" style={{ fontSize: '0.75rem' }}>
                {stageDef.category}
              </span>
            </div>
          </div>

          {event?.summary && (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', background: 'var(--bg-primary)', padding: '0.5rem 0.6rem', borderRadius: '6px' }}>
              <strong>Summary:</strong> {event.summary}
            </div>
          )}

          {event?.decision && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
              <Target size={14} style={{ color: 'var(--accent-primary)' }} />
              <span>
                <strong>Decision:</strong>{' '}
                <span className="node-decision-badge">{event.decision}</span>
              </span>
            </div>
          )}

          {event?.explanation && (
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              {event.explanation}
            </p>
          )}
        </div>

        {/* Specialized Stage Evidence */}
        {renderStageSpecificEvidence(stageId, event, onSelectMemoryId)}

        {/* Correlated IDs */}
        {(event?.candidate_ids?.length || 0) > 0 && (
          <div className="card-panel">
            <div className="card-panel-title">
              <FileText size={14} />
              <span>Correlated Candidate IDs ({event!.candidate_ids.length})</span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
              {event!.candidate_ids.map((cid) => (
                <span key={cid} className="citation-tag">
                  {cid.slice(0, 8)}...
                </span>
              ))}
            </div>
          </div>
        )}

        {(event?.memory_ids?.length || 0) > 0 && (
          <div className="card-panel">
            <div className="card-panel-title">
              <FileText size={14} />
              <span>Correlated Memory IDs ({event!.memory_ids.length})</span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
              {event!.memory_ids.map((mid) => (
                <button
                  key={mid}
                  onClick={() => onSelectMemoryId && onSelectMemoryId(mid)}
                  className="citation-tag"
                  style={{ cursor: 'pointer' }}
                  title="Click to inspect memory record"
                >
                  {mid.slice(0, 8)}...
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Error Details if Failed or Warning */}
        {event?.error_details && (
          <div className="card-panel" style={{ borderColor: 'rgba(239, 68, 68, 0.4)' }}>
            <div className="card-panel-title" style={{ color: 'var(--status-danger)' }}>
              <AlertTriangle size={14} />
              <span>Error Details</span>
            </div>
            <pre className="json-viewer">{JSON.stringify(event.error_details, null, 2)}</pre>
          </div>
        )}

        {/* Raw JSON Inspection */}
        <div className="card-panel">
          <button
            onClick={() => setShowRawJson(!showRawJson)}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              width: '100%',
              color: 'var(--text-muted)',
              fontSize: '0.8rem',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Code2 size={14} />
              <span>Raw JSON Telemetry</span>
            </span>
            {showRawJson ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>

          {showRawJson && (
            <pre className="json-viewer">
              {event ? JSON.stringify(event, null, 2) : '// No trace event recorded for this stage'}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
};

function renderStageSpecificEvidence(
  stageId: TraceStage,
  event?: TraceEvent,
  onSelectMemoryId?: (id: string) => void
) {
  if (!event) return null;
  const outputs = event.outputs || {};
  const inputs = event.inputs || {};

  switch (stageId) {
    case 'candidate_extraction': {
      const candidates = (outputs.candidates as Array<Record<string, unknown>>) || [];
      return (
        <div className="card-panel">
          <div className="card-panel-title">Extraction Evidence</div>
          <div className="metric-grid">
            <div className="metric-box">
              <span className="metric-label">Extractor Used</span>
              <span className="metric-val" style={{ fontSize: '0.75rem' }}>
                {(outputs.extractor_used as string) || (inputs.extractor as string) || 'Standard'}
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Candidates Extracted</span>
              <span className="metric-val">{(outputs.candidate_count as number) ?? candidates.length}</span>
            </div>
          </div>

          {outputs.fallback_triggered ? (
            <div style={{ color: 'var(--status-warning)', fontSize: '0.75rem', background: 'var(--status-warning-bg)', padding: '0.4rem', borderRadius: '4px' }}>
              ⚠️ SLM extraction fallback triggered. Heuristic extractor utilized.
            </div>
          ) : null}

          {candidates.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
              {candidates.map((cand, idx) => (
                <div key={(cand.id as string) || idx} className="candidate-card">
                  <div className="candidate-header">
                    <span className="badge badge-episodic">{String(cand.memory_type || 'episodic')}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                      {cand.token_cost != null ? `${cand.token_cost} tokens` : ''}
                    </span>
                  </div>
                  <div style={{ color: 'var(--text-primary)' }}>{String(cand.content || '')}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    case 'utility_scoring': {
      const scored = (outputs.scored_candidates as Array<Record<string, unknown>>) || [];
      return (
        <div className="card-panel">
          <div className="card-panel-title">Utility Scoring Model U(m)</div>
          <div className="formula-box">
            U(m) = w_r·R + w_i·I + w_f·F + w_t·T + w_c·C
          </div>

          {scored.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
              {scored.map((item, idx) => (
                <div key={(item.id as string) || idx} className="candidate-card">
                  <div style={{ fontWeight: 600, color: 'var(--accent-primary)', fontFamily: 'var(--font-mono)' }}>
                    Utility: {typeof item.utility_score === 'number' ? item.utility_score.toFixed(4) : '-'}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-primary)' }}>{String(item.content || '')}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.3rem', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                    <div>Rec: {item.recency != null ? String(item.recency) : '-'}</div>
                    <div>Imp: {item.importance != null ? String(item.importance) : '-'}</div>
                    <div>Stab: {item.stability != null ? String(item.stability) : '-'}</div>
                    <div>Cost: {item.token_cost != null ? String(item.token_cost) : '-'}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    case 'admission_gate': {
      return (
        <div className="card-panel">
          <div className="card-panel-title">Admission Quality Thresholds</div>
          <div className="metric-grid">
            <div className="metric-box">
              <span className="metric-label">Min Importance</span>
              <span className="metric-val">{String(inputs.min_importance ?? '0.20')}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Min Confidence</span>
              <span className="metric-val">{String(inputs.min_confidence ?? '0.30')}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Admitted</span>
              <span className="metric-val" style={{ color: 'var(--status-success)' }}>
                {String(outputs.admitted_count ?? 0)}
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Rejected</span>
              <span className="metric-val" style={{ color: outputs.rejected_count ? 'var(--status-danger)' : 'var(--text-muted)' }}>
                {String(outputs.rejected_count ?? 0)}
              </span>
            </div>
          </div>
        </div>
      );
    }

    case 'adaptive_allocation': {
      const budget = (outputs.allocated_budget as Record<string, number>) || {};
      const tokens = (outputs.tokens_used as Record<string, number>) || {};
      return (
        <div className="card-panel">
          <div className="card-panel-title">Dynamic Budget Allocations</div>
          <div className="metric-grid">
            <div className="metric-box">
              <span className="metric-label">Total Budget</span>
              <span className="metric-val">{String(budget.total ?? inputs.total_budget ?? 150)} tokens</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Tokens Used</span>
              <span className="metric-val">{String(tokens.total ?? 0)} tokens</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Episodic Pool</span>
              <span className="metric-val" style={{ fontSize: '0.75rem' }}>
                {tokens.episodic ?? 0} / {budget.episodic ?? '-'} tokens
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Semantic Pool</span>
              <span className="metric-val" style={{ fontSize: '0.75rem' }}>
                {tokens.semantic ?? 0} / {budget.semantic ?? '-'} tokens
              </span>
            </div>
          </div>
        </div>
      );
    }

    case 'memory_retrieval': {
      const retrieved = (outputs.retrieved_memories as Array<Record<string, unknown>>) || [];
      return (
        <div className="card-panel">
          <div className="card-panel-title">Retrieval Assertions</div>
          <div className="metric-grid">
            <div className="metric-box">
              <span className="metric-label">Historical Query</span>
              <span className="metric-val">{outputs.is_historical_query ? 'Yes' : 'No'}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Retrieved Items</span>
              <span className="metric-val">{String(outputs.retrieved_count ?? retrieved.length)}</span>
            </div>
          </div>

          {retrieved.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
              {retrieved.map((m, idx) => (
                <div key={(m.id as string) || idx} className="candidate-card">
                  <div className="candidate-header">
                    <span className="badge badge-semantic">{String(m.type || 'memory')}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <span style={{ color: 'var(--accent-primary)', fontFamily: 'var(--font-mono)', fontSize: '0.7rem' }}>
                        Relevance: {typeof m.relevance_score === 'number' ? m.relevance_score.toFixed(4) : '-'}
                      </span>
                      {Boolean(m.id && onSelectMemoryId) ? (
                        <button
                          onClick={() => onSelectMemoryId?.(String(m.id))}
                          className="citation-tag"
                          style={{ cursor: 'pointer' }}
                          title="Inspect memory"
                        >
                          View
                        </button>
                      ) : null}
                    </div>
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)' }}>{String(m.content || '')}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    case 'context_assembly': {
      return (
        <div className="card-panel">
          <div className="card-panel-title">Context Assembly</div>
          <div className="metric-grid">
            <div className="metric-box">
              <span className="metric-label">Context Tokens</span>
              <span className="metric-val">
                {String(outputs.context_token_cost ?? 0)} / {String(outputs.context_budget ?? inputs.max_context_tokens ?? 256)}
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Selected Items</span>
              <span className="metric-val">{String(outputs.selected_memories_count ?? 0)}</span>
            </div>
          </div>

          {Boolean(outputs.formatted_context_preview) ? (
            <div style={{ marginTop: '0.5rem' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Assembled Prompt Context:</span>
              <pre className="json-viewer" style={{ whiteSpace: 'pre-wrap', maxHeight: '140px' }}>
                {String(outputs.formatted_context_preview)}
              </pre>
            </div>
          ) : null}
        </div>
      );
    }

    default:
      return null;
  }
}

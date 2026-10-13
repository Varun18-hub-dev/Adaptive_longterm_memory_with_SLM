import React from 'react';
import {
  STAGE_DEFINITIONS,
  type EventStatus,
  type ExecutionTrace,
  type TraceEvent,
  type TraceStage,
} from '../types';
import {
  Clock,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  MinusCircle,
  Zap,
} from 'lucide-react';

interface ExecutionGraphProps {
  trace: ExecutionTrace | null;
  selectedStage: TraceStage | null;
  onSelectStage: (stage: TraceStage) => void;
  isLoading: boolean;
}

const PHASES: Array<{
  id: string;
  name: string;
  stages: TraceStage[];
  color: string;
}> = [
  {
    id: 'ingestion',
    name: '1. Candidate Ingestion & Gating',
    stages: [
      'prompt_received',
      'candidate_extraction',
      'candidate_normalization',
      'utility_scoring',
      'admission_gate',
    ],
    color: '#6366f1',
  },
  {
    id: 'lifecycle',
    name: '2. Lifecycle Management & Dynamic Allocation',
    stages: [
      'deduplication',
      'conflict_resolution',
      'memory_consolidation',
      'lifecycle_transitions',
      'adaptive_allocation',
      'persistence',
    ],
    color: '#8b5cf6',
  },
  {
    id: 'retrieval',
    name: '3. Query Retrieval & SLM Generation',
    stages: [
      'memory_retrieval',
      'reranking',
      'context_assembly',
      'slm_generation',
      'response_completed',
    ],
    color: '#06b6d4',
  },
];

export const ExecutionGraph: React.FC<ExecutionGraphProps> = ({
  trace,
  selectedStage,
  onSelectStage,
  isLoading,
}) => {
  // Map trace events by stage
  const eventMap = React.useMemo(() => {
    const map = new Map<TraceStage, TraceEvent>();
    if (trace && trace.events) {
      trace.events.forEach((ev) => {
        map.set(ev.stage, ev);
      });
    }
    return map;
  }, [trace]);

  const getStatusIcon = (status: EventStatus) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 size={13} className="text-emerald-400" style={{ color: '#10b981' }} />;
      case 'warning':
        return <AlertTriangle size={13} className="text-amber-400" style={{ color: '#f59e0b' }} />;
      case 'failed':
        return <XCircle size={13} className="text-rose-400" style={{ color: '#ef4444' }} />;
      case 'skipped':
        return <MinusCircle size={13} className="text-slate-400" style={{ color: '#64748b' }} />;
      default:
        return <span className="node-status-indicator status-pending" />;
    }
  };

  return (
    <div className="graph-pane">
      <div className="graph-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Zap size={16} style={{ color: 'var(--accent-primary)' }} />
          <h2 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Memory Execution Pipeline</h2>
          {trace ? (
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
                background: 'var(--bg-tertiary)',
                padding: '0.15rem 0.5rem',
                borderRadius: '4px',
                border: '1px solid var(--border-subtle)',
              }}
            >
              {trace.trace_id}
            </span>
          ) : (
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              (Awaiting prompt execution)
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {trace && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem',
                fontSize: '0.75rem',
                color: 'var(--text-secondary)',
                fontFamily: 'var(--font-mono)',
              }}
            >
              <Clock size={12} />
              {trace.total_duration_ms != null ? `${trace.total_duration_ms.toFixed(2)} ms` : '-'}
            </span>
          )}
          {trace && (
            <span
              className={`badge ${
                trace.status === 'completed'
                  ? 'badge-active'
                  : trace.status === 'failed'
                  ? 'badge-superseded'
                  : 'badge-temporary'
              }`}
            >
              {trace.status}
            </span>
          )}
        </div>
      </div>

      <div className="graph-canvas">
        {PHASES.map((phase) => (
          <div key={phase.id} className="phase-group">
            <div className="phase-header">
              <span className="phase-title" style={{ color: phase.color }}>
                {phase.name}
              </span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                {phase.stages.filter((s) => eventMap.has(s)).length} / {phase.stages.length} events
              </span>
            </div>

            <div className="nodes-grid">
              {phase.stages.map((stageId) => {
                const def = STAGE_DEFINITIONS[stageId];
                const event = eventMap.get(stageId);
                const isSelected = selectedStage === stageId;
                const status: EventStatus = event ? event.status : isLoading ? 'running' : 'pending';

                return (
                  <div
                    key={stageId}
                    className={`stage-node ${isSelected ? 'selected' : ''}`}
                    onClick={() => onSelectStage(stageId)}
                    role="button"
                    tabIndex={0}
                    aria-label={`Inspect ${def.label}`}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        onSelectStage(stageId);
                      }
                    }}
                  >
                    <div className="stage-node-header">
                      <span className="node-number">#{def.number}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        {getStatusIcon(status)}
                        <span className={`node-status-indicator status-${status}`} />
                      </div>
                    </div>

                    <div className="node-label">{def.label}</div>

                    <div className="node-meta">
                      <span>
                        {event && event.duration_ms != null
                          ? `${event.duration_ms.toFixed(2)} ms`
                          : event
                          ? '0.00 ms'
                          : '-'}
                      </span>
                      {event?.decision && (
                        <span className="node-decision-badge" title={event.decision}>
                          {event.decision}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

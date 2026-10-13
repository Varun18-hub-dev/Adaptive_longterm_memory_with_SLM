import React, { useEffect, useState } from 'react';
import {
  X,
  Database,
  History,
  ArrowRight,
  GitCommit,
} from 'lucide-react';
import { api } from '../services/api';
import type { MemoryDetailResponse } from '../types';

interface MemoryDetailModalProps {
  memoryId: string | null;
  sessionId: string;
  onClose: () => void;
}

export const MemoryDetailModal: React.FC<MemoryDetailModalProps> = ({
  memoryId,
  sessionId,
  onClose,
}) => {
  const [detail, setDetail] = useState<MemoryDetailResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!memoryId) return;

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    api
      .getMemoryDetail(memoryId, sessionId)
      .then((res) => {
        if (isMounted) setDetail(res);
      })
      .catch((err) => {
        if (isMounted) setError(err.message || 'Failed to load memory details');
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [memoryId, sessionId]);

  if (!memoryId) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Database size={16} style={{ color: 'var(--accent-primary)' }} />
            <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>Memory Item Inspector</h3>
            {detail && (
              <span className={`badge badge-${detail.store_type}`}>
                {detail.store_type}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="preset-chip"
            style={{ padding: '0.2rem', borderRadius: '50%' }}
            title="Close"
          >
            <X size={16} />
          </button>
        </div>

        <div className="modal-body">
          {isLoading && (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
              Loading memory record and provenance...
            </div>
          )}

          {error && (
            <div style={{ color: 'var(--status-danger)', background: 'var(--status-danger-bg)', padding: '0.75rem', borderRadius: '6px', fontSize: '0.85rem' }}>
              {error}
            </div>
          )}

          {detail && detail.memory && (
            <>
              {/* Content Box */}
              <div className="card-panel">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    ID: {detail.memory.id}
                  </span>
                  <span className={`badge badge-${detail.memory.status}`}>
                    {detail.memory.status}
                  </span>
                </div>
                <div style={{ fontSize: '0.95rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                  {detail.memory.content}
                </div>
              </div>

              {/* Utility & Scoring Metrics */}
              <div className="card-panel">
                <div className="card-panel-title">Scoring Metrics & Budget Cost</div>
                <div className="metric-grid">
                  <div className="metric-box">
                    <span className="metric-label">Utility Score U(m)</span>
                    <span className="metric-val" style={{ color: 'var(--accent-primary)' }}>
                      {detail.memory.utility_score != null ? detail.memory.utility_score.toFixed(4) : '-'}
                    </span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-label">Token Cost</span>
                    <span className="metric-val">{detail.memory.token_cost} tokens</span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-label">Importance</span>
                    <span className="metric-val">{detail.memory.importance}</span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-label">Stability</span>
                    <span className="metric-val">{detail.memory.stability}</span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-label">Recency</span>
                    <span className="metric-val">{detail.memory.recency}</span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-label">Frequency</span>
                    <span className="metric-val">{detail.memory.frequency}</span>
                  </div>
                </div>
              </div>

              {/* Superseding & Conflict History */}
              {(detail.memory.supersedes_id || detail.memory.superseded_by_id) && (
                <div className="card-panel">
                  <div className="card-panel-title">Temporal Conflict Relations</div>
                  {detail.memory.supersedes_id && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--status-warning)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <ArrowRight size={13} />
                      <span>Superseded older assertion: {detail.memory.supersedes_id}</span>
                    </div>
                  )}
                  {detail.memory.superseded_by_id && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--status-danger)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <ArrowRight size={13} />
                      <span>Superseded by newer assertion: {detail.memory.superseded_by_id}</span>
                    </div>
                  )}
                </div>
              )}

              {/* Provenance */}
              {detail.memory.provenance && detail.memory.provenance.length > 0 && (
                <div className="card-panel">
                  <div className="card-panel-title">
                    <History size={14} />
                    <span>Provenance History ({detail.memory.provenance.length} occurrences)</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {detail.memory.provenance.map((prov, i) => (
                      <div key={i} className="candidate-card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                          <span>Session: {prov.session_id} • Turn: #{prov.turn_index}</span>
                          <span>{new Date(prov.timestamp * 1000).toLocaleTimeString()}</span>
                        </div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)' }}>
                          "{prov.source_text}"
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Audit Log */}
              {detail.audit_history && detail.audit_history.length > 0 && (
                <div className="card-panel">
                  <div className="card-panel-title">
                    <GitCommit size={14} />
                    <span>Audit Log Transitions ({detail.audit_history.length} events)</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {detail.audit_history.map((audit, i) => (
                      <div key={i} className="candidate-card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent-primary)' }}>
                          <span>{audit.event_type}</span>
                          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 400 }}>
                            {new Date(audit.timestamp * 1000).toLocaleTimeString()}
                          </span>
                        </div>
                        {audit.details && Object.keys(audit.details).length > 0 && (
                          <pre className="json-viewer" style={{ fontSize: '0.7rem', maxHeight: '100px' }}>
                            {JSON.stringify(audit.details, null, 2)}
                          </pre>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

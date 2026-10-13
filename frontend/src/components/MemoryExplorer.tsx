import React, { useState } from 'react';
import {
  Database,
  Search,
  Layers,
  ArrowRight,
  ShieldAlert,
  ExternalLink,
} from 'lucide-react';
import type {
  MemoriesResponse,
  MemoryItem,
  RejectedCandidateRecord,
} from '../types';

interface MemoryExplorerProps {
  memoriesData: MemoriesResponse | null;
  onSelectMemory: (item: MemoryItem) => void;
  isLoading: boolean;
}

export const MemoryExplorer: React.FC<MemoryExplorerProps> = ({
  memoriesData,
  onSelectMemory,
  isLoading,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [storeFilter, setStoreFilter] = useState<'all' | 'episodic' | 'semantic' | 'archive' | 'rejected'>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const storageBudget = memoriesData?.storage_budget || 150;
  const usedTokens = memoriesData?.used_tokens || 0;
  const remainingTokens = memoriesData?.remaining_tokens || Math.max(0, storageBudget - usedTokens);
  const usedPercent = Math.min(100, Math.round((usedTokens / storageBudget) * 100));

  // Collect memory items based on active storeFilter
  const allItems: MemoryItem[] = React.useMemo(() => {
    if (!memoriesData) return [];
    let items: MemoryItem[] = [];
    if (storeFilter === 'all' || storeFilter === 'episodic') {
      items = items.concat(memoriesData.episodic_store || []);
    }
    if (storeFilter === 'all' || storeFilter === 'semantic') {
      items = items.concat(memoriesData.semantic_store || []);
    }
    if (storeFilter === 'all' || storeFilter === 'archive') {
      items = items.concat(memoriesData.archive_store || []);
    }
    return items;
  }, [memoriesData, storeFilter]);

  // Apply search query and status filter
  const filteredItems = React.useMemo(() => {
    return allItems.filter((m) => {
      if (statusFilter !== 'all' && m.status !== statusFilter) {
        return false;
      }
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        return (
          m.content.toLowerCase().includes(query) ||
          m.id.toLowerCase().includes(query)
        );
      }
      return true;
    });
  }, [allItems, statusFilter, searchQuery]);

  const rejectedCandidates: RejectedCandidateRecord[] = memoriesData?.rejected_candidates || [];

  return (
    <div className="memory-explorer-layout">
      {/* Storage Token Budget Bar */}
      <div className="budget-bar-container">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Database size={18} style={{ color: 'var(--accent-primary)' }} />
            <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>Persistent Memory Storage Budget</span>
          </div>

          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
            <strong>{usedTokens}</strong> / {storageBudget} tokens ({usedPercent}% allocated)
          </div>
        </div>

        <div className="budget-progress-track">
          <div className="budget-progress-used" style={{ width: `${usedPercent}%` }} />
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span>
            Used: {usedTokens} tokens ({filteredItems.length} active items)
            {isLoading && <span style={{ marginLeft: '0.5rem', color: 'var(--accent-primary)' }}>• Refreshing stores...</span>}
          </span>
          <span>Remaining Capacity: {remainingTokens} tokens</span>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="filter-bar">
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
          <Search size={15} style={{ position: 'absolute', left: '10px', color: 'var(--text-muted)' }} />
          <input
            type="text"
            className="search-input"
            style={{ paddingLeft: '32px' }}
            placeholder="Search memory content or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        {/* Store selector */}
        <div style={{ display: 'flex', gap: '0.3rem' }}>
          {(['all', 'episodic', 'semantic', 'archive', 'rejected'] as const).map((store) => (
            <button
              key={store}
              className={`preset-chip ${storeFilter === store ? 'active' : ''}`}
              style={{
                textTransform: 'capitalize',
                background: storeFilter === store ? 'var(--accent-primary-subtle)' : undefined,
                borderColor: storeFilter === store ? 'var(--accent-primary)' : undefined,
                color: storeFilter === store ? 'white' : undefined,
              }}
              onClick={() => setStoreFilter(store)}
            >
              {store}
            </button>
          ))}
        </div>

        {/* Status selector */}
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          style={{
            background: 'var(--bg-card)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '6px',
            padding: '0.35rem 0.6rem',
            fontSize: '0.8rem',
          }}
        >
          <option value="all">All Statuses</option>
          <option value="active">Active</option>
          <option value="temporary">Temporary</option>
          <option value="superseded">Superseded</option>
          <option value="archived">Archived</option>
        </select>
      </div>

      {/* Memory Cards Grid */}
      {storeFilter !== 'rejected' && (
        <>
          {filteredItems.length === 0 ? (
            <div className="card-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              <Layers size={32} style={{ opacity: 0.4, margin: '0 auto 0.75rem' }} />
              <p>No memory items found in selected store matching filter.</p>
            </div>
          ) : (
            <div className="memory-grid">
              {filteredItems.map((item) => (
                <div
                  key={item.id}
                  className="memory-card"
                  onClick={() => onSelectMemory(item)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') onSelectMemory(item);
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span className={`badge badge-${item.memory_type}`}>{item.memory_type}</span>
                    <span className={`badge badge-${item.status}`}>{item.status}</span>
                  </div>

                  <p style={{ fontSize: '0.875rem', color: 'var(--text-primary)', lineHeight: 1.4 }}>
                    {item.content}
                  </p>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.4rem', fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    <div>Util: {item.utility_score != null ? item.utility_score.toFixed(3) : '-'}</div>
                    <div>Cost: {item.token_cost} tok</div>
                    <div>Freq: {item.frequency}</div>
                  </div>

                  {item.supersedes_id && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--status-warning)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <ArrowRight size={11} />
                      <span>Supersedes: {item.supersedes_id.slice(0, 8)}...</span>
                    </div>
                  )}

                  {item.superseded_by_id && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--status-danger)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <ArrowRight size={11} />
                      <span>Superseded by: {item.superseded_by_id.slice(0, 8)}...</span>
                    </div>
                  )}

                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                    <span>ID: {item.id.slice(0, 10)}...</span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', color: 'var(--accent-primary)' }}>
                      Inspect <ExternalLink size={10} />
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Rejected Candidates Section */}
      {(storeFilter === 'all' || storeFilter === 'rejected') && rejectedCandidates.length > 0 && (
        <div className="card-panel" style={{ marginTop: '1rem', borderColor: 'rgba(239, 68, 68, 0.3)' }}>
          <div className="card-panel-title" style={{ color: 'var(--status-warning)' }}>
            <ShieldAlert size={16} />
            <span>Admission Gate Filtered Candidates ({rejectedCandidates.length})</span>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            These candidates were evaluated and rejected by the admission gate due to low confidence or utility.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginTop: '0.5rem' }}>
            {rejectedCandidates.map((rej, idx) => (
              <div key={idx} className="candidate-card" style={{ borderColor: 'rgba(239, 68, 68, 0.2)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="badge badge-superseded">{rej.decision}</span>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    Cost: {rej.memory.token_cost} tokens
                  </span>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)' }}>{rej.memory.content}</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--status-danger)', fontStyle: 'italic' }}>
                  Reason: {rej.reason}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

import React from 'react';
import {
  BarChart3,
  Clock,
  Layers,
  Database,
  ShieldCheck,
  Zap,
  Info,
} from 'lucide-react';
import type {
  ExecutionTrace,
  MemoriesResponse,
} from '../types';

interface ResearchOverviewProps {
  trace: ExecutionTrace | null;
  memoriesData: MemoriesResponse | null;
}

export const ResearchOverview: React.FC<ResearchOverviewProps> = ({
  trace,
  memoriesData,
}) => {
  // Compute stage durations by category
  const phaseLatencies = React.useMemo(() => {
    let ingestionMs = 0;
    let lifecycleMs = 0;
    let retrievalMs = 0;

    if (trace?.events) {
      trace.events.forEach((ev) => {
        const ms = ev.duration_ms || 0;
        if (
          [
            'prompt_received',
            'candidate_extraction',
            'candidate_normalization',
            'utility_scoring',
            'admission_gate',
          ].includes(ev.stage)
        ) {
          ingestionMs += ms;
        } else if (
          [
            'deduplication',
            'conflict_resolution',
            'memory_consolidation',
            'lifecycle_transitions',
            'adaptive_allocation',
            'persistence',
          ].includes(ev.stage)
        ) {
          lifecycleMs += ms;
        } else {
          retrievalMs += ms;
        }
      });
    }

    return {
      ingestionMs,
      lifecycleMs,
      retrievalMs,
      totalMs: ingestionMs + lifecycleMs + retrievalMs,
    };
  }, [trace]);

  // Compute memory budget allocation stats
  const budgetStats = React.useMemo(() => {
    const totalBudget = memoriesData?.storage_budget || 150;
    const episodicCount = memoriesData?.episodic_store?.length || 0;
    const semanticCount = memoriesData?.semantic_store?.length || 0;
    const episodicTokens = memoriesData?.episodic_store?.reduce((sum, m) => sum + m.token_cost, 0) || 0;
    const semanticTokens = memoriesData?.semantic_store?.reduce((sum, m) => sum + m.token_cost, 0) || 0;
    const totalUsed = episodicTokens + semanticTokens;

    return {
      totalBudget,
      episodicCount,
      semanticCount,
      episodicTokens,
      semanticTokens,
      totalUsed,
      freeTokens: Math.max(0, totalBudget - totalUsed),
    };
  }, [memoriesData]);

  // Get trace-specific counts
  const traceStats = React.useMemo(() => {
    if (!trace) return null;
    const extractionEv = trace.events?.find((e) => e.stage === 'candidate_extraction');
    const gateEv = trace.events?.find((e) => e.stage === 'admission_gate');
    const allocEv = trace.events?.find((e) => e.stage === 'adaptive_allocation');
    const retrievalEv = trace.events?.find((e) => e.stage === 'memory_retrieval');
    const contextEv = trace.events?.find((e) => e.stage === 'context_assembly');

    return {
      candidatesExtracted: (extractionEv?.outputs?.candidate_count as number) || 0,
      candidatesAdmitted: (gateEv?.outputs?.admitted_count as number) || 0,
      candidatesRejected: (gateEv?.outputs?.rejected_count as number) || 0,
      evictedCount: (allocEv?.outputs?.evicted_count as number) || 0,
      retrievedCount: (retrievalEv?.outputs?.retrieved_count as number) || 0,
      contextTokens: (contextEv?.outputs?.context_token_cost as number) || 0,
      contextBudget: (contextEv?.outputs?.context_budget as number) || 256,
    };
  }, [trace]);

  return (
    <div className="memory-explorer-layout">
      {/* Research Scope Clarification Notice */}
      <div className="card-panel" style={{ borderColor: 'var(--border-focus)', background: 'rgba(99, 102, 241, 0.05)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-primary)', fontWeight: 600, fontSize: '0.85rem' }}>
          <Info size={16} />
          <span>Research Observability Scope</span>
        </div>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
          The metrics displayed below reflect measured live telemetry from real runtime execution of the
          active session. Per-request allocation, scoring weights, and candidate yields demonstrate live
          behavior under evolving facts. Benchmark-wide multi-session retention and recall comparisons are
          evaluated via the Phase 4 test and evaluation harness.
        </p>
      </div>

      {/* Top Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        <div className="card-panel">
          <div className="card-panel-title">
            <Clock size={14} />
            <span>Total Turn Latency</span>
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
            {trace?.total_duration_ms != null ? `${trace.total_duration_ms.toFixed(2)} ms` : '-'}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Status: {trace?.status || 'No active trace'}
          </span>
        </div>

        <div className="card-panel">
          <div className="card-panel-title">
            <Database size={14} />
            <span>Persistent Memory Budget</span>
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)' }}>
            {budgetStats.totalUsed} / {budgetStats.totalBudget} tok
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {budgetStats.freeTokens} tokens remaining capacity
          </span>
        </div>

        <div className="card-panel">
          <div className="card-panel-title">
            <ShieldCheck size={14} />
            <span>Candidate Admission Yield</span>
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--status-success)' }}>
            {traceStats ? `${traceStats.candidatesAdmitted} / ${traceStats.candidatesExtracted}` : '-'}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {traceStats ? `${traceStats.candidatesRejected} rejected by gate` : 'Awaiting prompt'}
          </span>
        </div>

        <div className="card-panel">
          <div className="card-panel-title">
            <Zap size={14} />
            <span>Prompt Context Usage</span>
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
            {traceStats ? `${traceStats.contextTokens} / ${traceStats.contextBudget} tok` : '-'}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {traceStats ? `${traceStats.retrievedCount} memories retrieved` : 'Awaiting prompt'}
          </span>
        </div>
      </div>

      {/* Latency by Macro-Phase */}
      <div className="card-panel">
        <div className="card-panel-title">
          <BarChart3 size={15} />
          <span>Pipeline Macro-Phase Latency Breakdown</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginTop: '0.5rem' }}>
          <div className="metric-box">
            <span className="metric-label" style={{ color: '#6366f1' }}>
              Phase 1: Ingestion & Gating (Stages 1-5)
            </span>
            <span className="metric-val">{phaseLatencies.ingestionMs.toFixed(2)} ms</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Extraction, Normalization, Utility Scoring, Admission Gate
            </span>
          </div>

          <div className="metric-box">
            <span className="metric-label" style={{ color: '#8b5cf6' }}>
              Phase 2: Lifecycle & Dynamic Allocation (Stages 6-11)
            </span>
            <span className="metric-val">{phaseLatencies.lifecycleMs.toFixed(2)} ms</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Deduplication, Conflict Superseding, Consolidation, Adaptive Budgeting
            </span>
          </div>

          <div className="metric-box">
            <span className="metric-label" style={{ color: '#06b6d4' }}>
              Phase 3: Retrieval & Generation (Stages 12-16)
            </span>
            <span className="metric-val">{phaseLatencies.retrievalMs.toFixed(2)} ms</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Relevance Search, Reranking, Context Fitting, SLM Response
            </span>
          </div>
        </div>
      </div>

      {/* Memory Pool Split Breakdown */}
      <div className="card-panel">
        <div className="card-panel-title">
          <Layers size={15} />
          <span>Hierarchical Memory Pool Distribution</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem', marginTop: '0.5rem' }}>
          <div className="card-panel" style={{ background: 'var(--bg-primary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="badge badge-episodic">Episodic Store</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                {budgetStats.episodicTokens} tokens ({budgetStats.episodicCount} items)
              </span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Holds recent, fine-grained conversational turns and temporary observations subject to TTL and recency decay.
            </p>
          </div>

          <div className="card-panel" style={{ background: 'var(--bg-primary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="badge badge-semantic">Semantic Store</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                {budgetStats.semanticTokens} tokens ({budgetStats.semanticCount} items)
              </span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Holds consolidated durable facts, stable personal attributes, and repeatedly reinforced user knowledge.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

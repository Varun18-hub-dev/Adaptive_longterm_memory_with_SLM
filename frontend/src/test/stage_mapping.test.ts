import { describe, it, expect } from 'vitest';
import { STAGE_DEFINITIONS, EVIDENCE_GROUPS, type TraceStage } from '../types';

describe('16 Pipeline Stages Mapping Tests', () => {
  const expectedStages: TraceStage[] = [
    'prompt_received',
    'candidate_extraction',
    'candidate_normalization',
    'utility_scoring',
    'admission_gate',
    'deduplication',
    'conflict_resolution',
    'memory_consolidation',
    'lifecycle_transitions',
    'adaptive_allocation',
    'persistence',
    'memory_retrieval',
    'reranking',
    'context_assembly',
    'slm_generation',
    'response_completed',
  ];

  it('contains all 16 sequential stage definitions', () => {
    expect(Object.keys(STAGE_DEFINITIONS)).toHaveLength(16);
    expectedStages.forEach((stageId, index) => {
      const def = STAGE_DEFINITIONS[stageId];
      expect(def).toBeDefined();
      expect(def.id).toBe(stageId);
      expect(def.number).toBe(index + 1);
      expect(def.label.length).toBeGreaterThan(0);
      expect(def.description.length).toBeGreaterThan(0);
    });
  });

  it('correctly maps macro-phase categories', () => {
    // Phase 1 (Ingestion): 1 to 5
    for (let i = 0; i < 5; i++) {
      expect(STAGE_DEFINITIONS[expectedStages[i]].category).toBe('Ingestion');
    }
    // Phase 2 (Lifecycle & Allocation): 6 to 11
    for (let i = 5; i < 11; i++) {
      expect(STAGE_DEFINITIONS[expectedStages[i]].category).toBe('Lifecycle & Allocation');
    }
    // Phase 3 (Retrieval & Inference): 12 to 16
    for (let i = 11; i < 16; i++) {
      expect(STAGE_DEFINITIONS[expectedStages[i]].category).toBe('Retrieval & Inference');
    }
  });

  it('correctly defines all 5 plain-language evidence groups', () => {
    expect(EVIDENCE_GROUPS).toHaveLength(5);
    const expectedGroupIds = [
      'understand_prompt',
      'evaluate_candidates',
      'update_memory',
      'retrieve_memories',
      'generate_response',
    ];

    EVIDENCE_GROUPS.forEach((group, index) => {
      expect(group.id).toBe(expectedGroupIds[index]);
      expect(group.number).toBe(index + 1);
      expect(group.title.length).toBeGreaterThan(0);
      expect(group.stages.length).toBeGreaterThan(0);
      expect(group.description.length).toBeGreaterThan(0);
    });
  });
});

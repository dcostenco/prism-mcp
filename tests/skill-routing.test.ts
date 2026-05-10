/**
 * Skill routing client tests
 *
 * Verify that prism-mcp's `resolveSkillsForProject` correctly:
 *   - returns the universal skill set when no project pattern matches
 *   - unions multiple matching project patterns
 *   - falls back to OFFLINE_FALLBACK when synalux is unreachable
 *   - caches the routing table in-memory across calls
 *   - case-insensitively substring-matches the project name
 *
 * The canonical routing source is synalux at /api/v1/skills/routing.
 * If you change the routing schema, update both this test and the route.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { resolveSkillsForProject, _invalidateRoutingCache, _OFFLINE_FALLBACK } from '../src/tools/skillRouting.js';

const ORIGINAL_FETCH = globalThis.fetch;

beforeEach(() => {
  _invalidateRoutingCache();
});

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
  _invalidateRoutingCache();
});

function mockFetch(table: unknown, ok = true) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok,
    json: async () => table,
    status: ok ? 200 : 500,
  } as Response);
}

describe('skill routing — happy path', () => {
  it('returns universal skill for any project', async () => {
    mockFetch({
      version: 2,
      universal: ['bcba_ai_assistant'],
      projects: { 'prism-aac': ['i18n-tts'] },
      user_local: { enabled: false, key_prefix: 'user_skill:' },
    });
    const result = await resolveSkillsForProject('unknown-project');
    expect(result.names).toContain('bcba_ai_assistant');
    expect(result.user_local.enabled).toBe(false);
  });

  it('matches project pattern as substring', async () => {
    mockFetch({
      version: 2,
      universal: ['bcba_ai_assistant'],
      projects: { 'prism-aac': ['i18n-tts'] },
      user_local: { enabled: false, key_prefix: 'user_skill:' },
    });
    const result = await resolveSkillsForProject('my-prism-aac-fork');
    expect(result.names).toContain('i18n-tts');
    expect(result.names).toContain('bcba_ai_assistant');
  });

  it('unions multiple matching patterns', async () => {
    mockFetch({
      version: 2,
      universal: ['bcba_ai_assistant'],
      projects: {
        'prism': ['session-memory'],
        'prism-mcp': ['ai-agent-super-skill'],
      },
      user_local: { enabled: false, key_prefix: 'user_skill:' },
    });
    const result = await resolveSkillsForProject('prism-mcp');
    expect(result.names).toContain('session-memory');
    expect(result.names).toContain('ai-agent-super-skill');
    expect(result.names).toContain('bcba_ai_assistant');
  });

  it('matches case-insensitively', async () => {
    mockFetch({
      version: 2,
      universal: [],
      projects: { 'prism-aac': ['i18n-tts'] },
      user_local: { enabled: false, key_prefix: 'user_skill:' },
    });
    expect((await resolveSkillsForProject('Prism-AAC')).names).toContain('i18n-tts');
    expect((await resolveSkillsForProject('PRISM-AAC')).names).toContain('i18n-tts');
  });

  it('returns no duplicates when universal + project skill match', async () => {
    mockFetch({
      version: 2,
      universal: ['shared-skill'],
      projects: { 'prism': ['shared-skill', 'unique-skill'] },
      user_local: { enabled: false, key_prefix: 'user_skill:' },
    });
    const result = await resolveSkillsForProject('prism');
    expect(result.names.filter((s) => s === 'shared-skill').length).toBe(1);
    expect(result.names).toContain('unique-skill');
  });

  it('user_local.enabled=true is returned when routing table sets it', async () => {
    mockFetch({
      version: 2,
      universal: ['bcba_ai_assistant'],
      projects: {},
      user_local: { enabled: true, key_prefix: 'user_skill:' },
    });
    const result = await resolveSkillsForProject('any');
    expect(result.user_local.enabled).toBe(true);
    expect(result.user_local.key_prefix).toBe('user_skill:');
  });
});

describe('skill routing — offline fallback', () => {
  it('returns OFFLINE_FALLBACK when synalux unreachable', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('network'));
    const result = await resolveSkillsForProject('prism-aac');
    expect(result.names).toEqual(_OFFLINE_FALLBACK.universal);
    expect(result.user_local.enabled).toBe(false);
  });

  it('returns OFFLINE_FALLBACK on 500 error', async () => {
    mockFetch({}, false);
    const result = await resolveSkillsForProject('any');
    expect(result.names).toEqual(_OFFLINE_FALLBACK.universal);
  });

  it('rejects malformed routing table', async () => {
    mockFetch({ version: 'wrong-type', universal: 'not-array' });
    const result = await resolveSkillsForProject('any');
    expect(result.names).toEqual(_OFFLINE_FALLBACK.universal);
  });
});

describe('skill routing — caching behavior', () => {
  it('uses cache on second call within TTL', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        version: 2,
        universal: ['bcba_ai_assistant'],
        projects: { 'prism': ['session-memory'] },
        user_local: { enabled: false, key_prefix: 'user_skill:' },
      }),
      status: 200,
    } as Response);
    globalThis.fetch = fetchMock;

    await resolveSkillsForProject('prism');
    await resolveSkillsForProject('prism');
    await resolveSkillsForProject('prism');

    // 1 fetch, even though resolveSkillsForProject was called 3 times
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('refetches after invalidation', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        version: 2, universal: [], projects: {},
        user_local: { enabled: false, key_prefix: 'user_skill:' },
      }),
      status: 200,
    } as Response);
    globalThis.fetch = fetchMock;

    await resolveSkillsForProject('prism');
    _invalidateRoutingCache();
    await resolveSkillsForProject('prism');

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe('skill routing — fail-safe defaults', () => {
  it('OFFLINE_FALLBACK includes the universal BCBA skill', () => {
    expect(_OFFLINE_FALLBACK.universal).toContain('bcba_ai_assistant');
  });

  it('OFFLINE_FALLBACK has no project-specific skills', () => {
    // Project skills require synalux to resolve correctly; offline mode
    // falls back to universal-only so we don't ship stale project mappings.
    expect(Object.keys(_OFFLINE_FALLBACK.projects).length).toBe(0);
  });
});

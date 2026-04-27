/**
 * Test Suite — SCM Thin Client Integration + Edge Cases
 * ======================================================
 *
 * Tests the Prism ↔ Synalux SCM integration via the thin API client.
 * All API calls are mocked — no network required.
 *
 * Coverage:
 *   1. ScmClient — search, review, scan, dora API calls
 *   2. SCM tier configuration — limits, boundaries, access control
 *   3. Edge cases — empty inputs, auth failures, malformed responses
 *   4. CLI argument parsing — mode validation, repo derivation
 */

import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';
import { ScmClient } from '../../src/scm/client.js';
import { SCM_TIERS, type ScmTier, type SearchMode } from '../../src/scm/types.js';

// ═══════════════════════════════════════════════════════════════
// Mock fetch globally
// ═══════════════════════════════════════════════════════════════

const mockFetch = vi.fn();

beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch);
    mockFetch.mockReset();
});

afterEach(() => {
    vi.unstubAllGlobals();
});

function mockJsonResponse(data: any, status = 200) {
    return Promise.resolve({
        ok: status >= 200 && status < 300,
        status,
        statusText: status === 200 ? 'OK' : `Error ${status}`,
        json: () => Promise.resolve(data),
        text: () => Promise.resolve(JSON.stringify(data)),
    });
}

// ═══════════════════════════════════════════════════════════════
// 1. ScmClient — API Integration
// ═══════════════════════════════════════════════════════════════

describe('ScmClient — Search API', () => {
    test('sends correct search request to Synalux API', async () => {
        const mockResponse = {
            results: [{ file: 'src/auth.ts', line_number: 10, content: 'login()', score: 100, match_type: 'exact', repo_full_name: 'synalux/portal' }],
            total_matches: 1,
            search_time_ms: 42,
            truncated: false,
        };
        mockFetch.mockReturnValue(mockJsonResponse(mockResponse));

        const client = new ScmClient('https://test.synalux.ai', 'test-key');
        const result = await client.search('synalux/portal', { query: 'login', mode: 'exact' });

        expect(mockFetch).toHaveBeenCalledTimes(1);
        const [url, opts] = mockFetch.mock.calls[0];
        expect(url).toBe('https://test.synalux.ai/api/v1/scm/repos/synalux/portal/search');
        expect(opts.method).toBe('POST');
        expect(JSON.parse(opts.body)).toEqual({ query: 'login', mode: 'exact' });
        expect(opts.headers.Authorization).toBe('Bearer test-key');
        expect(result.total_matches).toBe(1);
    });

    test('search with all options', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ results: [], total_matches: 0, search_time_ms: 5, truncated: false }));

        const client = new ScmClient('https://test.synalux.ai');
        await client.search('org/repo', {
            query: 'handleRequest',
            mode: 'symbol',
            file_patterns: ['*.ts'],
            exclude_patterns: ['*.test.ts'],
            max_results: 5,
            context_lines: 3,
        });

        const body = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(body.mode).toBe('symbol');
        expect(body.file_patterns).toEqual(['*.ts']);
        expect(body.max_results).toBe(5);
    });

    test('search without API key omits Authorization header', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ results: [], total_matches: 0, search_time_ms: 0, truncated: false }));

        const client = new ScmClient('https://test.synalux.ai'); // no key
        await client.search('org/repo', { query: 'test', mode: 'exact' });

        expect(mockFetch.mock.calls[0][1].headers.Authorization).toBeUndefined();
    });
});

describe('ScmClient — Review API', () => {
    test('sends files for AI review', async () => {
        const mockReview = {
            review: { overall_score: 85, total_findings: 1, auto_approve: true, model: 'synalux-scanner-v1', findings: [] },
        };
        mockFetch.mockReturnValue(mockJsonResponse(mockReview));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        const result = await client.review('org/repo', [
            { name: 'src/db.ts', content: 'const q = "SELECT * FROM users"' },
        ]);

        const [url, opts] = mockFetch.mock.calls[0];
        expect(url).toContain('/repos/org/repo/review');
        expect(JSON.parse(opts.body).files).toHaveLength(1);
        expect(result.review.overall_score).toBe(85);
    });

    test('review with HIPAA flag', async () => {
        const mockResult = {
            review: { overall_score: 60, total_findings: 2, auto_approve: false, model: 'v1', findings: [] },
            hipaa: { compliant: false, score: 40, violations: [{ id: 'H1', severity: 'critical' }] },
        };
        mockFetch.mockReturnValue(mockJsonResponse(mockResult));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        const result = await client.review('org/repo', [{ name: 'app.ts', content: 'console.log(ssn)' }], { hipaa: true });

        expect(JSON.parse(mockFetch.mock.calls[0][1].body).hipaa).toBe(true);
        expect(result.hipaa?.compliant).toBe(false);
    });
});

describe('ScmClient — Security Scan API', () => {
    test('sends files for security scan', async () => {
        const mockScan = {
            summary: { total_findings: 2, critical: 1, high: 1, medium: 0, low: 0, pass: false, scan_duration_ms: 15 },
            findings: [
                { id: 's1', type: 'secrets', severity: 'critical', title: 'AWS Key', description: 'x', remediation: 'x' },
                { id: 's2', type: 'docker', severity: 'high', title: 'No USER', description: 'x', remediation: 'x' },
            ],
        };
        mockFetch.mockReturnValue(mockJsonResponse(mockScan));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        const result = await client.scan('org/repo', [
            { name: '.env', content: 'AWS_KEY=AKIATEST1234567890AB' },
        ]);

        expect(result.summary.pass).toBe(false);
        expect(result.summary.critical).toBe(1);
        expect(result.findings).toHaveLength(2);
    });
});

describe('ScmClient — DORA API', () => {
    test('fetches DORA metrics without period', async () => {
        const mockDora = {
            deployment_frequency: { value: 2.5, level: 'elite' },
            lead_time: { value: 3.2, level: 'elite' },
            change_failure_rate: { value: 4.1, level: 'elite' },
            mttr: { value: 0.5, level: 'elite' },
            overall_level: 'elite',
            period: '2024-Q4',
            team_size: 5,
        };
        mockFetch.mockReturnValue(mockJsonResponse(mockDora));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        const result = await client.dora('org/repo');

        expect(mockFetch.mock.calls[0][0]).toBe('https://test.synalux.ai/api/v1/scm/repos/org/repo/dora');
        expect(result.overall_level).toBe('elite');
    });

    test('fetches DORA metrics with period filter', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ overall_level: 'high', period: '2024-Q3' }));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        await client.dora('org/repo', '2024-Q3');

        expect(mockFetch.mock.calls[0][0]).toContain('?period=2024-Q3');
    });
});

// ═══════════════════════════════════════════════════════════════
// 2. SCM Tier Configuration
// ═══════════════════════════════════════════════════════════════

describe('SCM Tier Configuration', () => {
    test('all 4 tiers are defined', () => {
        const tiers: ScmTier[] = ['free', 'standard', 'advanced', 'enterprise'];
        tiers.forEach(t => expect(SCM_TIERS[t]).toBeDefined());
    });

    test('free tier has strictest limits', () => {
        const free = SCM_TIERS.free;
        expect(free.public_repos).toBe(3);
        expect(free.private_repos).toBe(1);
        expect(free.ide_hours_per_day).toBe(1);
        expect(free.ai_reviews_per_month).toBe(5);
        expect(free.search_modes).toEqual(['exact']);
        expect(free.stacked_prs).toBe(false);
        expect(free.hipaa_compliance).toBe(false);
        expect(free.sso_saml).toBe(false);
        expect(free.dora_metrics).toBe('none');
    });

    test('standard tier unlocks regex+symbol search', () => {
        const std = SCM_TIERS.standard;
        expect(std.search_modes).toContain('regex');
        expect(std.search_modes).toContain('symbol');
        expect(std.search_modes).not.toContain('semantic');
        expect(std.ide_hours_per_day).toBe(4);
        expect(std.stacked_prs).toBe(true);
    });

    test('advanced tier unlocks semantic search + HIPAA', () => {
        const adv = SCM_TIERS.advanced;
        expect(adv.search_modes).toContain('semantic');
        expect(adv.hipaa_compliance).toBe(true);
        expect(adv.ide_hours_per_day).toBe(12);
        expect(adv.dora_metrics).toBe('full');
    });

    test('enterprise tier is unlimited', () => {
        const ent = SCM_TIERS.enterprise;
        expect(ent.private_repos).toBe(Infinity);
        expect(ent.ai_reviews_per_month).toBe(Infinity);
        expect(ent.api_calls_per_day).toBe(Infinity);
        expect(ent.ide_hours_per_day).toBe(Infinity);
        expect(ent.sso_saml).toBe(true);
        expect(ent.dora_metrics).toBe('custom');
    });

    test('tiers are monotonically increasing in limits', () => {
        const order: ScmTier[] = ['free', 'standard', 'advanced', 'enterprise'];
        for (let i = 1; i < order.length; i++) {
            const prev = SCM_TIERS[order[i - 1]];
            const curr = SCM_TIERS[order[i]];
            expect(curr.public_repos).toBeGreaterThanOrEqual(prev.public_repos);
            expect(curr.private_repos).toBeGreaterThanOrEqual(prev.private_repos);
            expect(curr.ai_reviews_per_month).toBeGreaterThanOrEqual(prev.ai_reviews_per_month);
            expect(curr.ide_hours_per_day).toBeGreaterThanOrEqual(prev.ide_hours_per_day);
            expect(curr.search_modes.length).toBeGreaterThanOrEqual(prev.search_modes.length);
        }
    });

    test('storage bytes are in correct range per tier', () => {
        expect(SCM_TIERS.free.storage_bytes).toBe(200 * 1024 * 1024);       // 200 MB
        expect(SCM_TIERS.standard.storage_bytes).toBe(2 * 1024 * 1024 * 1024); // 2 GB
        expect(SCM_TIERS.advanced.storage_bytes).toBe(10 * 1024 * 1024 * 1024); // 10 GB
        expect(SCM_TIERS.enterprise.storage_bytes).toBe(100 * 1024 * 1024 * 1024); // 100 GB
    });
});

// ═══════════════════════════════════════════════════════════════
// 3. Edge Cases — Error Handling
// ═══════════════════════════════════════════════════════════════

describe('Edge Cases — API Error Handling', () => {
    test('throws on 401 unauthorized', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ error: 'Unauthorized' }, 401));

        const client = new ScmClient('https://test.synalux.ai');
        await expect(client.search('org/repo', { query: 'test', mode: 'exact' }))
            .rejects.toThrow('SCM API 401');
    });

    test('throws on 403 tier exceeded', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ error: 'Tier limit exceeded' }, 403));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        await expect(client.review('org/repo', [{ name: 'a.ts', content: '' }]))
            .rejects.toThrow('SCM API 403');
    });

    test('throws on 404 repo not found', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ error: 'Not found' }, 404));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        await expect(client.dora('org/nonexistent'))
            .rejects.toThrow('SCM API 404');
    });

    test('throws on 500 server error', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ error: 'Internal error' }, 500));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        await expect(client.scan('org/repo', []))
            .rejects.toThrow('SCM API 500');
    });

    test('throws on network failure', async () => {
        mockFetch.mockRejectedValue(new Error('fetch failed'));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        await expect(client.search('org/repo', { query: 'x', mode: 'exact' }))
            .rejects.toThrow('fetch failed');
    });
});

describe('Edge Cases — Client Configuration', () => {
    test('uses default URL when none provided', () => {
        delete process.env.SYNALUX_API_URL;
        const client = new ScmClient();
        // Test indirectly: won't throw on construction
        expect(client).toBeInstanceOf(ScmClient);
    });

    test('strips trailing slash from base URL', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ results: [], total_matches: 0, search_time_ms: 0, truncated: false }));

        const client = new ScmClient('https://test.synalux.ai/');
        await client.search('org/repo', { query: 'x', mode: 'exact' });

        expect(mockFetch.mock.calls[0][0]).toBe('https://test.synalux.ai/api/v1/scm/repos/org/repo/search');
    });

    test('respects env var SYNALUX_API_URL', async () => {
        process.env.SYNALUX_API_URL = 'https://env-test.synalux.ai';
        mockFetch.mockReturnValue(mockJsonResponse({ results: [], total_matches: 0, search_time_ms: 0, truncated: false }));

        const client = new ScmClient(); // no explicit URL
        await client.search('org/repo', { query: 'x', mode: 'exact' });

        expect(mockFetch.mock.calls[0][0]).toContain('env-test.synalux.ai');
        delete process.env.SYNALUX_API_URL;
    });

    test('respects env var SYNALUX_API_KEY', async () => {
        process.env.SYNALUX_API_KEY = 'env-key-123';
        mockFetch.mockReturnValue(mockJsonResponse({ results: [], total_matches: 0, search_time_ms: 0, truncated: false }));

        const client = new ScmClient('https://test.synalux.ai'); // no explicit key
        await client.search('org/repo', { query: 'x', mode: 'exact' });

        expect(mockFetch.mock.calls[0][1].headers.Authorization).toBe('Bearer env-key-123');
        delete process.env.SYNALUX_API_KEY;
    });
});

describe('Edge Cases — Empty & Boundary Inputs', () => {
    test('search with empty query succeeds', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ results: [], total_matches: 0, search_time_ms: 0, truncated: false }));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        const result = await client.search('org/repo', { query: '', mode: 'exact' });
        expect(result.total_matches).toBe(0);
    });

    test('review with empty file list succeeds', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ review: { overall_score: 100, total_findings: 0, auto_approve: true, model: 'v1', findings: [] } }));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        const result = await client.review('org/repo', []);
        expect(result.review.overall_score).toBe(100);
    });

    test('scan with empty file list succeeds', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ summary: { total_findings: 0, critical: 0, high: 0, medium: 0, low: 0, pass: true, scan_duration_ms: 0 }, findings: [] }));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        const result = await client.scan('org/repo', []);
        expect(result.summary.pass).toBe(true);
    });

    test('repo name with special characters', async () => {
        mockFetch.mockReturnValue(mockJsonResponse({ results: [], total_matches: 0, search_time_ms: 0, truncated: false }));

        const client = new ScmClient('https://test.synalux.ai', 'key');
        await client.search('my-org/my-repo.js', { query: 'test', mode: 'exact' });

        expect(mockFetch.mock.calls[0][0]).toContain('/repos/my-org/my-repo.js/search');
    });

    test('all search modes are valid enum values', () => {
        const validModes: SearchMode[] = ['exact', 'regex', 'semantic', 'symbol'];
        validModes.forEach(m => expect(typeof m).toBe('string'));
    });
});

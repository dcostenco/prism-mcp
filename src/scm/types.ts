/**
 * SCM Types — Synalux Interface Re-exports
 * ==========================================
 * These interfaces mirror the Synalux SCM engines.
 * Only types live in Prism — implementations stay in Synalux.
 *
 * @see synalux-private/portal/src/lib/code-search.ts
 * @see synalux-private/portal/src/lib/ai-review.ts
 * @see synalux-private/portal/src/lib/security-scanner.ts
 * @see synalux-private/portal/src/lib/dora-metrics.ts
 */

// ── Code Search ─────────────────────────────────────────────

export type SearchMode = 'exact' | 'regex' | 'semantic' | 'symbol';

export interface SearchQuery {
    query: string;
    mode: SearchMode;
    repos?: string[];
    file_patterns?: string[];
    exclude_patterns?: string[];
    max_results?: number;
    context_lines?: number;
}

export interface SearchResult {
    file: string;
    repo_full_name: string;
    line_number: number;
    content: string;
    score: number;
    match_type: SearchMode;
}

export interface SearchResponse {
    results: SearchResult[];
    total_matches: number;
    search_time_ms: number;
    truncated: boolean;
}

// ── AI Review ───────────────────────────────────────────────

export type ReviewSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';

export interface AIReviewFinding {
    id: string;
    file: string;
    line_start: number;
    severity: ReviewSeverity;
    title: string;
    description: string;
    rule_id?: string;
}

export interface AIReviewResult {
    overall_score: number;
    total_findings: number;
    findings: AIReviewFinding[];
    auto_approve: boolean;
    model: string;
}

export interface HipaaResult {
    compliant: boolean;
    score: number;
    violations: AIReviewFinding[];
}

// ── Security Scanner ────────────────────────────────────────

export interface SecurityFinding {
    id: string;
    type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    title: string;
    description: string;
    file?: string;
    line?: number;
    remediation: string;
}

export interface ScanSummary {
    total_findings: number;
    critical: number;
    high: number;
    medium: number;
    low: number;
    pass: boolean;
    scan_duration_ms: number;
}

// ── DORA Metrics ────────────────────────────────────────────

export type DoraLevel = 'elite' | 'high' | 'medium' | 'low';

export interface DoraMetrics {
    deployment_frequency: { value: number; level: DoraLevel };
    lead_time: { value: number; level: DoraLevel };
    change_failure_rate: { value: number; level: DoraLevel };
    mttr: { value: number; level: DoraLevel };
    overall_level: DoraLevel;
    period: string;
    team_size: number;
}

// ── SCM Tiers ───────────────────────────────────────────────

export type ScmTier = 'free' | 'standard' | 'advanced' | 'enterprise';

export interface ScmTierLimits {
    public_repos: number;
    private_repos: number;
    collaborators_per_repo: number;
    storage_bytes: number;
    ai_reviews_per_month: number;
    api_calls_per_day: number;
    ci_minutes_per_month: number;
    webhooks_per_repo: number;
    ide_hours_per_day: number;
    stacked_prs: boolean;
    dora_metrics: 'none' | 'basic' | 'full' | 'custom';
    search_modes: SearchMode[];
    hipaa_compliance: boolean;
    sso_saml: boolean;
    /** Max concurrent VMs (see src/vm/types.ts VM_TIERS for full VM limits) */
    vm_concurrent: number;
    /** VMware/Parallels import support */
    vm_import: boolean;
    /** User-defined custom device parameters */
    custom_devices: boolean;
    /** Max deploys per day */
    deploys_per_day: number;
    /** Thin-client proxy to Synalux Cloud */
    thin_client: boolean;
}

export const SCM_TIERS: Record<ScmTier, ScmTierLimits> = {
    free: {
        public_repos: 3, private_repos: 1, collaborators_per_repo: 2,
        storage_bytes: 200 * 1024 * 1024, ai_reviews_per_month: 5,
        api_calls_per_day: 100, ci_minutes_per_month: 50,
        webhooks_per_repo: 2, ide_hours_per_day: 1,
        stacked_prs: false, dora_metrics: 'none',
        search_modes: ['exact'], hipaa_compliance: false, sso_saml: false,
        vm_concurrent: 1, vm_import: false, custom_devices: false, deploys_per_day: 3, thin_client: false,
    },
    standard: {
        public_repos: 20, private_repos: 10, collaborators_per_repo: 5,
        storage_bytes: 2 * 1024 * 1024 * 1024, ai_reviews_per_month: 50,
        api_calls_per_day: 2_000, ci_minutes_per_month: 500,
        webhooks_per_repo: 10, ide_hours_per_day: 4,
        stacked_prs: true, dora_metrics: 'basic',
        search_modes: ['exact', 'regex', 'symbol'], hipaa_compliance: false, sso_saml: false,
        vm_concurrent: 3, vm_import: true, custom_devices: true, deploys_per_day: 25, thin_client: true,
    },
    advanced: {
        public_repos: Infinity, private_repos: 50, collaborators_per_repo: 25,
        storage_bytes: 10 * 1024 * 1024 * 1024, ai_reviews_per_month: 500,
        api_calls_per_day: 5_000, ci_minutes_per_month: 5_000,
        webhooks_per_repo: 50, ide_hours_per_day: 12,
        stacked_prs: true, dora_metrics: 'full',
        search_modes: ['exact', 'regex', 'symbol', 'semantic'], hipaa_compliance: true, sso_saml: false,
        vm_concurrent: 8, vm_import: true, custom_devices: true, deploys_per_day: 100, thin_client: true,
    },
    enterprise: {
        public_repos: Infinity, private_repos: Infinity, collaborators_per_repo: Infinity,
        storage_bytes: 100 * 1024 * 1024 * 1024, ai_reviews_per_month: Infinity,
        api_calls_per_day: Infinity, ci_minutes_per_month: Infinity,
        webhooks_per_repo: Infinity, ide_hours_per_day: Infinity,
        stacked_prs: true, dora_metrics: 'custom',
        search_modes: ['exact', 'regex', 'symbol', 'semantic'], hipaa_compliance: true, sso_saml: true,
        vm_concurrent: Infinity, vm_import: true, custom_devices: true, deploys_per_day: Infinity, thin_client: true,
    },
};

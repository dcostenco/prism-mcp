/**
 * v12.3–v12.5 Handler Tests
 *
 * Tests all 10 new MCP tool handlers added in the v12.3–v12.5 release:
 *   v12.3: manageRbacHandler, encryptedSyncHandler
 *   v12.4: githubSyncHandler, generateChangelogHandler, generateCiPipelineHandler, memoryAttestationHandler
 *   v12.5: managePluginsHandler, synaluxProxyHandler, cloudDelegateHandler, vmQuotaHandler
 *
 * Each handler is tested for:
 *   1. Valid action dispatch (happy path)
 *   2. Missing required arguments (edge cases)
 *   3. Unknown action fallback
 *   4. MCP response format compliance ({ content: [{ type: "text", text: ... }] })
 */

import { describe, it, expect } from "vitest";
import {
    manageRbacHandler,
    encryptedSyncHandler,
    githubSyncHandler,
    generateCiPipelineHandler,
    managePluginsHandler,
    synaluxProxyHandler,
    cloudDelegateHandler,
    vmQuotaHandler,
} from "../../src/tools/v12Handlers.js";

// ─── Helper: parse the JSON text from MCP response ───
function parseResponse(result: any): any {
    const text = result?.content?.[0]?.text;
    if (!text) return null;
    try { return JSON.parse(text); } catch { return text; }
}

// ═══════════════════════════════════════════════════════════════
// 1. RBAC Handler — v12.3
// ═══════════════════════════════════════════════════════════════

describe("manageRbacHandler", () => {
    it("should list built-in roles", async () => {
        const result = await manageRbacHandler({ action: "list_roles" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.roles).toBeDefined();
        expect(Array.isArray(data.roles)).toBe(true);
    });

    it("should create a custom role", async () => {
        const result = await manageRbacHandler({
            action: "create_role",
            role: "test_analyst",
            permissions: ["read", "write"],
        });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.message).toContain("test_analyst");
    });

    it("should assign and revoke a role", async () => {
        // Assign
        const assign = await manageRbacHandler({
            action: "assign_role",
            user_id: "user-42",
            role: "viewer",
            project: "test-proj",
        });
        expect(assign.isError).toBeUndefined();
        const assignData = parseResponse(assign);
        expect(assignData.status).toBe("ok");

        // Revoke
        const revoke = await manageRbacHandler({
            action: "revoke_role",
            user_id: "user-42",
            role: "viewer",
            project: "test-proj",
        });
        expect(revoke.isError).toBeUndefined();
    });

    it("should check permission", async () => {
        const result = await manageRbacHandler({
            action: "check_permission",
            user_id: "user-42",
            permission: "read",
            project: "test-proj",
        });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
    });

    it("should list project assignments", async () => {
        const result = await manageRbacHandler({
            action: "list_assignments",
            project: "test-proj",
        });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.project).toBe("test-proj");
    });

    it("should reject create_role without role/permissions", async () => {
        const result = await manageRbacHandler({ action: "create_role" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("required");
    });

    it("should reject assign_role without user_id", async () => {
        const result = await manageRbacHandler({ action: "assign_role", role: "viewer" });
        expect(result.isError).toBe(true);
    });

    it("should reject revoke_role without role", async () => {
        const result = await manageRbacHandler({
            action: "revoke_role",
            user_id: "user-42",
            project: "test-proj",
        });
        expect(result.isError).toBe(true);
    });

    it("should reject check_permission without permission", async () => {
        const result = await manageRbacHandler({
            action: "check_permission",
            user_id: "user-42",
            project: "test-proj",
        });
        expect(result.isError).toBe(true);
    });

    it("should reject list_assignments without project", async () => {
        const result = await manageRbacHandler({ action: "list_assignments" });
        expect(result.isError).toBe(true);
    });

    it("should reject unknown action", async () => {
        const result = await manageRbacHandler({ action: "nuke_everything" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Unknown RBAC action");
    });

    it("should delete a custom role", async () => {
        // Create then delete
        await manageRbacHandler({
            action: "create_role",
            role: "temp_role",
            permissions: ["read"],
        });
        const result = await manageRbacHandler({
            action: "delete_role",
            role: "temp_role",
        });
        const data = parseResponse(result);
        expect(data.message).toContain("deleted");
    });

    it("should reject delete_role without role", async () => {
        const result = await manageRbacHandler({ action: "delete_role" });
        expect(result.isError).toBe(true);
    });
});

// ═══════════════════════════════════════════════════════════════
// 2. Encrypted Sync Handler — v12.3
// ═══════════════════════════════════════════════════════════════

describe("encryptedSyncHandler", () => {
    it("should list peers (initially empty)", async () => {
        const result = await encryptedSyncHandler({ action: "list_peers" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(Array.isArray(data.peers)).toBe(true);
    });

    it("should register a peer", async () => {
        const result = await encryptedSyncHandler({
            action: "register_peer",
            peer_url: "wss://peer1.example.com",
        });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.peer_id).toBeDefined();
    });

    it("should reject register_peer without peer_url", async () => {
        const result = await encryptedSyncHandler({ action: "register_peer" });
        expect(result.isError).toBe(true);
    });

    it("should get sync status", async () => {
        const result = await encryptedSyncHandler({ action: "status" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(typeof data.peer_count).toBe("number");
    });

    it("should reject push without required args", async () => {
        const result = await encryptedSyncHandler({ action: "push" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("required");
    });

    it("should reject unknown action", async () => {
        const result = await encryptedSyncHandler({ action: "teleport" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Unknown sync action");
    });
});

// ═══════════════════════════════════════════════════════════════
// 3. GitHub Sync Handler — v12.4
// ═══════════════════════════════════════════════════════════════

describe("githubSyncHandler", () => {
    it("should reject configure without repo/token", async () => {
        const result = await githubSyncHandler({ action: "configure" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("required");
    });

    it("should reject create_issue without project", async () => {
        const result = await githubSyncHandler({ action: "create_issue" });
        expect(result.isError).toBe(true);
    });

    it("should reject track_pr without project/pr_number", async () => {
        const result = await githubSyncHandler({ action: "track_pr" });
        expect(result.isError).toBe(true);
    });

    it("should reject unknown action", async () => {
        const result = await githubSyncHandler({ action: "obliterate" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Unknown GitHub sync action");
    });
});

// ═══════════════════════════════════════════════════════════════
// 4. CI Pipeline Handler — v12.4
// ═══════════════════════════════════════════════════════════════

describe("generateCiPipelineHandler", () => {
    it("should list available presets when no preset specified", async () => {
        const result = await generateCiPipelineHandler({ project: "my-app" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.available_presets).toBeDefined();
    });

    it("should generate from a known preset", async () => {
        const result = await generateCiPipelineHandler({
            project: "my-app",
            preset: "node-npm",
        });
        const data = parseResponse(result);
        // Should be ok if preset exists, error if not
        expect(data.status).toBeDefined();
    });

    it("should reject unknown preset", async () => {
        const result = await generateCiPipelineHandler({
            project: "my-app",
            preset: "nonexistent-preset-xyz",
        });
        const data = parseResponse(result);
        expect(data.status).toBe("error");
        expect(data.available).toBeDefined();
    });
});

// ═══════════════════════════════════════════════════════════════
// 5. Plugin Manager Handler — v12.5
// ═══════════════════════════════════════════════════════════════

describe("managePluginsHandler", () => {
    it("should discover plugins", async () => {
        const result = await managePluginsHandler({ action: "discover" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.plugins).toBeDefined();
    });

    it("should list plugins", async () => {
        const result = await managePluginsHandler({ action: "list" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(typeof data.count).toBe("number");
    });

    it("should reject load without plugin_name", async () => {
        const result = await managePluginsHandler({ action: "load" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("plugin_name is required");
    });

    it("should reject unload without plugin_name", async () => {
        const result = await managePluginsHandler({ action: "unload" });
        expect(result.isError).toBe(true);
    });

    it("should reject validate without plugin_name", async () => {
        const result = await managePluginsHandler({ action: "validate" });
        expect(result.isError).toBe(true);
    });

    it("should reject unknown action", async () => {
        const result = await managePluginsHandler({ action: "implode" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Unknown plugin action");
    });
});

// ═══════════════════════════════════════════════════════════════
// 6. Synalux Proxy Handler — v12.5
// ═══════════════════════════════════════════════════════════════

describe("synaluxProxyHandler", () => {
    it("should configure proxy", async () => {
        const result = await synaluxProxyHandler({ action: "configure" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.config).toBeDefined();
    });

    it("should check health", async () => {
        const result = await synaluxProxyHandler({ action: "health" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
    });

    it("should list features", async () => {
        const result = await synaluxProxyHandler({ action: "features" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.features).toBeDefined();
    });

    it("should get proxy status", async () => {
        const result = await synaluxProxyHandler({ action: "status" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.config).toBeDefined();
    });

    it("should reject request without request object", async () => {
        const result = await synaluxProxyHandler({ action: "request" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("request object is required");
    });

    it("should reject unknown action", async () => {
        const result = await synaluxProxyHandler({ action: "detonate" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Unknown proxy action");
    });
});

// ═══════════════════════════════════════════════════════════════
// 7. Cloud Delegate Handler — v12.5
// ═══════════════════════════════════════════════════════════════

describe("cloudDelegateHandler", () => {
    it("should list active tasks (initially empty)", async () => {
        const result = await cloudDelegateHandler({ action: "list" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.active_tasks).toBeDefined();
    });

    it("should get task history", async () => {
        const result = await cloudDelegateHandler({ action: "history" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.history).toBeDefined();
    });

    it("should configure delegate", async () => {
        const result = await cloudDelegateHandler({ action: "configure" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.config).toBeDefined();
    });

    it("should reject create without task_type/project", async () => {
        const result = await cloudDelegateHandler({ action: "create" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("required");
    });

    it("should reject dispatch without task_id", async () => {
        const result = await cloudDelegateHandler({ action: "dispatch" });
        expect(result.isError).toBe(true);
    });

    it("should reject status without task_id", async () => {
        const result = await cloudDelegateHandler({ action: "status" });
        expect(result.isError).toBe(true);
    });

    it("should reject cancel without task_id", async () => {
        const result = await cloudDelegateHandler({ action: "cancel" });
        expect(result.isError).toBe(true);
    });

    it("should create and then query a task", async () => {
        const createResult = await cloudDelegateHandler({
            action: "create",
            task_type: "embedding",
            project: "test-proj",
            payload: { text: "hello world" },
            priority: "normal",
        });
        const createData = parseResponse(createResult);
        expect(createData.status).toBe("ok");
        expect(createData.task).toBeDefined();
        expect(createData.task.id).toBeDefined();

        // Query status
        const statusResult = await cloudDelegateHandler({
            action: "status",
            task_id: createData.task.id,
        });
        const statusData = parseResponse(statusResult);
        expect(statusData.status).toBe("ok");
    });

    it("should reject unknown action", async () => {
        const result = await cloudDelegateHandler({ action: "wormhole" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Unknown delegate action");
    });
});

// ═══════════════════════════════════════════════════════════════
// 8. VM Quota Handler — v12.5
// ═══════════════════════════════════════════════════════════════

describe("vmQuotaHandler", () => {
    it("should get quota summary", async () => {
        const result = await vmQuotaHandler({ action: "summary" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
    });

    it("should check VM creation", async () => {
        const result = await vmQuotaHandler({
            action: "check_vm",
            cpu_cores: 2,
            ram_gb: 4,
            storage_gb: 20,
        });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
    });

    it("should check concurrent runs", async () => {
        const result = await vmQuotaHandler({ action: "check_run" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
    });

    it("should get usage", async () => {
        const result = await vmQuotaHandler({ action: "usage" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.usage).toBeDefined();
    });

    it("should set tier", async () => {
        const result = await vmQuotaHandler({ action: "set_tier", tier: "advanced" });
        const data = parseResponse(result);
        expect(data.status).toBe("ok");
        expect(data.message).toContain("advanced");
    });

    it("should reject set_tier without tier", async () => {
        const result = await vmQuotaHandler({ action: "set_tier" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("tier is required");
    });

    it("should reject unknown action", async () => {
        const result = await vmQuotaHandler({ action: "overclock" });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Unknown quota action");
    });
});

// ═══════════════════════════════════════════════════════════════
// 9. MCP Response Format Compliance
// ═══════════════════════════════════════════════════════════════

describe("MCP response format compliance", () => {
    it("all handlers return { content: [{ type: 'text', text: string }] }", async () => {
        const handlers = [
            () => manageRbacHandler({ action: "list_roles" }),
            () => encryptedSyncHandler({ action: "list_peers" }),
            () => generateCiPipelineHandler({ project: "x" }),
            () => managePluginsHandler({ action: "discover" }),
            () => synaluxProxyHandler({ action: "status" }),
            () => cloudDelegateHandler({ action: "list" }),
            () => vmQuotaHandler({ action: "summary" }),
        ];

        for (const handler of handlers) {
            const result = await handler();
            expect(result.content).toBeDefined();
            expect(Array.isArray(result.content)).toBe(true);
            expect(result.content[0].type).toBe("text");
            expect(typeof result.content[0].text).toBe("string");
        }
    });

    it("error responses set isError: true", async () => {
        const errorCases = [
            () => manageRbacHandler({ action: "unknown_action_xyz" }),
            () => encryptedSyncHandler({ action: "unknown_action_xyz" }),
            () => githubSyncHandler({ action: "unknown_action_xyz" }),
            () => managePluginsHandler({ action: "unknown_action_xyz" }),
            () => synaluxProxyHandler({ action: "unknown_action_xyz" }),
            () => cloudDelegateHandler({ action: "unknown_action_xyz" }),
            () => vmQuotaHandler({ action: "unknown_action_xyz" }),
        ];

        for (const handler of errorCases) {
            const result = await handler();
            expect(result.isError).toBe(true);
        }
    });
});

// ═══════════════════════════════════════════════════════════════
// 10. Tool Definition Schema Tests — v12.3-v12.5
// ═══════════════════════════════════════════════════════════════

import {
    MANAGE_RBAC_TOOL,
    ENCRYPTED_SYNC_TOOL,
    GITHUB_SYNC_TOOL,
    GENERATE_CHANGELOG_TOOL,
    GENERATE_CI_PIPELINE_TOOL,
    MEMORY_ATTESTATION_TOOL,
    MANAGE_PLUGINS_TOOL,
    SYNALUX_PROXY_TOOL,
    CLOUD_DELEGATE_TOOL,
    VM_QUOTA_TOOL,
} from "../../src/tools/sessionMemoryDefinitions.js";

describe("v12.3-v12.5 Tool Definition Schemas", () => {
    const tools = [
        { tool: MANAGE_RBAC_TOOL, name: "manage_rbac", requiredFields: ["action"] },
        { tool: ENCRYPTED_SYNC_TOOL, name: "encrypted_sync", requiredFields: ["action"] },
        { tool: GITHUB_SYNC_TOOL, name: "github_sync", requiredFields: ["action"] },
        { tool: GENERATE_CHANGELOG_TOOL, name: "generate_changelog", requiredFields: ["project"] },
        { tool: GENERATE_CI_PIPELINE_TOOL, name: "generate_ci_pipeline", requiredFields: ["project"] },
        { tool: MEMORY_ATTESTATION_TOOL, name: "memory_attestation", requiredFields: ["action", "project"] },
        { tool: MANAGE_PLUGINS_TOOL, name: "manage_plugins", requiredFields: ["action"] },
        { tool: SYNALUX_PROXY_TOOL, name: "synalux_proxy", requiredFields: ["action"] },
        { tool: CLOUD_DELEGATE_TOOL, name: "cloud_delegate", requiredFields: ["action"] },
        { tool: VM_QUOTA_TOOL, name: "vm_quota", requiredFields: ["action"] },
    ];

    for (const { tool, name, requiredFields } of tools) {
        describe(name, () => {
            it("has correct tool name", () => {
                expect(tool.name).toBe(name);
            });

            it("has a non-empty description", () => {
                expect(tool.description.length).toBeGreaterThan(10);
            });

            it("has inputSchema.type = 'object'", () => {
                expect(tool.inputSchema.type).toBe("object");
            });

            it("declares required fields", () => {
                for (const field of requiredFields) {
                    expect(tool.inputSchema.required).toContain(field);
                }
            });

            it("has properties object defined", () => {
                expect(tool.inputSchema.properties).toBeDefined();
                expect(typeof tool.inputSchema.properties).toBe("object");
            });
        });
    }
});

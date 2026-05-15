import { describe, it, expect } from "vitest";
import { pickLocalModel, MODEL_TIERS, fmtGb } from "../../src/utils/modelPicker.js";

const GB = 1024 ** 3;

describe("pickLocalModel", () => {
    it("picks 32B when ≥24 GB free", () => {
        const choice = pickLocalModel(30 * GB);
        expect(choice?.tag).toBe("prism-coder:32b");
    });

    it("picks 14B when 12–23 GB free", () => {
        expect(pickLocalModel(12 * GB)?.tag).toBe("prism-coder:14b");
        expect(pickLocalModel(20 * GB)?.tag).toBe("prism-coder:14b");
        expect(pickLocalModel(23 * GB)?.tag).toBe("prism-coder:14b");
    });

    it("picks 8B when 7–11 GB free", () => {
        expect(pickLocalModel(7 * GB)?.tag).toBe("prism-coder:8b");
        expect(pickLocalModel(11 * GB)?.tag).toBe("prism-coder:8b");
    });

    it("picks 1.7B when 3–6 GB free", () => {
        expect(pickLocalModel(3 * GB)?.tag).toBe("prism-coder:1b7");
        expect(pickLocalModel(6 * GB)?.tag).toBe("prism-coder:1b7");
    });

    it("returns null below 3 GB free", () => {
        expect(pickLocalModel(2 * GB)).toBeNull();
        expect(pickLocalModel(0)).toBeNull();
        expect(pickLocalModel(-1)).toBeNull();
        expect(pickLocalModel(Number.NaN)).toBeNull();
    });

    it("honors ceiling — '14b' forbids 32B even on a 64 GB box", () => {
        const choice = pickLocalModel(64 * GB, "14b");
        expect(choice?.tag).toBe("prism-coder:14b");
    });

    it("ceiling '8b' on 30 GB still picks 8B", () => {
        expect(pickLocalModel(30 * GB, "8b")?.tag).toBe("prism-coder:8b");
    });

    it("ceiling '1b7' on 64 GB picks 1.7B", () => {
        expect(pickLocalModel(64 * GB, "1b7")?.tag).toBe("prism-coder:1b7");
    });

    it("respects `available` whitelist — skips tiers not pulled", () => {
        // 30 GB free would normally pick 32B, but if only 14B is installed:
        const available = new Set(["prism-coder:14b", "prism-coder:8b"]);
        const choice = pickLocalModel(30 * GB, undefined, available);
        expect(choice?.tag).toBe("prism-coder:14b");
    });

    it("returns null when nothing in `available` matches", () => {
        const available = new Set(["llama3:8b"]); // unrelated
        expect(pickLocalModel(30 * GB, undefined, available)).toBeNull();
    });

    it("MODEL_TIERS is ordered largest → smallest", () => {
        for (let i = 1; i < MODEL_TIERS.length; i++) {
            expect(MODEL_TIERS[i].minFreeGb).toBeLessThan(MODEL_TIERS[i - 1].minFreeGb);
        }
    });
});

describe("fmtGb", () => {
    it("formats binary GB to one decimal", () => {
        expect(fmtGb(12 * GB)).toBe("12.0 GB");
        expect(fmtGb(1.5 * GB)).toBe("1.5 GB");
        expect(fmtGb(0)).toBe("0.0 GB");
    });
});

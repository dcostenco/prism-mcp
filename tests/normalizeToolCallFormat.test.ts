import { describe, it, expect } from "vitest";
import { normalizeToolCallFormat } from "../src/utils/normalizeToolCallFormat.js";

const CANONICAL = (name: string, args: Record<string, unknown>) =>
  `<tool_call>\n${JSON.stringify({ name, arguments: args })}\n</tool_call>`;

describe("normalizeToolCallFormat", () => {
  it("rewrites plural-wrapper + XML-attr to canonical (v18-clean-epoch0 emission)", () => {
    const input =
      '<tool_calls><tool_call name="get_weather"><param name="city" value="Paris"/></tool_call></tool_calls>';
    expect(normalizeToolCallFormat(input)).toBe(
      CANONICAL("get_weather", { city: "Paris" })
    );
  });

  it("rewrites CJK-bracket variant (prev-20260503-1831 emission)", () => {
    const input =
      "〈tool_calls〉〈tool_call function_name=\"get_weather\"〉〈param name=\"city\" value=\"Paris\" /〉〈/tool_call〉〈/tool_calls〉";
    expect(normalizeToolCallFormat(input)).toBe(
      CANONICAL("get_weather", { city: "Paris" })
    );
  });

  it("rewrites bare XML-attr <tool_call> without plural wrapper", () => {
    const input =
      '<tool_call name="get_stock_info"><param name="symbol" value="AAPL"/></tool_call>';
    expect(normalizeToolCallFormat(input)).toBe(
      CANONICAL("get_stock_info", { symbol: "AAPL" })
    );
  });

  it("handles multiple tool calls inside a single plural wrapper", () => {
    const input =
      '<tool_calls>' +
      '<tool_call name="get_stock_info"><param name="symbol" value="AAPL"/></tool_call>' +
      '<tool_call name="get_stock_info"><param name="symbol" value="TSLA"/></tool_call>' +
      '</tool_calls>';
    const out = normalizeToolCallFormat(input);
    expect(out).toContain(CANONICAL("get_stock_info", { symbol: "AAPL" }));
    expect(out).toContain(CANONICAL("get_stock_info", { symbol: "TSLA" }));
    // The two canonical blocks should be the entire output
    expect(out).toBe(
      CANONICAL("get_stock_info", { symbol: "AAPL" }) +
        CANONICAL("get_stock_info", { symbol: "TSLA" })
    );
  });

  it("rewrites <functioncall> with stringified arguments (v18bfcl emission)", () => {
    const input = `<functioncall> {"name": "get_weather", "arguments": '{"city": "Paris"}'}`;
    expect(normalizeToolCallFormat(input)).toBe(
      CANONICAL("get_weather", { city: "Paris" })
    );
  });

  it("rewrites <functioncall> with object arguments", () => {
    const input = `<functioncall> {"name": "get_weather", "arguments": {"city": "Tokyo"}}`;
    expect(normalizeToolCallFormat(input)).toBe(
      CANONICAL("get_weather", { city: "Tokyo" })
    );
  });

  it("passes canonical input through unchanged", () => {
    const canonical = CANONICAL("get_weather", { city: "Paris" });
    expect(normalizeToolCallFormat(canonical)).toBe(canonical);
  });

  it("handles multiple params correctly", () => {
    const input =
      '<tool_call name="search"><param name="q" value="hello"/><param name="limit" value="10"/></tool_call>';
    expect(normalizeToolCallFormat(input)).toBe(
      CANONICAL("search", { q: "hello", limit: "10" })
    );
  });

  it("preserves surrounding text when only part of the response is a tool call", () => {
    const input =
      'Sure, let me check.\n<tool_calls><tool_call name="get_weather"><param name="city" value="Paris"/></tool_call></tool_calls>\nDone.';
    const out = normalizeToolCallFormat(input);
    expect(out).toContain("Sure, let me check.");
    expect(out).toContain(CANONICAL("get_weather", { city: "Paris" }));
    expect(out).toContain("Done.");
  });

  it("returns empty string unchanged", () => {
    expect(normalizeToolCallFormat("")).toBe("");
  });

  it("returns plain text without tool calls unchanged", () => {
    expect(normalizeToolCallFormat("Hello world.")).toBe("Hello world.");
  });

  it("leaves malformed <functioncall> unchanged when JSON is unparseable", () => {
    const input = "<functioncall> {not json}";
    expect(normalizeToolCallFormat(input)).toBe(input);
  });
});

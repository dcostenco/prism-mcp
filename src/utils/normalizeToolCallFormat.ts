/**
 * normalizeToolCallFormat — coerce stochastic Qwen2.5-Coder tool-call
 * formats into the canonical singular-wrapper JSON form.
 *
 * v18-clean SFT models occasionally emit non-canonical formats despite
 * training data being clean canonical. Three known variants in the wild:
 *
 *   1. plural wrapper + XML-attr params:
 *      <tool_calls><tool_call name="X"><param name="Y" value="Z"/></tool_call></tool_calls>
 *   2. CJK angle brackets:
 *      〈tool_calls〉〈tool_call function_name="X"〉〈param name="Y" value="Z" /〉〈/tool_call〉〈/tool_calls〉
 *   3. <functioncall> with stringified arguments:
 *      <functioncall> {"name":"X","arguments":'{"Y":"Z"}'}
 *
 * Each is rewritten to:
 *      <tool_call>
 *      {"name":"X","arguments":{"Y":"Z"}}
 *      </tool_call>
 *
 * Canonical inputs pass through unchanged.
 */

const PARAM_ATTR_RE = /<param\s+name="([^"]+)"\s+value="([^"]*)"\s*\/?>/gi;

const XML_ATTR_TOOL_CALL_RE =
  /<tool_call\s+(?:name|function_name)="([^"]+)"[^>]*>([\s\S]*?)<\/tool_call>/gi;

const PLURAL_WRAPPER_RE = /<tool_calls>\s*([\s\S]*?)\s*<\/tool_calls>/gi;

const FUNCTION_CALL_RE =
  /<functioncall>\s*(\{[\s\S]*?\})(?=\s*$|\s*<|\s*\n\s*\n)/gi;

function rewriteXmlAttrToolCalls(input: string): string {
  return input.replace(XML_ATTR_TOOL_CALL_RE, (_match, name: string, body: string) => {
    const args: Record<string, string> = {};
    PARAM_ATTR_RE.lastIndex = 0;
    let pm: RegExpExecArray | null;
    while ((pm = PARAM_ATTR_RE.exec(body)) !== null) {
      args[pm[1]] = pm[2];
    }
    return `<tool_call>\n${JSON.stringify({ name, arguments: args })}\n</tool_call>`;
  });
}

export function normalizeToolCallFormat(text: string): string {
  if (!text) return text;
  let s = text;

  // Step 1: replace CJK angle brackets so subsequent regex matches
  if (s.includes("〈") || s.includes("〉")) {
    s = s.replace(/〈/g, "<").replace(/〉/g, ">");
  }

  // Step 2: unwrap plural <tool_calls>...</tool_calls> and rewrite each inner call
  s = s.replace(PLURAL_WRAPPER_RE, (_m, inner: string) => rewriteXmlAttrToolCalls(inner));

  // Step 3: rewrite any remaining bare XML-attr <tool_call> outside plural wrapper
  s = rewriteXmlAttrToolCalls(s);

  // Step 4: rewrite <functioncall> {...} forms with stringified arguments
  s = s.replace(FUNCTION_CALL_RE, (match, jsonish: string) => {
    try {
      // Strip single-quote wrapping that occasionally surrounds nested JSON args
      const cleaned = jsonish
        .replace(/"arguments":\s*'(\{[\s\S]*?\})'/g, '"arguments": $1')
        .replace(/"arguments":\s*'([^']*)'/g, '"arguments": "$1"');
      const obj = JSON.parse(cleaned);
      if (obj && typeof obj.name === "string") {
        const args = typeof obj.arguments === "string"
          ? JSON.parse(obj.arguments)
          : (obj.arguments ?? {});
        return `<tool_call>\n${JSON.stringify({ name: obj.name, arguments: args })}\n</tool_call>`;
      }
    } catch {
      /* fall through — leave original */
    }
    return match;
  });

  return s;
}

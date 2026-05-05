#!/usr/bin/env python3
"""Generate multi-turn step-2 SFT data for v18.1 surgical fix.

Targets the BFCL multi_turn weakness: model fails at step-2 tool selection
after first tool returns. Generates examples where the tokens AFTER the tool
response demonstrate correct followup behavior — either next tool call or
final answer.

Output format matches v18 dominant Qwen2-style chatml:
  <|im_start|>system\n...tools...\n<|im_end|>
  <|im_start|>user\n...query...\n<|im_end|>
  <|im_start|>assistant\n<tool_call>{...}</tool_call>\n<|im_end|>
  <|im_start|>tool\n<tool_response>{...}</tool_response>\n<|im_end|>
  <|im_start|>assistant\n<tool_call>{...}</tool_call> OR <text answer>\n<|im_end|>

5 patterns covered (target ~150 examples each = ~750 rows):
  P1: tool_chain          — search → fetch_detail
  P2: error_retry         — first call errors, retry with fix
  P3: insufficient_data   — first call returns empty, ask user / try alt tool
  P4: synthesize_final    — first call succeeds, summarize as text (no more tools)
  P5: three_step_chain    — A → B → final synthesis

Usage:
  python3 generate_multiturn_step2_sft.py --out data/v18_1_multiturn.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

random.seed(42)

# ──────────────────────────────────────────────────────────────────────────
# Tool catalog — realistic mix matching BFCL multi_turn schemas (file ops,
# search, math, calendar, weather, ticket system, math, http). Schemas are
# loose but type-correct.
# ──────────────────────────────────────────────────────────────────────────

TOOLS = {
    "search_web": {
        "name": "search_web",
        "description": "Search the web for relevant pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 5},
            },
            "required": ["query"],
        },
    },
    "fetch_page": {
        "name": "fetch_page",
        "description": "Fetch the full text content of a single web page by URL.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    "list_files": {
        "name": "list_files",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["c", "f"], "default": "c"},
            },
            "required": ["city"],
        },
    },
    "get_forecast": {
        "name": "get_forecast",
        "description": "Get N-day weather forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city", "days"],
        },
    },
    "create_ticket": {
        "name": "create_ticket",
        "description": "Create a support ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "med", "high"]},
            },
            "required": ["title"],
        },
    },
    "assign_ticket": {
        "name": "assign_ticket",
        "description": "Assign a ticket to a user.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "integer"},
                "assignee": {"type": "string"},
            },
            "required": ["ticket_id", "assignee"],
        },
    },
    "list_events": {
        "name": "list_events",
        "description": "List calendar events on a given date (ISO YYYY-MM-DD).",
        "parameters": {
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
        },
    },
    "create_event": {
        "name": "create_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "starts_at": {"type": "string"},
                "duration_min": {"type": "integer"},
            },
            "required": ["title", "starts_at"],
        },
    },
    "calculate": {
        "name": "calculate",
        "description": "Evaluate a math expression and return the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    },
    "translate": {
        "name": "translate",
        "description": "Translate text between languages.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "to": {"type": "string"},
            },
            "required": ["text", "to"],
        },
    },
}


def render_system(tool_subset: list[dict]) -> str:
    tools_json = json.dumps([{"type": "function", "function": t} for t in tool_subset], ensure_ascii=False)
    return (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tools_json}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within "
        "<tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


def assemble(messages: list[tuple[str, str]]) -> str:
    """Render a list of (role, content) into chatml."""
    parts = []
    for role, content in messages:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def tool_call(name: str, arguments: dict) -> str:
    return f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(arguments, ensure_ascii=False)}}}\n</tool_call>'


def tool_response(payload) -> str:
    return f"<tool_response>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_response>"


# ──────────────────────────────────────────────────────────────────────────
# Pattern generators — each returns a single training row dict {"text": ...}
# ──────────────────────────────────────────────────────────────────────────

def p1_tool_chain() -> dict:
    """Pattern 1: search → fetch_detail."""
    queries = [
        ("electric vehicle range comparison 2025", "https://example.com/ev-2025-roundup",
         "Top EVs 2025: Lucid Air 516mi, Tesla S 405mi, Mercedes EQS 350mi."),
        ("rust async runtime benchmarks", "https://blog.rust-lang.org/async-2025",
         "tokio leads at 1.2M req/s, smol close behind, async-std deprecated."),
        ("best espresso beans 2025", "https://coffeereview.com/top-2025",
         "Top picks: Counter Culture Hologram, Stumptown Hairbender, Verve Streetlevel."),
        ("kubernetes 1.30 release notes", "https://kubernetes.io/blog/2025-04-release",
         "v1.30 adds JobSet GA, sidecar containers stable, removes deprecated PodSecurityPolicy."),
    ]
    q, url, content = random.choice(queries)
    sys_msg = render_system([TOOLS["search_web"], TOOLS["fetch_page"]])
    msgs = [
        ("system", sys_msg),
        ("user", f"Find me detailed info about {q}."),
        ("assistant", tool_call("search_web", {"query": q, "limit": 3})),
        ("tool", tool_response({"results": [{"title": "Top result", "url": url, "snippet": content[:80]}]})),
        ("assistant", tool_call("fetch_page", {"url": url})),
    ]
    return {"text": assemble(msgs)}


def p2_error_retry() -> dict:
    """Pattern 2: first call errors, model retries with fixed args."""
    cases = [
        ("Read the file at /etc/config.yml.", "read_file", {"path": "/etc/config.yml"},
         {"error": "permission denied"},
         {"path": "/usr/local/etc/config.yml"}),  # retry alternative path
        ("What's the weather in Pari?", "get_weather", {"city": "Pari", "units": "c"},
         {"error": "city not found: Pari"},
         {"city": "Paris", "units": "c"}),
        ("Calculate 12*(7+3", "calculate", {"expr": "12*(7+3"},
         {"error": "syntax error: unbalanced parenthesis"},
         {"expr": "12*(7+3)"}),
        ("List events on 2026/05/03.", "list_events", {"date": "2026/05/03"},
         {"error": "invalid date format, expected YYYY-MM-DD"},
         {"date": "2026-05-03"}),
    ]
    user_q, tool, bad_args, err, fixed_args = random.choice(cases)
    sys_msg = render_system([TOOLS[tool]])
    msgs = [
        ("system", sys_msg),
        ("user", user_q),
        ("assistant", tool_call(tool, bad_args)),
        ("tool", tool_response(err)),
        ("assistant", tool_call(tool, fixed_args)),
    ]
    return {"text": assemble(msgs)}


def p3_insufficient_data() -> dict:
    """Pattern 3: first call returns empty/insufficient → model asks for clarification."""
    cases = [
        ("Find me events tomorrow.", "list_events", {"date": "2026-05-03"},
         {"events": []},
         "I checked your calendar for tomorrow (2026-05-03) and there are no events scheduled. Would you like me to check a different date or create a new event?"),
        ("Search for 'foobarbazquux'.", "search_web", {"query": "foobarbazquux", "limit": 5},
         {"results": []},
         "The search returned no results for 'foobarbazquux'. Could you give me a different query or check the spelling?"),
        ("Read /var/log/missing.log.", "read_file", {"path": "/var/log/missing.log"},
         {"content": ""},
         "The file at /var/log/missing.log exists but is empty — there's no content to summarize. Would you like me to look at a different log file?"),
    ]
    user_q, tool, args, response, followup = random.choice(cases)
    sys_msg = render_system([TOOLS[tool]])
    msgs = [
        ("system", sys_msg),
        ("user", user_q),
        ("assistant", tool_call(tool, args)),
        ("tool", tool_response(response)),
        ("assistant", followup),
    ]
    return {"text": assemble(msgs)}


def p4_synthesize_final() -> dict:
    """Pattern 4: tool succeeds → model gives natural-language answer (no more tools)."""
    cases = [
        ("What's the weather in Tokyo?", "get_weather", {"city": "Tokyo", "units": "c"},
         {"temp_c": 18, "condition": "partly cloudy", "humidity": 62},
         "It's currently 18°C and partly cloudy in Tokyo, with humidity at 62%."),
        ("Translate 'good morning' to Spanish.", "translate", {"text": "good morning", "to": "es"},
         {"translated": "buenos días"},
         "'Good morning' in Spanish is **buenos días**."),
        ("How much is 12.5% of 4800?", "calculate", {"expr": "0.125 * 4800"},
         {"result": 600},
         "12.5% of 4800 is **600**."),
        ("Get the 3-day forecast for Berlin.", "get_forecast", {"city": "Berlin", "days": 3},
         {"forecast": [{"day": 1, "high": 22, "low": 14}, {"day": 2, "high": 19, "low": 12}, {"day": 3, "high": 16, "low": 10}]},
         "Berlin 3-day forecast:\n- Day 1: high 22°C / low 14°C\n- Day 2: high 19°C / low 12°C\n- Day 3: high 16°C / low 10°C"),
    ]
    user_q, tool, args, response, answer = random.choice(cases)
    sys_msg = render_system([TOOLS[tool]])
    msgs = [
        ("system", sys_msg),
        ("user", user_q),
        ("assistant", tool_call(tool, args)),
        ("tool", tool_response(response)),
        ("assistant", answer),
    ]
    return {"text": assemble(msgs)}


def p5_three_step() -> dict:
    """Pattern 5: A → B → final synthesis (3-step plan)."""
    cases = [
        # create_ticket → assign_ticket → confirmation
        ("Create a high-priority ticket about the broken CI and assign it to dmitri.",
         "create_ticket", {"title": "CI is broken", "priority": "high"},
         {"ticket_id": 4521, "status": "open"},
         "assign_ticket", {"ticket_id": 4521, "assignee": "dmitri"},
         {"ticket_id": 4521, "assignee": "dmitri", "status": "assigned"},
         "Done — ticket #4521 'CI is broken' was created with high priority and assigned to dmitri."),
        # list_files → read_file → summary
        ("Look in /tmp/reports and read the most recent file.",
         "list_files", {"path": "/tmp/reports"},
         {"files": [{"name": "weekly.txt", "mtime": 1714665600}, {"name": "daily.txt", "mtime": 1714752000}]},
         "read_file", {"path": "/tmp/reports/daily.txt"},
         {"content": "Total runs: 142. Pass rate: 98.6%. Avg latency: 230ms."},
         "The most recent file (daily.txt) shows 142 total runs with a 98.6% pass rate and 230ms average latency."),
    ]
    case = random.choice(cases)
    user_q, t1, a1, r1, t2, a2, r2, final = case
    sys_msg = render_system([TOOLS[t1], TOOLS[t2]])
    msgs = [
        ("system", sys_msg),
        ("user", user_q),
        ("assistant", tool_call(t1, a1)),
        ("tool", tool_response(r1)),
        ("assistant", tool_call(t2, a2)),
        ("tool", tool_response(r2)),
        ("assistant", final),
    ]
    return {"text": assemble(msgs)}


# ──────────────────────────────────────────────────────────────────────────
# Disambiguation + abstention patterns — folded into the surgical pass to
# fix all three BFCL bottlenecks (multi_turn, disambiguation, irrelevance)
# in one SFT pass. Without these, multi-turn alone caps around top-5 7B.
# ──────────────────────────────────────────────────────────────────────────

# Confused tool pairs from session memory TODOs — model conflates these.
# Each entry: (correct_tool, wrong_tool, prompt_template_for_correct)
CONFUSED_PAIRS = [
    # search vs fetch — search returns list, fetch returns one page
    ("search_web", "fetch_page", [
        "Find articles about {topic}.",
        "What's the latest news on {topic}?",
        "I want to discover sources about {topic}.",
        "Look up information on {topic}.",
    ]),
    ("fetch_page", "search_web", [
        "Get the full content of {url}.",
        "Read the page at {url}.",
        "Pull the article from {url}.",
        "Show me what's at {url}.",
    ]),
    # list vs read — list returns directory, read returns content
    ("list_files", "read_file", [
        "What files are in {path}?",
        "Show me the contents of the {path} directory.",
        "List everything under {path}.",
        "What's inside {path}?",
    ]),
    ("read_file", "list_files", [
        "Show me the contents of the file {path}.",
        "Read {path} and tell me what's in it.",
        "What does {path} say?",
        "Open {path} and show me.",
    ]),
    # weather vs forecast
    ("get_weather", "get_forecast", [
        "What's the weather in {city} right now?",
        "Is it raining in {city}?",
        "Current conditions in {city}?",
        "How's the weather today in {city}?",
    ]),
    ("get_forecast", "get_weather", [
        "What's the weather going to be in {city} for the next {days} days?",
        "Give me a {days}-day forecast for {city}.",
        "Will it rain in {city} this week?",
        "Predict the weather for {city} over {days} days.",
    ]),
    # list_events vs create_event
    ("list_events", "create_event", [
        "What's on my calendar for {date}?",
        "Show me my schedule on {date}.",
        "Any meetings on {date}?",
        "What appointments do I have on {date}?",
    ]),
    ("create_event", "list_events", [
        "Add a meeting '{title}' on {starts_at}.",
        "Schedule '{title}' for {starts_at}.",
        "Book {title} at {starts_at}.",
        "Put '{title}' on my calendar at {starts_at}.",
    ]),
]

TOPICS = ["climate policy", "AI safety regulation", "next-gen batteries",
          "quantum networking", "remote work trends", "ocean acidification",
          "longevity research", "satellite internet", "robotics in healthcare"]
URLS = ["https://example.com/article-1", "https://blog.acme.com/2025/post",
        "https://news.io/q/breaking", "https://docs.io/guide", "https://research.org/2025/p3"]
PATHS = ["/tmp/data", "/var/log", "/home/admin/projects", "/etc/configs",
         "/opt/cache", "/usr/local/share", "/root/scripts"]
CITIES = ["Tokyo", "Paris", "Berlin", "São Paulo", "Mumbai", "Seattle",
          "Cairo", "Reykjavik", "Singapore", "Mexico City"]
DATES = ["2026-05-15", "2026-06-01", "2026-07-04", "2026-08-22", "2026-12-25"]


def p6_disambiguation() -> dict:
    """Pattern 6: prompt looks ambiguous but only ONE tool is correct.

    Generates single-turn correct tool selection between confused pairs.
    No tool response — just user query → correct tool call.
    """
    correct, wrong, templates = random.choice(CONFUSED_PAIRS)
    template = random.choice(templates)
    args = {}
    if "{topic}" in template:
        topic = random.choice(TOPICS)
        template = template.replace("{topic}", topic)
        args = {"query": topic, "limit": 5} if correct == "search_web" else {"url": random.choice(URLS)}
    elif "{url}" in template:
        url = random.choice(URLS)
        template = template.replace("{url}", url)
        args = {"url": url}
    elif "{path}" in template:
        path = random.choice(PATHS)
        template = template.replace("{path}", path)
        args = {"path": path}
    elif "{city}" in template and "{days}" in template:
        city = random.choice(CITIES)
        days = random.choice([3, 5, 7, 10])
        template = template.replace("{city}", city).replace("{days}", str(days))
        args = {"city": city, "days": days}
    elif "{city}" in template:
        city = random.choice(CITIES)
        template = template.replace("{city}", city)
        args = {"city": city, "units": "c"}
    elif "{date}" in template:
        date = random.choice(DATES)
        template = template.replace("{date}", date)
        args = {"date": date}
    elif "{title}" in template and "{starts_at}" in template:
        title = random.choice(["Standup", "Design Review", "1:1", "Demo", "Planning"])
        starts = random.choice(["2026-05-15T10:00", "2026-06-01T14:30", "2026-07-04T09:00"])
        template = template.replace("{title}", title).replace("{starts_at}", starts)
        args = {"title": title, "starts_at": starts}
    sys_msg = render_system([TOOLS[correct], TOOLS[wrong]])  # both visible — model must pick correct
    msgs = [
        ("system", sys_msg),
        ("user", template),
        ("assistant", tool_call(correct, args)),
    ]
    return {"text": assemble(msgs)}


def p7_abstention() -> dict:
    """Pattern 7: no tool needed — model should answer directly without calling.

    Critical for BFCL irrelevance/relevance categories. Tools are visible but
    the user query is conversational/factual/opinion — model must abstain
    from tool calls and respond in plain text.
    """
    cases = [
        ("Tell me a joke about programmers.",
         [TOOLS["calculate"], TOOLS["get_weather"]],
         "Why do programmers prefer dark mode? Because light attracts bugs!"),
        ("What is the capital of France?",
         [TOOLS["search_web"], TOOLS["translate"]],
         "The capital of France is Paris."),
        ("Why is the sky blue?",
         [TOOLS["get_weather"], TOOLS["search_web"]],
         "The sky appears blue because of Rayleigh scattering — shorter blue wavelengths of sunlight scatter more in Earth's atmosphere than longer red ones."),
        ("Hello, how are you?",
         [TOOLS["get_weather"], TOOLS["calculate"]],
         "I'm doing well, thanks for asking! How can I help you today?"),
        ("What's the meaning of life?",
         [TOOLS["search_web"], TOOLS["translate"]],
         "That's a deep philosophical question. Different traditions answer it differently — meaning may come from relationships, contribution, growth, or simply the act of asking. What does it mean to you?"),
        ("Can you help me think through a career decision?",
         [TOOLS["search_web"], TOOLS["create_event"]],
         "Of course — I'd be happy to help you think it through. Tell me about the options you're weighing and what feels uncertain."),
        ("What's 2 + 2?",
         [TOOLS["search_web"], TOOLS["translate"]],
         "2 + 2 = 4."),
        ("Recommend a good book.",
         [TOOLS["search_web"], TOOLS["get_weather"]],
         "It depends on what genre you enjoy — for fiction, *Project Hail Mary* by Andy Weir is widely loved; for nonfiction, *Thinking, Fast and Slow* by Kahneman is excellent. What kind of book are you in the mood for?"),
    ]
    user_q, tools, answer = random.choice(cases)
    sys_msg = render_system(tools)
    msgs = [
        ("system", sys_msg),
        ("user", user_q),
        ("assistant", answer),
    ]
    return {"text": assemble(msgs)}


PATTERNS = [
    ("p1_tool_chain", p1_tool_chain, 350),
    ("p2_error_retry", p2_error_retry, 250),
    ("p3_insufficient_data", p3_insufficient_data, 250),
    ("p4_synthesize_final", p4_synthesize_final, 350),
    ("p5_three_step", p5_three_step, 200),
    ("p6_disambiguation", p6_disambiguation, 600),
    ("p7_abstention", p7_abstention, 500),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiply target counts (default 1.0 ≈ 800 rows)")
    args = ap.parse_args()

    rows = []
    for name, fn, count in PATTERNS:
        n = int(count * args.scale)
        for _ in range(n):
            rows.append(fn())
        print(f"  {name}: {n} rows")

    random.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

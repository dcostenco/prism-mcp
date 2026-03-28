---
description: How to write correct GitHub README anchor links
---

# GitHub Anchor Link Rules

GitHub uses the `github-slugger` package to generate heading anchors.
**Always derive anchors from the actual heading text using these rules — never guess.**

## The Algorithm

Given a heading like `## 🚀 Quick Start`, the anchor is computed as:

1. **Lowercase** the full heading text
2. **Strip** any character that is NOT `[a-z0-9 \-]`:
   - Emoji are stripped (they are not `\w` chars)
   - Punctuation like `?`, `'`, `!` is stripped
   - Variation selectors (U+FE0F) are stripped
3. **Replace** spaces with `-`
4. **Trim** and collapse repeated `-`

## Common Examples

| Heading | Correct anchor |
|---------|---------------|
| `## Why Prism?` | `#why-prism` |
| `## 🚀 Quick Start` | `#-quick-start` |
| `## ⚠️ Limitations` | `#-limitations` |
| `## 🆕 What's New` | `#-whats-new` |
| `## How Prism Compares` | `#how-prism-compares` |
| `## Environment Variables` | `#environment-variables` |

> **Key insight:** Emoji at the start of a heading leave a leading `-` because the space after the emoji is converted to `-`.
> So `## 🔧 Tool Reference` → strips `🔧` → ` tool reference` → `-tool-reference` → `#-tool-reference`.

## Pitfalls to Avoid

- **Never URL-encode emoji** in anchors (e.g. `#%EF%B8%8F-limitations` is WRONG — the emoji is simply stripped, not encoded)
- **Avoid duplicate emoji** across headings that would produce the same anchor (e.g. two `## 🚀 ...` headings → both resolve to `#-...`, second one gets `-1` suffix)
- **HTML `id=` on `<details>`** bypasses slugger entirely — `<details id="my-id">` creates anchor `#my-id` directly

## Quick Validation Script

Run locally to check all anchors before pushing:

```bash
npx github-slugger README.md
```

Or use the online tool: https://hashify.me/

#!/bin/bash
# =============================================================================
# BFCL Review Package Builder
# Combines REVIEW_PROMPT.md + both repomix bundles into a single file
# for pasting into a large-context LLM for external code review.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$SCRIPT_DIR/BFCL_REVIEW_PACKAGE.md"

echo "📦 Building BFCL Review Package..."

# Start with the review prompt
cat "$SCRIPT_DIR/REVIEW_PROMPT.md" > "$OUTPUT"

# Add separator and eval bundle
cat >> "$OUTPUT" << 'SEPARATOR'

---

# APPENDIX A: Evaluation Harness Code (bfcl_repomix_eval.txt)

The following is a repomix bundle of 14 files from the BFCL evaluation harness,
including our handler (`prism_coder.py`), the competitor handler (`salesforce_qwen.py`),
the eval checkers, and the memory/web APIs that the model must interact with.

```
SEPARATOR

cat "$SCRIPT_DIR/bfcl_repomix_eval.txt" >> "$OUTPUT"

cat >> "$OUTPUT" << 'SEPARATOR'
```

---

# APPENDIX B: Training Pipeline Code (bfcl_repomix_training.txt)

The following is a repomix bundle of 5 files from our MLX-native training pipeline,
including QLoRA SFT, GRPO alignment, data generation, and the 35-case test suite.

```
SEPARATOR

cat "$SCRIPT_DIR/bfcl_repomix_training.txt" >> "$OUTPUT"

echo '```' >> "$OUTPUT"

# Report stats
TOTAL_LINES=$(wc -l < "$OUTPUT")
TOTAL_BYTES=$(wc -c < "$OUTPUT")
ESTIMATED_TOKENS=$((TOTAL_BYTES / 4))

echo ""
echo "✅ Review package built: $OUTPUT"
echo "   Lines:  $TOTAL_LINES"
echo "   Bytes:  $TOTAL_BYTES"
echo "   ~Tokens: $ESTIMATED_TOKENS"
echo ""
echo "📋 Usage:"
echo "   1. Copy contents of $OUTPUT"
echo "   2. Paste into Claude/Gemini/GPT-4.1 (128K+ context)"
echo "   3. The review prompt at the top will guide the analysis"

# Hybrid Deployment Strategy: Best-in-Class AAC + Top-Rank BFCL

**Date**: 2026-05-03
**Question**: How to achieve top-rank BFCL coding performance for prism-coder while preserving best-in-class AAC quality for prism-aac, without compromising either?
**Sources**: Local training inventory + SOTA research (`research_brief_bfcl_7b_2026.md`)

---

## ⚠️ Critical Reality Check First

### Our scores ≠ public BFCL scores

What we've been calling "BFCL 80.6%" is OUR custom `bfcl_eval.py` — 64 tests against Prism-specific tools (`session_load_context`, `knowledge_search`, etc.). This is NOT the Berkeley Function Calling Leaderboard.

### Berkeley BFCL V4 reset the field (2026-04-12)

V4 added Web Search + Memory subcategories that need an actual agent harness, not just SFT. Result: **no open 7-9B model exceeds 50% Overall on V4**. The "Hammer 89%, Watt 85%" numbers I quoted earlier were from V2/V3 — obsolete.

**Current top open 7-9B on Berkeley BFCL V4** (snapshot 2026-04-12):

| Model | Overall | Non-Live AST | Live | Multi-Turn | Web Search | Memory | Irrelevance |
|---|---|---|---|---|---|---|---|
| xLAM-2-8b-fc-r | **46.68%** | 84.58% | 67.95% | 70.00% | 6.50% | 13.98% | 63.28% |
| BitAgent-Bounty-8B | 46.23% | 81.60% | 93.12% | 62.38% | 0.00% | 1.51% | 97.48% |
| Qwen3-8B (FC) | 42.57% | 87.58% | 80.53% | 41.75% | 12.00% | 14.62% | 79.07% |
| ToolACE-2-8B (FC) | 42.44% | 87.10% | 77.42% | 38.38% | 8.50% | 18.49% | 90.79% |
| Hammer2.1-7b (FC) | 31.67% | 85.50% | 69.50% | 23.87% | 0.00% | 0.00% | 90.12% |

**Realistic 1-week H100 target for prism-coder:7b-coder: 40–47% Overall**. Breaking 50% would set new open SOTA.

### Healthcare/AAC LLM literature gap

Closest published work: Cai et al. *Nat. Commun.* 2024 (LLMs for ALS eye-gaze typing), Gaines & Vertanen 2025 (character-based AAC), CHI 2025 (AAC + LLM expressivity), BehaveAgent 2025 (BCBA-adjacent). **No published 7-9B model jointly optimizes BFCL + clinical safety.** v17.4's emergency 13/13 + ~80% custom BFCL is unique on the Pareto frontier.

---

## 🎯 Recommended Architecture: Single Base + Multi-LoRA

**Why this beats two-model deployment:**
- Production-verified pattern (vLLM `--enable-lora`, Punica/S-LoRA kernels at MLSys 2024)
- Single base model in memory; LoRA adapters hot-swap per request
- vLLM Semantic Router documents 70% cost savings vs separate models
- Can serve from one Ollama-equivalent endpoint via vLLM
- AAC and BFCL adapters trained independently — no Pareto interference

**Hybrid topology:**

```
                 ┌─ prism-aac (caregiver, askAI, textCorrect, emergency)
                 │
prism-coder:7b ─┤    ↓ system-prompt-based or explicit LoRA selection
(base)          │
                 ├─ aac LoRA (v17.4-style, AAC-perfect)
                 ├─ coder LoRA (xLAM/ToolACE-style, BFCL-optimized)  
                 └─ optional: bcba LoRA, emergency LoRA
```

**For now (without vLLM rollout):** maintain two GGUF tags — `prism-coder:7b-aac` (= v17.4) and `prism-coder:7b-coder` (NEW). Each consumer points to its tag. Migrate to vLLM multi-LoRA later as polish.

---

## 📦 Training Data Inventory (Already Local)

### BFCL data ready

| Source | Lines | Format | License | Quality |
|---|---|---|---|---|
| `data/v17_1_bfcl/` (15 files) | 12,712 | Custom mixed `<\|tool_call\|>` | internal | Mid — need format conversion |
| `data/v18_external_train.jsonl` | 20,691 | glaive/hermes mix | mixed | **POLLUTED — exclude** (Russian noise injection observed in v18) |
| `data/contrastive_sft.jsonl` | 318 | Tool-disambiguation pairs | internal | High quality |
| `data/v16_gen_72b/bfcl_*.jsonl` | ~1,500 | Categorized BFCL variants | internal | High |

### AAC data ready

| Source | Lines | Domain | Quality |
|---|---|---|---|
| `data/v16_gen_72b/caregiver_parse*.jsonl` | 2,007 | BCBA configuration | High |
| `data/v16_gen_72b/emergency_qa*.jsonl` | 1,251 | 911/operator Q&A | High |
| `data/v16_gen_72b/text_correct.jsonl` | 967 | Typo cleanup | High |
| `data/v16_gen_72b/ask_ai_aac.jsonl` | 484 | Child Q&A | High |
| `data/v16_gen_72b/translate.jsonl` | 496 | Multilingual | Medium |
| `data/v16_gen_72b/word_predict_aac.jsonl` | 1,026 | Predictive completion | Medium |
| `data/v16_corrective/emergency_qa_extra.jsonl` | 300 | Emergency hardening | High (in v17.4) |
| `data/train_v17_4.jsonl` (emergency subset) | ~400 | Emergency v17.4-style | High |

### Public datasets to download

| Dataset | Lines | License | Purpose | Download |
|---|---|---|---|---|
| **glaiveai/glaive-function-calling-v2** | 112,960 | Apache 2.0 | Base BFCL volume | HuggingFace |
| **Team-ACE/ToolACE** | 11,300 multi-turn dialogues | Apache 2.0 | Multi-turn BFCL quality | HuggingFace |
| **MadeAgents/xlam-irrelevance-7.5k** | 7,500 | CC-BY-NC-4.0 | Irrelevance gate (research only) | HuggingFace |
| Salesforce/xlam-function-calling-60k | 60,000 | CC-BY-4.0 | Highest-quality SFT (research only) | HuggingFace |
| Salesforce/APIGen-MT-5k | 5,000 multi-turn | CC-BY-NC-4.0 | Best multi-turn agentic (research only) | HuggingFace |

**Permissively licensed mix (commercial-deployable): ~131k examples**
glaive-v2 (112,960) + ToolACE (11,300) + xlam-irrelevance (7,500) = sufficient for SOTA-ish 7B SFT

---

## 🛠 Concrete Recipes

### Model 1: `prism-coder:7b-aac` (KEEP v17.4 as-is)

**Status**: Already deployed as `prism-coder:7b` (= v17.4, hash `c8bdf0c6174a`).
**Scores**: emergency 13/13, caregiver 5/7+20/20 targeted, text_correct 15/15, ask_ai 5/5, translate 7/8, custom-BFCL 79.7%.
**Action**: Re-tag to `prism-coder:7b-aac` so the production tag becomes free for the dual-model layout.
```bash
ollama cp prism-coder:7b prism-coder:7b-aac
```

### Model 2: `prism-coder:7b-coder` (NEW — top-rank BFCL build)

**Goal**: Berkeley BFCL V4 ≥ 45% Overall (xLAM-class), ≥ 85% Non-Live AST, ≥ 90% Irrelevance.

**Recipe** (Hammer-style + light DMPO):

| Step | Detail |
|---|---|
| Base | `Qwen/Qwen2.5-Coder-7B-Instruct` (clean, NO v17 chain) |
| Data | glaive-v2 (112k) + ToolACE (11.3k) + xlam-irrelevance (7.5k) + our v17_1_bfcl (12.7k canonicalized, deduped against glaive) ≈ **140k examples** |
| Excludes | v18_external_train.jsonl (polluted), v18_1_multiturn.jsonl (multi-turn poison), all AAC data |
| Method | DoRA r=256, all 7 projections; full SFT 3 epochs; LR 1.5e-5 cosine; warmup 5%; eff_batch 16 |
| Special | **Function masking** (Hammer) — randomly mask function names 30% of training examples to force binding via descriptions, not name patterns |
| Optional | DMPO pass (Watt-Tool recipe) on synthesized multi-turn preference pairs (1 week if added) |
| Hardware | Modal H100, ~6 hours full SFT (140k × 3 epochs) |
| Eval | Official Berkeley BFCL via `~/gorilla-bfcl/` — full V4 categories |

**Smoke test gates before promotion:**
- Berkeley BFCL Overall ≥ 40% (matches CoALM-8B floor)
- Non-Live AST ≥ 85%
- Irrelevance ≥ 90%
- Custom Prism BFCL ≥ 75% (since prism tools are out-of-distribution for general training data)

### Model 3 (Future): `prism-coder:7b-bcba` LoRA (later sprint)

When ready, train a small BCBA-domain LoRA on top of prism-coder:7b-aac base — clinical reasoning, FBA/BIP drafting, Vineland data parsing. Layer onto AAC base for Synalux clinical workflows.

---

## 🔧 Consumer Code Changes (Minimal)

### prism-aac (NO CHANGES required)

`localModel.ts` already references `prism-coder:7b`:
```typescript
export const LOCAL_MODEL = 'prism-coder:7b';
```

Decision: keep the production tag pointing to the AAC model (the AAC use case is far more sensitive to regression than coding). prism-aac stays unchanged.

### prism-mcp (ONE constant change)

In whatever file references the local model in prism-mcp:
```typescript
// Change:
const LOCAL_MODEL = 'prism-coder:7b';
// To:
const LOCAL_MODEL = 'prism-coder:7b-coder';
```

### Prism Coder IDE / coding consumers

Same one-line change as above.

### Synalux portal

If portal routes to prism-coder for coding tasks: same one-line change.

---

## 📅 Timeline (1-week sprint)

### Day 1 (today, May 3): Inventory + data prep

- [x] Inventory current data (done — see Section 📦)
- [x] Research SOTA recipes (done)
- [ ] Download glaive-v2, ToolACE, xlam-irrelevance from HuggingFace (~5 GB)
- [ ] Write `prepare_v18coder_data.py` — convert all to canonical Qwen2 format, dedupe, hold-out 1k for validation
- [ ] Wait for v18-clean (currently running) — reuse its insights

### Day 2: First training run

- [ ] Submit `modal_v18coder_sft.py` — 140k examples × 3 epochs on H100, ETA ~6 hours
- [ ] Save per-epoch checkpoints (epoch_1, epoch_2, epoch_3)

### Day 3: Berkeley BFCL eval

- [ ] Set up `bfcl-eval` package (`pip install bfcl-eval==2025.12.17`)
- [ ] Run official `bfcl generate` + `bfcl evaluate` on each checkpoint
- [ ] Pick best checkpoint by Overall score with Irrelevance ≥ 85%

### Day 4: Function masking enhancement (if BFCL < 40%)

- [ ] Modify training data: add 30% rows with masked function names (random `func_xxx` placeholders) — Hammer's recipe
- [ ] Re-train with masked data
- [ ] Re-eval

### Day 5: Optional DMPO pass (if BFCL < 45%)

- [ ] Synthesize multi-turn preference pairs from APIGen-MT-5k structure
- [ ] DMPO training (arXiv:2406.14868) on top of best SFT checkpoint
- [ ] Re-eval

### Day 6: Production deployment

- [ ] PEFT merge → GGUF Q4_K_M for best checkpoint
- [ ] Tag as `prism-coder:7b-coder`
- [ ] Update prism-mcp + Prism Coder IDE constants
- [ ] Deploy snapshot rollback safety
- [ ] Smoke-test all consumers

### Day 7: Verification + report

- [ ] Run full prism-aac consumer test suite (`/tmp/test_production_v17_2.sh`) on aac tag
- [ ] Run BFCL official + custom on coder tag
- [ ] Document final model cards
- [ ] Update Prism memory ledger

---

## ⚖️ Risk Analysis

### Risks for prism-coder:7b-coder

| Risk | Likelihood | Mitigation |
|---|---|---|
| Berkeley BFCL still < 40% after SFT | Medium | DMPO pass adds another shot; if still failing, add APIGen-MT-5k (NC license) |
| Format pollution from glaive-v2 (lower quality) | Medium | Canonicalize to Qwen2 `<tool_call>` format; mix ToolACE for higher-quality signal |
| Web Search / Memory categories at 0% | High | Expected — these need agent harness training, not just SFT. Accept as known weakness; ship v1 without them |
| Live AST low (< 70%) | Low | glaive-v2 emphasizes single-turn; should land 80%+ |
| Conflicts with existing v17.x snapshots | None | New tag, doesn't touch production |

### Risks for prism-coder:7b-aac (no change)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Re-tag confuses consumers | None | All consumers reference `prism-coder:7b`; the new `prism-coder:7b-aac` tag is just an alias |

### Risks for the hybrid deployment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Two models = 2× memory on local Ollama | Low | Each is 4.4 GB; total 8.8 GB. M-series Macs (≥ 16 GB) handle this fine |
| Routing logic incorrect in consumers | Medium | Keep change to ONE constant per consumer file; verify with smoke test |
| User accidentally uses wrong tag | Low | Tag names are explicit (`-aac` vs `-coder`); document in CLAUDE.md |

---

## 💰 Resource Budget

| Item | Cost |
|---|---|
| Modal H100 for SFT (140k × 3 epochs ≈ 6 hr) | ~$25 |
| Modal H100 for DMPO (5k pairs × 2 epochs ≈ 2 hr) | ~$8 |
| BFCL official eval (cloud-vllm or local 4.7 GB GGUF) | ~$2 (Modal) or free (local) |
| HuggingFace dataset downloads (~5 GB) | free |
| Local disk for additional GGUF (4.4 GB) | free |
| **Total** | **~$35** for the whole campaign |

---

## 📚 Reference Implementation Notes

### Data download script (Day 1)
```bash
huggingface-cli download glaiveai/glaive-function-calling-v2 --repo-type dataset --local-dir ~/prism/training/data/external/glaive-v2
huggingface-cli download Team-ACE/ToolACE --repo-type dataset --local-dir ~/prism/training/data/external/toolace
huggingface-cli download MadeAgents/xlam-irrelevance-7.5k --repo-type dataset --local-dir ~/prism/training/data/external/xlam-irrelevance
```

### Function masking implementation (Day 4)
```python
def mask_functions(text: str, mask_prob: float = 0.30) -> str:
    """Hammer-style: replace function names with placeholder, force model to bind by description."""
    if random.random() > mask_prob:
        return text
    # Find function names in <tools> block, replace with func_001, func_002, ...
    # Same in <tool_call> calls
    ...
```

### DMPO references (Day 5)
- Watt-Tool DMPO paper: arXiv:2406.14868
- Implementation reference: `trl.DPOTrainer` with multi-turn preference format
- Synthesis: take APIGen-MT-5k trajectories, randomly perturb tool args/names → rejected; original → chosen

### Berkeley BFCL eval (Day 3)
```bash
pip install bfcl-eval==2025.12.17
cd ~/gorilla-bfcl/berkeley-function-call-leaderboard
bfcl generate --model prism-coder:7b-coder --backend ollama
bfcl evaluate --model prism-coder:7b-coder
```

---

## 🎯 Honest Expected Outcomes

| Metric | Pessimistic | Realistic | Optimistic | Reach |
|---|---|---|---|---|
| Berkeley BFCL V4 Overall | 35% | **42%** | 47% | 50%+ (would set 7-9B SOTA) |
| Berkeley BFCL Non-Live AST | 78% | **86%** | 90% | 92% |
| Berkeley BFCL Live | 65% | **75%** | 82% | 88% |
| Berkeley BFCL Multi-Turn | 30% | **45%** | 60% | 70% |
| Berkeley BFCL Irrelevance | 80% | **90%** | 95% | 97% |
| Berkeley BFCL Web Search | 0% | 5% | 10% | (needs agent harness) |
| Berkeley BFCL Memory | 0% | 5% | 12% | (needs agent harness) |
| Custom Prism BFCL | 70% | **78%** | 85% | (out-of-distribution for general training) |
| AAC (prism-aac, unchanged) | 100% | 100% | 100% | (untouched) |

**Realistic outcome**: prism-coder:7b-coder lands in **xLAM-2-8b-fc-r territory (42–47% Overall)** — competitive with current open SOTA in the 7-9B class — while prism-coder:7b-aac retains its AAC perfection. Both shipping by end of week.

---

## 🚦 Decision Points

**Before kickoff, confirm:**

1. **License acceptance**: OK to use NC datasets (xlam-60k, APIGen-MT-5k) for the prism-coder:7b-coder build? They unlock the strongest signal but restrict commercial deployment for that model.

2. **Tag scheme**: Approve splitting production into `prism-coder:7b-aac` (= v17.4) + `prism-coder:7b-coder` (NEW)? This requires the one-line consumer changes documented above.

3. **vLLM migration timing**: Ollama dual-tag is the simplest start. vLLM multi-LoRA migration can be Week 2 polish. Approve sequential rollout?

4. **Berkeley BFCL submission**: After hitting target scores, submit `prism-coder:7b-coder` to the public Berkeley BFCL leaderboard? (Free, gets us a public SOTA-competitive score.)

---

## 📎 Source Documents

- Local inventory: this document, Section 📦
- SOTA research: `/Users/admin/prism/training/research_brief_bfcl_7b_2026.md` (1900 words, with citations)
- Berkeley BFCL V4 CSV cache: `/tmp/bfcl_overall.csv`
- HuggingFace model cards: `/tmp/{hammer,xlam,toolace}_*.html`

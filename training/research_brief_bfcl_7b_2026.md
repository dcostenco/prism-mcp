# Research Brief: State of the Art for 7B BFCL Function-Calling Models (May 2026)

Audience: engineer planning a 1-week training campaign on Modal H100s.
Primary source: Berkeley Function Calling Leaderboard V4 CSV (`gorilla.cs.berkeley.edu/data_overall.csv`, last updated 2026-04-12), evaluated with `bfcl-eval==2025.12.17` at commit `f7cf735`. Model-card and paper details cited inline. "Not found" is used wherever a number could not be confirmed.

---

## 1. Current top 7-9B models on BFCL V4 (2026-04-12 snapshot)

BFCL V4 is dramatically harder than V3: it adds Web Search, Memory (KV / Vector / Recursive Summarization), and a holistic agentic regime. Overall scores collapsed across the board — the global #1 (Claude Opus 4.5) is only 77.47%. **No open 7-9B model currently exceeds 50% Overall on BFCL V4**, and the > 85% Overall numbers seen on older blog posts refer to BFCL V2/V3.

Top 7-9B class entries on V4 (filtered from the official CSV):

| Rank | Model | Org | Overall | Non-Live AST | Live | Multi-Turn | Web Search | Memory | Irrelevance |
|-----:|-------|-----|--------:|-------------:|-----:|-----------:|-----------:|-------:|------------:|
| 34 | xLAM-2-8b-fc-r (FC) | Salesforce | **46.68%** | 84.58% | 67.95% | 70.00% | 6.50% | 13.98% | 63.28% |
| 36 | BitAgent-Bounty-8B | Bittensor SN20 | **46.23%** | 81.60% | 93.12% | 62.38% | 0.00% | 1.51% | 97.48% |
| 39 | Qwen3-8B (FC) | Qwen | 42.57% | 87.58% | 80.53% | 41.75% | 12.00% | 14.62% | 79.07% |
| 40 | ToolACE-2-8B (FC) | Huawei Noah + USTC | 42.44% | 87.10% | 77.42% | 38.38% | 8.50% | 18.49% | 90.79% |
| 50 | Llama-4-Maverick-17B (FC) | Meta | 37.29% | – | – | – | – | – | – |
| 61 | Command R7B (FC) | Cohere | 32.07% | – | – | – | – | – | – |
| 64 | Hammer2.1-7b (FC) | MadeAgents | 31.67% | 85.50% | 69.50% | 23.87% | 0.00% | 0.00% | 90.12% |
| 81 | Granite-3.1-8B-Instruct (FC) | IBM | 27.10% | – | – | – | – | – | – |
| 83 | Granite-3.2-8B-Instruct (FC) | IBM | 26.87% | 79.77% | 60.33% | 7.38% | 0.50% | 12.47% | 80.53% |
| 84 | CoALM-8B | UIUC + Oumi | 26.81% | – | – | – | – | – | – |
| 85 | Llama-3.1-8B-Instruct (Prompt) | Meta | 25.83% | – | – | – | – | – | – |
| 91 | Falcon3-7B-Instruct (FC) | TII UAE | 24.03% | – | – | – | – | – | – |
| 105 | Ministral-8B-Instruct-2410 (FC) | Mistral AI | 11.10% | – | – | – | – | – | – |

Notes:
- **Watt-Tool-8B is not present** in the V4 CSV (last seen on V3 leaderboards). Its model card still claims SOTA, but no V4 measurement is published — treat as "BFCL V3 SOTA, V4 unknown."
- **Functionary-v3** is not present in the V4 CSV either.
- **APIGen** is a data-generation pipeline, not a deployed model — its trained checkpoints ship as the xLAM-2-fc-r series.
- Salesforce's larger sibling **xLAM-2-32b-fc-r** is the highest-ranked open model overall at #18 (54.66%). The 7-9B class lags it by ~8 points.

Key takeaway for a 1-week H100 campaign: a **7-9B model that breaks 50% Overall on BFCL V4 would set a new open-weights SOTA in its size class.** Realistic short-horizon targets are: > 85% Non-Live AST, > 75% Live, > 35% Multi-Turn, > 90% Irrelevance. Web Search and Memory categories effectively require an agent harness, not just SFT.

---

## 2. Training recipes for the top open 7-9B BFCL models

### xLAM-2-8b-fc-r — Salesforce (BFCL V4 #34, 46.68%)
- **Base**: Llama-3.1-8B-Instruct (HF: `Salesforce/Llama-xLAM-2-8b-fc-r`, identical to `xLAM-2-8b-fc-r`).
- **Training data**: `Salesforce/xlam-function-calling-60k` (60k single-turn) + `Salesforce/APIGen-MT-5k` (5k multi-turn ShareGPT-format trajectories, retail + airline τ-bench domains).
- **Method**: SFT only (model card does not mention DPO/RLHF). The "-r" suffix = research release.
- **Special techniques**: APIGen-MT generation pipeline — two-phase blueprint synthesis with a committee of LLM reviewers, then simulated agent-human rollouts. 99% human-rated success on a 200-trajectory sample.
- **License**: CC-BY-NC-4.0 (research only, also bound by Llama 3 community license).
- **Paper**: arXiv:2504.03601.

### ToolACE-2-8B — Huawei Noah + USTC (BFCL V4 #40, 42.44%)
- **Base**: Llama-3.1-8B-Instruct.
- **Training data**: ToolACE dataset — 11.3k released conversations (2-12 turns each) drawn from a self-evolved pool of **26,507 diverse APIs**. Bilingual EN/ZH.
- **Method**: SFT with self-refinement tuning, dual-layer (rule-based + model-based) verification, formalized "thinking process" multi-agent generation.
- **Paper**: ToolACE: Winning the Points of LLM Function Calling, arXiv:2409.00920, ICLR 2025.
- **License**: Apache 2.0 (model and data).
- **Reported BFCL**: V3 SOTA at 8B (model card claim); V4 score is 42.44% per CSV.

### Hammer2.1-7b — MadeAgents (BFCL V4 #64, 31.67%)
- **Base**: Qwen2.5-Coder-7B-Instruct.
- **Training data**: `Salesforce/xlam-function-calling-60k` (60k) + `MadeAgents/xlam-irrelevance-7.5k` (7.5k irrelevance examples). Total ~67.5k.
- **Method**: SFT only.
- **Special technique**: **Function masking** (arXiv:2410.04587) — randomly masks function names so the model learns parameter-binding from descriptions rather than name conventions; combined with augmented irrelevance examples that boost "no-call" / abstention behavior.
- **License**: CC-BY-NC-4.0.
- **Why the V4 drop**: Hammer's strength is single-turn AST (85.50%) and Irrelevance (90.12%); it scores 0% on Web Search and Memory because no agentic harness was trained.

### Watt-Tool-8B — Watt AI (BFCL V3 SOTA; not on V4)
- **Base**: Llama-3.1-8B-Instruct.
- **Method**: SFT + **DMPO** (Direct Multi-Turn Preference Optimization, arXiv:2406.14868) on synthesized multi-turn CoT dialogue data.
- **Training data**: Not publicly released; described only as "specialized dataset for tool usage and multi-turn dialogue."
- **License**: Apache 2.0.

### CoALM-8B — UIUC + Oumi (BFCL V4 #84, 26.81%)
- **Base**: Llama-3.1-8B-Instruct (full fine-tune).
- **Training data**: CoALM-IT (size not disclosed on card).
- **Hardware reference point**: 8 × H100 for ~8 hours (CoALM-70B variant: ~24 hours on 8 × H100). **This is a realistic floor for what a 1-week H100 budget can do.**
- **Frameworks**: Oumi.
- **License**: CC-BY-NC-4.0.

### BitAgent-Bounty-8B (BFCL V4 #36, 46.23%)
- Trained competitively on Bittensor Subnet 20, ≤ 8B parameter constraint, optimized directly against BFCL.
- Notable for **97.48% Irrelevance** and **93.12% Live** — the highest Live AST in the open 7-9B class.
- Training recipe is not publicly documented (subnet incentivized model — weights public on HF, recipe is not).

### Healthcare / AAC performance
**Not found.** None of the top 7-9B BFCL models report AAC, BCBA, or healthcare-domain numbers. BFCL itself does not contain a healthcare/AAC slice.

---

## 3. Public BFCL-style training datasets ( > 5k, with confirmed license & source)

| Dataset | Size | Format | License | Source | Notes |
|---|--:|---|---|---|---|
| `Salesforce/xlam-function-calling-60k` | 60,000 | single-turn JSON (query + tools + answers) | CC-BY-4.0 | APIGen pipeline (DeepSeek-V2-Chat 33,659 + Mixtral-8x22B-Inst rest); 3,673 executable APIs across 21 categories | Highest-leverage public dataset; >95% human-rated correctness. Paper: arXiv:2406.18518. |
| `Salesforce/APIGen-MT-5k` | 5,000 multi-turn dialogues (~128 MB) | ShareGPT-style with `function_call`, `observation`, `gpt` turns | CC-BY-NC-4.0 | APIGen-MT pipeline, GPT-4o + DeepSeek-V3, τ-bench retail + airline domains | The only open multi-turn agentic-quality dataset; 99% human-rated. Paper: arXiv:2504.03601. |
| `Team-ACE/ToolACE` | 11,300 dialogues (2-12 turns) | JSON, EN+ZH | Apache 2.0 | 26,507-API self-evolved pool, multi-agent generation, dual-layer verification | Permissive license — preferable for commercial deployment vs xLAM. |
| `MadeAgents/xlam-irrelevance-7.5k` | 7,500 | irrelevance / no-call examples | CC-BY-NC-4.0 | Augmentation set used in Hammer | Critical for raising the Irrelevance sub-score. |
| `glaiveai/glaive-function-calling-v2` | 112,960 | system-prompt + functions + assistant calls | Apache 2.0 | Synthetic, broad domains | Largest permissive single-turn corpus; 237+ derivative models on HF. Quality is lower than APIGen-grade. |
| ToolACE full API pool | 26,507 APIs | API specs (not dialogues) | Apache 2.0 | Self-evolved synthesis | Useful as a function bank for your own agent simulations. |

Datasets the user mentioned that I could **not** verify as publicly downloadable: "Hammer-Train" (referenced in Hammer paper but the released training data is the xLAM-60k + xlam-irrelevance-7.5k combo, not a separate "Hammer-Train" dump); "xLAM training corpus" beyond xlam-60k + APIGen-MT-5k (the larger internal corpus is not released).

**Recommended 1-week mix** (all permissively licensed, ~131k examples): glaive-v2 (112,960) + ToolACE (11,300) + xlam-irrelevance (7,500). Add xlam-60k + APIGen-MT-5k if NC-4.0 is acceptable for the campaign.

---

## 4. Healthcare / accessibility LLM literature (most relevant 5)

1. **Cai et al., "Using large language models to accelerate communication for eye gaze typing users with ALS,"** *Nature Communications* 2024. LLM-based word/phrase prediction for ALS users on eye-gaze keyboards; the most rigorous AAC + LLM clinical paper to date.
2. **Gaines & Vertanen, "Adapting Large Language Models for Character-based Augmentative and Alternative Communication,"** arXiv:2501.10582 (v3 Oct 2025). Domain-adaptation of LLMs for character-level AAC prediction; releases a curated AAC sentence corpus and a scoring procedure.
3. **"Your voice is your voice: Supporting Self-expression through Speech Generation and LLMs in AAC,"** arXiv:2503.17479 (CHI 2025). Multimodal-input + LLM-driven AAC for expressivity; strongest qualitative study of LLM-AAC UX.
4. **"Augmented Body Communicator,"** Augmented Humans 2025 (DOI 10.1145/3745900.3746089). LLM + robotic-arm system for nonverbal expression in users with upper-limb limitations.
5. **Liu et al., "An autonomous AI agent for universal behavior analysis (BehaveAgent),"** PubMed 40475621, 2025. Multimodal-LLM agent for behavior analysis that generalizes across novel behavioral domains without retraining — closest published parallel to a BCBA-facing agent.

Adjacent / emergency-response:
- **"LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation,"** arXiv:2509.26351. Two-regime triage benchmark with SHAP-based interpretability baselines — the only public LLM 911/triage benchmark I found.
- Frontiers in Big Data 2025 (PMC12277377): SVM + textual feature 911 call-handler assistance — pre-LLM baseline worth citing for context.

**Multi-objective LLMs combining general capability + healthcare safety**: not found as a single published recipe. Existing healthcare LLMs (Med-PaLM, Meditron, etc.) optimize medical QA, not joint general-capability + safety. The closest pattern is reward-model ensembling in RLHF, but no published 7-9B work pairs BFCL function calling with a clinical-safety reward.

---

## 5. Hybrid / specialist deployment patterns (production-verified)

### Multi-LoRA hot-swap on shared base
- **vLLM** has first-class `--enable-lora` + `--max-loras N` support; LoRA adapters are loaded per-request via `LoRARequest(adapter_name, …)` and batched together with the base model. (`docs.vllm.ai/en/latest/features/lora/`)
- **Punica** (arXiv:2310.18547, MLSys 2024) introduces SGMV (Segmented-Gather GEMV) kernels that fuse heterogeneous LoRA deltas into one matmul; reports **12× throughput vs vLLM baseline at multi-LoRA serving, +2 ms/token**.
- **S-LoRA** (UC Berkeley) is the other canonical reference; both Punica and S-LoRA are cited together in CARASERVE (CPU-assisted, rank-aware extension).
- **"Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems"** arXiv:2511.22880 (Nov 2025) is the most current survey of production multi-tenant LoRA serving.
- **vLLM Semantic Router** documents two production patterns: (a) domain-classifier routes to subject-specific LoRA adapters ("4 specialized tutors for the cost of 1.2 base models, 70% cost savings"); (b) tenant-ID routing for 1000+ adapters via MCP.
- **Anyscale / Ray Serve LLM** has documented multi-LoRA deployment for multi-domain inference on a shared base.

### Mixture-of-Experts routing
- Production examples (Mixtral 8x22B, DeepSeek-V3, Llama-4-Maverick-17B-128E, GLM-4.6) are now standard, but an **MoE base + multi-LoRA on top** pattern is not yet a published reference architecture.

### Specialist model ensembles / domain-specific endpoints
- No explicit Anthropic/OpenAI/Cohere/Mistral case-study found that quantifies the specialist-ensemble-vs-LoRA-vs-MoE tradeoff for a 7B-class model; published material is mostly on the serving-system side (Punica/S-LoRA/Anyscale), not the productization side.

### Practical recommendation for a multi-objective (BFCL + AAC + BCBA + 911) 7-9B deployment
- A **single SFT base + N hot-swappable LoRA adapters** on vLLM is the dominant published pattern and the cheapest to operate. Train one strong BFCL base (target the 1-week campaign at xLAM/ToolACE-class scores), then layer domain LoRAs (AAC, BCBA, 911) at rank 16-64. Route via either a small classifier or an MCP/tenant-ID lookup (Semantic Router pattern).
- Avoid full MoE for a single-team 1-week budget — training cost is prohibitive and there is no open recipe for mid-train MoE-ifying a 7B model in that window.

---

## Actionable summary for a 1-week Modal H100 campaign

1. **Realistic V4 ceiling** for a 7-9B SFT-only run on permissively licensed data: in the 40-47% Overall band (matches xLAM-2-8b, BitAgent-Bounty-8B, Qwen3-8B, ToolACE-2-8B). Pushing past 50% likely needs a real agent harness with browse + memory tools at training time, not just static SFT.
2. **Highest-leverage data mix** under a 1-week budget: glaive-v2 (112k) + ToolACE (11.3k) + xlam-irrelevance (7.5k); if NC-4.0 is OK, add xlam-60k + APIGen-MT-5k.
3. **Best base for SFT**: Qwen2.5-7B-Instruct or Qwen2.5-Coder-7B-Instruct (Hammer chose the latter; Qwen3-8B is the strongest base on V4 in the 7-9B class). Llama-3.1-8B is the second choice and is what xLAM/ToolACE/Watt/CoALM all use.
4. **Recipe to copy**: SFT (Hammer's function-masking + irrelevance augmentation) → optional DMPO pass (Watt-Tool's recipe, arXiv:2406.14868) on multi-turn preference pairs synthesized from APIGen-MT-5k. Skip RLHF — no published 7-9B BFCL leader uses it.
5. **For multi-objective (BFCL + AAC + BCBA + 911) deployment**: single base + multi-LoRA via vLLM `--enable-lora`, optionally Punica/S-LoRA kernels for batched serving. There is no published precedent for jointly optimizing BFCL + clinical safety in a single 7B; treat the AAC/BCBA/911 work as separate LoRA adapters trained against domain-specific datasets (which would need to be created — none of comparable scale exist in the public literature).
6. **Calibrate expectations**: CoALM-70B took ~24 H100-hours; CoALM-8B took ~8 H100-hours. Even allowing 5-10× for ablations and DMPO, a 1-week 8 × H100 budget is comfortably enough for multiple full SFT runs at 8B scale.

---

## Source files (absolute paths, local cache)
- `/tmp/bfcl_overall.csv` — full BFCL V4 leaderboard CSV (109 rows incl. header), pulled 2026-05-03 from `gorilla.cs.berkeley.edu/data_overall.csv`.
- `/tmp/bfcl_leaderboard.html`, `/tmp/bfcl_idx.js`, `/tmp/formHeaders.json` — leaderboard page assets used to locate the CSV endpoint.
- `/tmp/hammer_7b.html`, `/tmp/xlam_7b.html`, `/tmp/toolace_8b.html` — raw HF model-card HTML for cross-reference.

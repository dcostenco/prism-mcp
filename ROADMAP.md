# Prism Project Roadmap

## Current Phase: Agent Infrastructure Resilience

- [x] **v11.0.1** - GRPO Alignment (Phase 1)
    - [x] Zero-Search $O(1)$ Retrieval Integration
    - [x] Group Relative Policy Optimization (500 iters)
    - [x] VRAM-optimized DPO for Apple Silicon
    - [x] Initial structural alignment for `<think>` reasoning
- [x] **v11.5.0** - Structural GRPO Alignment
    - [x] Achieve 100% Tool-Call Accuracy (Cross-validated on Synalux)
    - [x] 100.0% JSON Validity via `<|synalux_think|>` → `<|tool_call|>` enforcement
    - [x] Centralized thinking tag stripping in `localLlm.ts`
    - [x] VRAM-optimized DPO training pipeline
- [x] **v11.6.0** - Agent Infrastructure Resilience
    - [x] Serialized execution queue via Python `fcntl.flock` (macOS-native)
    - [x] Memory guardian daemon for proactive RAM pressure management
    - [x] Queue watchdog for deadlock auto-drain
    - [x] Unified agent status dashboard (`agent_status.sh`)
    - [x] 115/115 tests across 5 suites (unit, concurrent, shell, mock, stress)
- [ ] **v12.0.0** - Distal Memory
    - [ ] Semantic clustering of long-term history
    - [ ] Active-Prism background maintenance
- [ ] **v13.0.0** - Team Handoff
    - [ ] Encrypted peer-to-peer session syncing
    - [ ] Multi-agent task routing with verifiable memory
- [ ] **v14.0.0** - VM & Device Integration
    - [ ] Hypervisor abstraction (Apple Virtualization, VMware, Parallels, QEMU, Hyper-V)
    - [ ] 9 built-in device templates (Linux, Windows, iOS, iPadOS, watchOS, Wear OS, Android)
    - [ ] VMware/Parallels import (vmx, pvm, ova, qcow2, vhdx)
    - [ ] Custom device parameters (user-defined specs, port forwarding, env vars)
    - [ ] GitHub Actions deployment pipeline with test suite integration
    - [ ] Thin-client proxy to Synalux Cloud
    - [ ] Tier-gated VM quotas (apps inherit Synalux subscription tiers)

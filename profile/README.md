# speech-refinement

**Diffusion-bridge based offline speech refinement.**

A research effort that takes speech degraded by a prior DNN stage — musical noise, distortion, band-limiting, codec artifacts — and refines it on perceptual metrics (PESQ, STOI, SI-SDR, MCD) using diffusion bridge models in the mel-spectrogram space, with a high-quality vocoder (BigVGAN) for waveform reconstruction.

We deliberately split the research into **three orthogonal axes** so each contribution can be ablated cleanly. A future `unified` model combines the best of all three.

## Three orthogonal research axes

| Axis | Improves | Repo | Status |
|---|---|---|---|
| **A** | **Conditioning** — text / region mask / NL quality descriptor for controllable refinement | [`mc-ddbm`](https://github.com/speech-refinement/mc-ddbm) | active |
| **B** | **Generative process** — Stochastic Optimal Control (SOC) bridge with UniDB-style γ; function-space lift via DBFS | [`soc-bridge`](https://github.com/speech-refinement/soc-bridge) | active |
| **C** | **Sampling speedup** — Consistency Model / Consistency Trajectory Model distillation | `fast-sampling` | future |
| **∪** | **Best-of unification** — combines A + B + C into one model | `unified` | future |

## Architecture

```
Degraded speech (waveform)
        │
        ▼
   ┌────────────────────────────────────┐
   │  Mel transform (shared core)       │
   │  - 22050 Hz, 80 mels, hop 256      │
   │  - normalized log-mel ∈ [-1, 1]    │
   └────────────────────────────────────┘
        │
        ▼  (B, 1, 80, T) mel-spectrogram pair: clean ↔ degraded
        │
   ┌────────────────────────────────────┐
   │  Diffusion bridge model            │
   │  ┌──────────────────────────────┐  │
   │  │ Axis A: conditioning         │  │
   │  │   - mask concat              │  │
   │  │   - transcript cross-attn    │  │
   │  │   - NL descriptor FiLM       │  │
   │  ├──────────────────────────────┤  │
   │  │ Axis B: SOC + γ              │  │
   │  │   - level (b): loss + drift  │  │
   │  │   - function-space lift      │  │
   │  ├──────────────────────────────┤  │
   │  │ Axis C: distillation         │  │
   │  │   - CM / CTM teacher → student│ │
   │  └──────────────────────────────┘  │
   │  Sampler: Heun (40 step) /         │
   │           Euler-Maruyama /         │
   │           CM (1 step)              │
   └────────────────────────────────────┘
        │
        ▼  refined mel-spectrogram
        │
   ┌────────────────────────────────────┐
   │  BigVGAN vocoder                   │
   │  (Vocos / Griffin-Lim fallback)    │
   └────────────────────────────────────┘
        │
        ▼
   Refined speech (waveform)
        │
        ▼
   ┌────────────────────────────────────┐
   │  Evaluation: PESQ, STOI, SI-SDR,   │
   │  MCD, NISQA, DNSMOS                │
   └────────────────────────────────────┘
```

## Repository map

| Repo | Purpose | Branching |
|---|---|---|
| [**mc-ddbm**](https://github.com/speech-refinement/mc-ddbm) | Axis A. Multi-conditional DDBM with mask / transcript / NL descriptor conditioning. Includes mel I/O, vocoder, dataset, degradation pipeline, evaluation metrics. | GitFlow |
| [**soc-bridge**](https://github.com/speech-refinement/soc-bridge) | Axis B. SOC-based function-space diffusion bridge with γ-tuning (UniDB) and DBFS-style operator transformer. | trunk-based |
| **fast-sampling** *(future)* | Axis C. CM / CTM distillation from mc-ddbm + soc-bridge teachers. | trunk-based |
| **unified** *(future)* | Combines A + B + C into a single model. | trunk-based |
| **core** *(future, Stage 1)* | Shared library extracted from mc-ddbm: mel I/O, vocoder, dataset, degradation, evaluation, base UNet, training scaffold. | trunk-based |
| **meta** *(future)* | Cross-repo umbrella issues, strategic documents, paper drafts, comparison reports. | trunk-based |
| **gui** *(future)* | Gradio + node UI extracted from mc-ddbm. | trunk-based |
| [**.github**](https://github.com/speech-refinement/.github) | This profile README + org-wide issue templates. | trunk-based |

Per-repo branching strategies are intentional: long-running experiments + frequent rollbacks make GitFlow safer for `mc-ddbm`, while greenfield code in `soc-bridge` and future repos benefits from trunk-based simplicity. See each repo's `CLAUDE.md` and `docs/COMPATIBILITY.md` for cross-repo coordination protocols.

## Theoretical foundation

- **DDBM** — Zhou et al. *"Denoising Diffusion Bridge Models."* ICLR 2024. The base bridge formulation. [alexzhou907/DDBM](https://github.com/alexzhou907/DDBM)
- **SOC function-space bridge** — Park, Choi, Lim, Lee. *"Stochastic Optimal Control for Diffusion Bridges in Function Spaces."* NeurIPS 2024. [arXiv:2405.20630](https://arxiv.org/abs/2405.20630). Official impl: [bw-park/DBFS](https://github.com/bw-park/DBFS) (MIT). The function-space lift target.
- **UniDB** — Zhu et al. *"UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control."* ICML 2025 Spotlight. [arXiv:2502.05749](https://arxiv.org/abs/2502.05749). [UniDB-SOC/UniDB](https://github.com/UniDB-SOC/UniDB). The γ parameter source.
- **UniDB++** — Zhu et al. [arXiv:2505.21528](https://arxiv.org/abs/2505.21528). Fast sampling extension.
- **Consistency Models** — Song et al. *"Consistency Models."* ICML 2023. The Axis C foundation.
- **CTM** — Kim et al. *"Consistency Trajectory Models."* ICLR 2024.
- **BigVGAN** — Lee et al. *"BigVGAN: A Universal Neural Vocoder with Large-Scale Training."* ICLR 2023. The default vocoder. [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)

## Demo

*Coming soon* — Phase 1 of `soc-bridge` produces a γ sweep across {0.1, 1, 10, 100, 1000, ∞}. Once results land, this section will host audio comparison samples (clean / degraded / γ ablations / Axis A conditioning ablations). Future plan: GitHub Pages site with embedded HTML5 audio elements.

## Roadmap

### `soc-bridge` phases (Axis B)
1. **Phase 1** — finite-dim γ-tuned baseline on mel-spectrograms. γ sweep across 6 values, full UniDB level (b) (loss + drift).
2. **Phase 2** — function-space lift via DBFS-style OperatorTransformer + DCT-basis SDE.
3. **Phase 3** — perceptually-weighted cost functionals; per-degradation γ optimization; degradation physics in forward drift.

### `mc-ddbm` phases (Axis A)
- **Phase A** — transcript conditioning (XPhoneBERT cross-attention)
- **Phase B** — region mask conditioning (SDEdit-style variable noise)
- **Phase D** — NL quality descriptor conditioning (sentence-transformers + FiLM)
- **Integration** — multi-conditional CFG combining A + B + D

### Org-wide stages
- **Stage 0** — bootstrap. mc-ddbm migrated from GitLab → GitHub. soc-bridge initialized. Sibling-clone editable-install dependency.
- **Stage 1** — `core` extraction from mc-ddbm. Both mc-ddbm and soc-bridge depend on `core`. Cross-repo umbrella issue infrastructure (`meta` repo) live.
- **Stage 2** — Axis C (`fast-sampling`) starts. CM/CTM distillation from soc-bridge / mc-ddbm checkpoints.
- **Stage 3** — `unified` combines A + B + C. Paper drafts in `meta`. Demo site live.
- **Stage 4** — OSS extraction. `core` published as a public package; research repos remain private until publication.

## How we work with Claude Code

This org uses [Claude Code](https://claude.com/claude-code) as a coding collaborator under a deterministic operation framework adapted from [poppo-ai](https://github.com/poppo-ai). Each repo has its own:

- `CLAUDE.md` — per-repo Claude instructions
- `.claude/settings.json` — hook event configuration (SessionStart, UserPromptSubmit, PreToolUse, Stop, ...)
- `.claude/hooks/*.sh` — enforcement scripts (deny edits on trunk branch, block deferral language, warn on cross-repo pin drift, ...)
- `.claude/agents/{macro-overseer,doc-hygiene-overseer}.md` — observer agents that detect strategic drift and tactical inconsistencies; report-only, no implementation
- `docs/NEXT_SESSION.md` — single source of truth for "what to do now" (parsed by hooks each turn)

Cross-repo work goes through umbrella issues in the `meta` repo with same-named feature branches in each affected repo. Each repo's Claude session edits only its own files (`guard-mc-ddbm-readonly.sh` and similar hooks enforce the boundary technically).

The most important meta-rule, inherited from poppo: **memory is not a fix.** If a rule should be enforced, write a hook, not a memory note.

## Tech stack

- **Python** 3.10+
- **PyTorch** 2.x with CUDA 12.x
- **BigVGAN** as the default mel→wav vocoder, with Vocos and Griffin-Lim fallbacks
- **TensorBoard** for training visualization
- **Docker** as the primary runtime (host installs are debugging-only)
- **GitHub Actions** for CI (lint via `ruff`, import smoke tests, mc-ddbm pin validation)

## Status

| Item | Status |
|---|---|
| GitHub Org created | ✅ 2026-04-08 |
| `.github` repo (this README placed) | 🚧 in progress |
| `mc-ddbm` migrated from GitLab | ⏳ pending (waits for Phase D v2 completion) |
| `soc-bridge` initial push | ⏳ pending |
| Stage 0 cutover complete | ⏳ pending |
| Stage 1 `core` extraction | ⏳ pending |
| First publication | ⏳ pending |

## License

All repositories are **private** during the research phase. We plan to extract the shared `core` library to a permissive open-source license after the first publication of results.

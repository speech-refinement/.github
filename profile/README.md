# speech-refinement

**Controlled Schrödinger Bridge for Multi-Conditional Speech Refinement.**

A research effort that takes speech degraded by noise, reverberation, temporal dropouts, band-limiting, or codec artifacts and reconstructs the clean reference using **Schrödinger Bridge (SB) generative models** in mel-spectrogram space. We combine a published SB formulation (SB-VE) with a Mamba state-space backbone and extend it along four research axes: adaptive stochastic optimal control, multi-metric perceptual loss, multi-conditional refinement (text / mask / descriptor), and TTS-guided long-dropout recovery.

## Why Schrödinger bridges?

Standard diffusion-based speech enhancement (SGMSE+, CDiffuSE) starts from Gaussian noise and treats the degraded signal as a conditioning variable. **Schrödinger bridges start directly from the degraded signal** and explicitly model the degraded → clean coupling as a stochastic optimal-control (SOC) problem. This gives us a clean mathematical handle to control the generation path (kinetic cost, perceptual cost, terminal penalty) and, when combined with a Mamba backbone, unlocks 1-step fast inference without a separate distillation stage.

## Research trajectory

| Direction | Content | Status |
|---|---|---|
| **Pre-pivot (archived)** | In-house γ-tuned bridge (UniDB-inspired) + flow-matching variant + perceptual SOC | Closed out 2026-04-14 after systematic γ-sweep showed no advantage over γ=1 baseline and a bug audit of the flow variant confirmed structural issues. Findings + code preserved. |
| **Phase S1** | **SB-VE baseline** (faithful port from sp-uhh/sgmse) over the existing real-mel UNet backbone | ✅ codebase complete, decision-gate training running |
| **Phase S2** | **SB Mamba core** — swap backbone to a SEMamba-style state-space model, targeting 1-step inference and smaller parameter count | 🔜 starts once S1 beats the γ=1 reference |
| **Phase S3** | Adaptive stochastic optimal control — per-sample SOC weights (kinetic / perceptual / terminal) conditioned on degradation type | ⏳ parallel after S2 |
| **Phase S4** | Multi-metric perceptual running cost — PESQ + STOI + SI-SDR via differentiable SQUIM + torch-pesq | ⏳ parallel after S2 |
| **Phase C** | Multi-conditional refinement — region mask, transcript (XPhoneBERT), natural-language descriptor | ⏳ parallel after S2 |
| **Phase E** | TTS-guided reference bridge (Matcha-TTS) for long-dropout restoration | ⏳ parallel after S2 |
| **Phase F** | SBCTM distillation as an **optional** fallback if the Mamba backbone does not reach the 1-step quality target | ⏳ conditional |
| **Phase P** | Paper integration (ICASSP-class) + audio demo site | ⏳ after S2 + at least two of S3/S4/C/E |

## Architecture

```
Degraded speech (waveform)
        │
        ▼
   ┌────────────────────────────────────┐
   │  Mel transform (shared)            │
   │  22050 Hz, 80 mels, hop 256        │
   │  log-mel ∈ [-1, 1]                 │
   └────────────────────────────────────┘
        │
        ▼  y: (B, 1, 80, T) degraded mel
        │
   ┌────────────────────────────────────┐
   │  Schrödinger Bridge (SB-VE)        │
   │  ┌──────────────────────────────┐  │
   │  │ Phase S1: mc-ddbm UNet       │  │
   │  │ Phase S2: Mamba backbone     │  │
   │  │   (state-space recurrence    │  │
   │  │    absorbs SB trajectory →   │  │
   │  │    1-step inference)         │  │
   │  ├──────────────────────────────┤  │
   │  │ Phase S3: adaptive SOC       │  │
   │  │   (λ_kin, λ_perc, γ)(d)      │  │
   │  ├──────────────────────────────┤  │
   │  │ Phase S4: multi-metric       │  │
   │  │   perceptual running cost    │  │
   │  │   (PESQ + STOI + SI-SDR)     │  │
   │  ├──────────────────────────────┤  │
   │  │ Phase C: multi-conditioning  │  │
   │  │   (mask + transcript + NL)   │  │
   │  ├──────────────────────────────┤  │
   │  │ Phase E: TTS reference pre-  │  │
   │  │   fill (Matcha-TTS)          │  │
   │  └──────────────────────────────┘  │
   └────────────────────────────────────┘
        │
        ▼  x̂_0: refined mel
        │
   ┌────────────────────────────────────┐
   │  BigVGAN vocoder                   │
   └────────────────────────────────────┘
        │
        ▼
   Refined speech (waveform)
```

## Repository map

| Repo | Purpose | Status |
|---|---|---|
| [**soc-bridge**](https://github.com/speech-refinement/soc-bridge) | Main research repo. SB-VE port, score-matching training, PC sampler, unified eval harness. Mamba backbone and the S3/S4/C/E extensions land here. | active |
| [**mc-ddbm**](https://github.com/speech-refinement/mc-ddbm) | Pre-pivot baseline + shared infrastructure: mel I/O, BigVGAN vocoder, dataset, degradation pipeline, eval metrics, UNet backbone. Frozen at v0.1.0, consumed by soc-bridge as an editable install. | frozen @ v0.1.0 |
| [**gui**](https://github.com/speech-refinement/gui) | Gradio + Node UI for interactive inference. | active |
| [**meta**](https://github.com/speech-refinement/meta) | Governance — shared Claude hooks, agents, roadmap, research docs. | active |
| **core** *(future)* | Shared library for mel I/O, vocoder, dataset, degradation, eval, training scaffold once extraction starts. | planned |
| [**.github**](https://github.com/speech-refinement/.github) | This README + org-wide config. | active |

## Current status (2026-04-15)

| Item | Status |
|---|---|
| Phase S1.1: SBVESDE faithful port | ✅ merged (13 math tests) |
| Phase S1.2: score wrapper + score-matching loss | ✅ merged (6 tests incl. real mc-ddbm UNet) |
| Phase S1.3: training script + config | ✅ merged |
| Phase S1.4: predictor-corrector sampler | ✅ merged (6 tests) |
| Phase S1.5: SB-VE integrated into `eval_all_configs.py` | ✅ merged |
| Phase S1 decision-gate training | 🔄 running (GPU 0) |
| Phase S2 (SB Mamba backbone) | ⏳ gated on S1 beating γ=1 reference |
| Pre-pivot checkpoints (γ=0.1 / 1 / 10 / 100, flow) | ✅ archived for ablation |

## Theoretical foundation

- **Schrödinger Bridge for Speech Enhancement** — Jukić et al. "Schrödinger Bridge for Generative Speech Enhancement," 2024. [sp-uhh/sgmse](https://github.com/sp-uhh/sgmse) (MIT). SB-VE training framework — Phase S1 source.
- **SB Mamba** — Yang et al. "Schrödinger Bridge with Mamba for Speech Enhancement," [arXiv:2510.16834](https://arxiv.org/abs/2510.16834), 2025. Target architecture; code unpublished, so we reproduce its spirit via sgmse + SEMamba.
- **SEMamba** — Chao et al. "An Investigation of Incorporating Mamba for Speech Enhancement," SLT 2024. [RoyChao19477/SEMamba](https://github.com/RoyChao19477/SEMamba) (MIT). Mamba backbone — Phase S2 source.
- **SBCTM** — Sony AI. "Schrödinger Bridge Consistency Trajectory Model," 2025. [sony/sbctm](https://github.com/sony/sbctm). Distillation fallback — Phase F source.
- **DBIM** — Shi et al. "Diffusion Bridge Implicit Models." NeurIPS 2024. [arXiv:2405.15885](https://arxiv.org/abs/2405.15885). Reference for fast bridge sampling.
- **BigVGAN** — Lee et al. "BigVGAN: A Universal Neural Vocoder with Large-Scale Training." ICLR 2023. [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN). Default vocoder.

## Demo

*Coming soon* — once Phase S1 clears the decision gate and Phase S2 produces a 1-step Mamba checkpoint, a GitHub Pages site with clean / degraded / refined audio triplets will be published alongside the paper.

## Getting started

```bash
mkdir speech-refinement && cd speech-refinement
git clone https://github.com/speech-refinement/meta.git
bash meta/scripts/bootstrap.sh
```

See [`meta/README.md`](https://github.com/speech-refinement/meta) for workspace setup and hook architecture.

## License

All repositories are **private** during active research. The shared `core` library will be released under a permissive open-source license after publication. Upstream licenses (MIT from sgmse, SEMamba, BigVGAN) are preserved in the respective port directories.

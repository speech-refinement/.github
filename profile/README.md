# speech-refinement

**Controlled Diffusion Bridge for Speech Refinement.**

A research effort that takes speech degraded by noise, distortion, band-limiting, or codec artifacts and restores it using **controlled diffusion bridge models** in mel-spectrogram space. We exploit the bridge formulation — where the degraded signal is the direct starting point of generation — and introduce systematic control over the bridge path via γ-tuning, degradation-adaptive conditioning, flow matching acceleration, and perceptual optimal control.

## Why bridges?

Standard diffusion-based speech enhancement (SGMSE+, CDiffuSE) starts from Gaussian noise and conditions on the degraded signal. **Bridge models start directly from the degraded signal**, making the degraded→clean mapping a first-class part of the generative process rather than an afterthought. This gives us a handle to *control* the generation path — how aggressively to denoise, how closely to follow the input, how fast to converge.

## Research phases

| Phase | Content | Status |
|---|---|---|
| **1** | **γ-tuned bridge** — UniDB-style γ parameter controls bridge tightness. Sweep γ ∈ {0.1, 1} on speech mel-spectrograms. | 🔄 training |
| **2a** | **Degradation-adaptive γ** — learn γ(degradation_type) instead of scalar γ. Noise→low γ, clipping→high γ. | planned |
| **2b** | **Bridge Flow Matching** — replace SDE with ODE flow for 10x sampling speedup. No teacher needed. | planned |
| **2c** | **Perceptual SOC** — neural optimal control with DNSMOS/NISQA as differentiable cost function. | exploratory |
| **3** | **Paper** — ablation study across all phases + audio demo site. | future |

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
        ▼  x_T: (B, 1, 80, T) degraded mel
        │
   ┌────────────────────────────────────┐
   │  Controlled Diffusion Bridge       │
   │  ┌──────────────────────────────┐  │
   │  │ Phase 1: γ-tuned bridge      │  │
   │  │   q_γ ∝ p(x_t|x_0)·p(xT|xt)^γ│ │
   │  ├──────────────────────────────┤  │
   │  │ Phase 2a: adaptive γ(d)      │  │
   │  │   γ = f_θ(deg_embedding)     │  │
   │  ├──────────────────────────────┤  │
   │  │ Phase 2b: flow matching      │  │
   │  │   ODE flow, 4-8 steps        │  │
   │  ├──────────────────────────────┤  │
   │  │ Phase 2c: perceptual SOC     │  │
   │  │   cost = DNSMOS surrogate    │  │
   │  └──────────────────────────────┘  │
   └────────────────────────────────────┘
        │
        ▼  x_0: refined mel
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
| [**mc-ddbm**](https://github.com/speech-refinement/mc-ddbm) | Baseline DDBM + conditioning experiments (Axis A, completed). Includes mel I/O, vocoder, dataset, degradation, eval metrics. Source for `core` extraction. | v0.1.0 |
| [**soc-bridge**](https://github.com/speech-refinement/soc-bridge) | Main research repo. γ-tuned bridge, adaptive γ, flow matching, perceptual SOC. | active |
| **core** *(Stage 1)* | Shared library: mel I/O, vocoder, dataset, degradation, eval, base UNet, training scaffold. | planned |
| [**meta**](https://github.com/speech-refinement/meta) | Governance: shared hooks, agents, roadmap, research docs. | active |
| [**.github**](https://github.com/speech-refinement/.github) | This README + org-wide config. | active |

## Theoretical foundation

- **DDBM** — Zhou et al. "Denoising Diffusion Bridge Models." ICLR 2024. [alexzhou907/DDBM](https://github.com/alexzhou907/DDBM). The base bridge formulation.
- **UniDB** — Zhu et al. "UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control." ICML 2025 Spotlight. [arXiv:2502.05749](https://arxiv.org/abs/2502.05749). The γ-tuning source.
- **Bridge Matching** — Shi et al. "Diffusion Bridge Implicit Models." NeurIPS 2024. [arXiv:2405.15885](https://arxiv.org/abs/2405.15885). Flow matching for bridges.
- **DBFS** — Park et al. "Stochastic Optimal Control for Diffusion Bridges in Function Spaces." NeurIPS 2024. [arXiv:2405.20630](https://arxiv.org/abs/2405.20630). Function-space bridge theory.
- **BigVGAN** — Lee et al. "BigVGAN: A Universal Neural Vocoder with Large-Scale Training." ICLR 2023. [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN). The default vocoder.

## Current status

| Item | Status |
|---|---|
| Phase 1: γ-tuned bridge implementation | ✅ complete |
| Phase 1: γ={1, 0.1} training | 🔄 running (GPU 0, 1) |
| DCT feasibility (function-space prep) | ✅ PASS (SNR 25+ dB) |
| Phase 2a: degradation-adaptive γ | 📋 design |
| Phase 2b: bridge flow matching | 📋 planned |
| Stage 0: workspace bootstrap | ✅ complete |
| Stage 1: core extraction | ⏳ pending |

## Demo

*Coming soon* — Phase 1 results will include audio comparison (clean / degraded / γ ablation). GitHub Pages site with embedded audio planned after Phase 2.

## Getting started

```bash
mkdir speech-refinement && cd speech-refinement
git clone https://github.com/speech-refinement/meta.git
bash meta/scripts/bootstrap.sh
```

See [`meta/README.md`](https://github.com/speech-refinement/meta) for workspace setup and hook architecture.

## License

All repositories are **private** during active research. The shared `core` library will be released under a permissive open-source license after publication.

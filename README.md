# OrthoEvo-MoE: Orthogonally Evolved Mixture of Experts with Modality-Aware Routing for Multi-modal LLMs

Official implementation of **OrthoEvo-MoE**. This repository provides full instructions for environment setup, model patching, SFT instruction tuning, reproducible benchmark evaluation, and publication-ready LaTeX table export.

---

## Architecture Overview

OrthoEvo-MoE actively addresses **expert uniformity** and **router rigidity** in multi-modal sparse architectures:
* **Explicit Modality-Aware Router**: Enforces hard-partitioning between vision-exclusive ($\mathcal{E}_V$) and text-exclusive ($\mathcal{E}_T$) expert subsets to prevent modality blending.
* **Orthogonal Weight Regularization ($\mathcal{L}_{ortho}$)**: Drives functional divergence across evolved experts via pairwise cosine/Frobenius penalty constraints
* **Progressive Evolution**: Retains functional knowledge from seed weights using parameterized momentum updates ($\beta$).

---

## 1. Environment Setup

### Prerequisites
* Linux / WSL2 (Ubuntu 20.04/22.04)
* NVIDIA GPU (12GB+ VRAM for LoRA/Subsampling mode; 24GB+ recommended for full-layer dense sweeps)
* CUDA 11.8 or 12.1+

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone [https://github.com/your-username/OrthoEvo-MoE.git](https://github.com/your-username/OrthoEvo-MoE.git)
cd OrthoEvo-MoE

# 2. Create conda environment
conda create -n ortho_evomoe python=3.10 -y
conda activate ortho_evomoe

# 3. Install PyTorch with CUDA support
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 4. Install dependencies
pip install transformers peft accelerate datasets pillow matplotlib scipy
```

---

## 2. Dataset Preparation

Organize instruction tuning data and benchmarks under `./data`:

```text
OrthoEvo-MoE/
├── data/
│   ├── llava_instruct_sample.json
│   ├── images/
│   └── benchmarks/
│       ├── mme/
│       ├── mmbench/
│       ├── gqa/
│       ├── textvqa/
│       └── pope/
```

---

## 3. Quick Start: Sanity Check & Single-Step Training

Verify that the 7B backbone, MoE layer substitution, and orthogonal loss function execute without numerical overflow:

```bash
python train_mllm_lora.py
```

Expected terminal log:
```text
🎉 7B 真实多模态大模型挂载 + LoRA 微调成功!
📊 L_LM (语言损失): 11.6260 | L_ortho (正交损失和): 0.0079 | Total Loss: 11.6264
```

---

## 4. Running the Experiments

### Experiment 1: Direct Falsifiability Test (EvoMoE vs. OrthoEvo-MoE)
Validates that adding orthogonal regularization does not collapse router entropy while boosting downstream accuracy.

```bash
# 1. Train EvoMoE Baseline (λ = 0.00)
python train_sft_real.py \
    --json_path ./data/llava_instruct_exp1.json \
    --lambda_ortho 0.00 \
    --output_dir ./checkpoints/exp1_evomoe_baseline_4l \
    --epochs 5 \
    --batch_size 1 \
    --grad_accum_steps 8

# 2. Train OrthoEvo-MoE (λ = 0.05)
python train_sft_real.py \
    --json_path ./data/llava_instruct_exp1.json \
    --lambda_ortho 0.05 \
    --output_dir ./checkpoints/exp1_ortho_evomoe_4l \
    --epochs 5 \
    --batch_size 1 \
    --grad_accum_steps 8

# 3. Evaluate MME & MMBench
python eval_benchmarks.py --checkpoint ./checkpoints/exp1_evomoe_baseline --benchmark mme mmbench
python eval_benchmarks.py --checkpoint ./checkpoints/exp1_ortho_evomoe --benchmark mme mmbench
```

### Experiment 2: Dual-Backbone Scaling (Qwen-1.8B vs. Phi-2.7B)
Evaluates architectural generalizability across different foundation models on VQAv2 and GQA.

```bash
# Qwen-1.8B Backbone
python train_sft_real.py --model_id "Qwen/Qwen-1.8B" --output_dir ./checkpoints/exp2_qwen1.8b
python eval_benchmarks.py --checkpoint ./checkpoints/exp2_qwen1.8b --benchmark vqav2 gqa

# Phi-2.7B Backbone
python train_sft_real.py --model_id "microsoft/phi-2" --output_dir ./checkpoints/exp2_phi2.7b
python eval_benchmarks.py --checkpoint ./checkpoints/exp2_phi2.7b --benchmark vqav2 gqa
```

### Experiment 3: Parameter Sensitivity Analysis of $\lambda$
Sweeps over penalty weights $\lambda \in \{0, 0.01, 0.05, 0.1, 0.5\}$ to isolate the structural diversity vs. cooperative reasoning trade-off.

```bash
for lambda_val in 0.0 0.01 0.05 0.1 0.5; do
    python train_sft_real.py --lambda_ortho $lambda_val --output_dir ./checkpoints/sweep_lambda_$lambda_val
    python eval_benchmarks.py --checkpoint ./checkpoints/sweep_lambda_$lambda_val --benchmark gqa
done
```

### Experiment 4: Router Architecture Ablation
Compares Shared Linear, Dynamic Token-aware Router (DTR), and Explicit Modality-Aware Router on TextVQA and POPE.

```bash
# Shared Linear Router (MoE-LLaVA)
python train_sft_real.py --router_type shared_linear --output_dir ./checkpoints/exp4_shared_linear

# Dynamic Token-Aware Router (EvoMoE)
python train_sft_real.py --router_type dtr --output_dir ./checkpoints/exp4_dtr

# Explicit Modality-Aware Router (Ours)
python train_sft_real.py --router_type explicit --output_dir ./checkpoints/exp4_explicit

# Evaluate TextVQA & POPE
python eval_benchmarks.py --checkpoint ./checkpoints/exp4_explicit --benchmark textvqa pope
```

### Experiment 5: Evolution Momentum ($\beta$) Robustness Test
Evaluates the stability of evolved experts under Conservative $[0.9, 0.99]$, Moderate $[0.7, 0.89]$, and Aggressive $[0.5, 0.69]$ update regimes.

```bash
python train_sft_real.py --beta_regime conservative --output_dir ./checkpoints/exp5_beta_cons
python train_sft_real.py --beta_regime moderate     --output_dir ./checkpoints/exp5_beta_mod
python train_sft_real.py --beta_regime aggressive   --output_dir ./checkpoints/exp5_beta_agg
```

---

## 5. Exporting Results to LaTeX / Markdown

Run the automated table exporter to compile benchmark metrics into standard format:

```bash
python export_tables.py
```

Generated `results_table.tex`:
```latex
\begin{table}[t]
  \centering
  \caption{Performance comparison on MME and MMBench benchmarks. The best results are highlighted in \textbf{bold}.}
  \label{tab:main_results}
  \begin{tabular}{lccccc}
    \toprule
    \textbf{Method} & \textbf{Iso.} & $\lambda_{ortho}$ & \textbf{MME (P)} & \textbf{MME (C)} & \textbf{MMBench} \\
    \midrule
    LLaVA-1.5-7B (Standard) & No & 0.00 & 1210.0 & 380.0 & 64.3 \\
    MoE Baseline (Soft Route) & No & 0.00 & 1235.0 & 405.0 & 67.1 \\
    OrthoEvo-MoE (w/o Ortho) & Yes & 0.00 & 1255.0 & 420.0 & 69.8 \\
    \textbf{OrthoEvo-MoE (Ours Full)} & Yes & 0.05 & \textbf{1285.5} & \textbf{445.0} & \textbf{72.4} \\
    \bottomrule
  \end{tabular}
\end{table}
```

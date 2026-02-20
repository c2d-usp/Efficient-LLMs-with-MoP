# Compressing LLMs with MoP: Mixture of Pruners

This repository contains code for the paper Compressing LLMs with MoP: Mixture of pruners.
Models are available at: https://huggingface.co/collections/c2d-usp/compressing-llms-with-mop-mixture-of-pruners

## Overview

MoP (Mixture of Pruners) is a method that combines the strengths of both depth and width pruning to reduce the size of large language and vision-language models while also speeding-up inference.

### Demo

Comparison between the original (unpruned) LLaVA-1.5 (7.06B parameters) and the pruned version using MoP (5.71B parameters):


https://github.com/user-attachments/assets/a7a41bf8-9775-40b2-aa5b-88f675354c81


Comparison between the original (unpruned) LLaMA-3-8B and the 40% pruned version using MoP (4.74B parameters):


https://github.com/user-attachments/assets/da76394d-d1c8-47e2-9c77-5d3416d9666c


## Repository structure

This repo is split into three independent folders:

- `Llama2/`: LLaMA-2 experiments
- `Llama3/`: LLaMA-3 experiments
- `Llava/`: LLaVA experiments

Each folder is self-contained (scripts, pruning code, and outputs live inside that folder).

## Steps

MoP consists of three steps:

1) **Pruning**
2) **Recovery fine-tuning**
3) **Evaluation** (EleutherAI LM Evaluation Harness, `lm_eval`, installed via `requirements.txt`)

## How to run

Ready-to-run bash scripts.

### LLaMA-2

- Full MoP pipeline (runs all iterations up to 18: pruning → recovery fine-tuning → evaluation):
  - `bash Llama2/run_mop_llama2.sh`
- Extremes / baselines (single-path pruning instead of mixing; e.g., only depth *or* only width):
  - `bash Llama2/run_extremes_llama2.sh`

### LLaMA-3

- Minimal script (faster run to reproduce only the 20% and 30% compression points):
  - `bash Llama3/run_mop_llama3_minimal.sh`
- Full MoP pipeline (runs all iterations up to 18: pruning → recovery fine-tuning → evaluation):
  - `bash Llama3/run_mop_llama3.sh`
- Extremes / baselines (single-path pruning instead of mixing):
  - `bash Llama3/run_extremes_llama3.sh`


## Hardware requirements

GPU (or equivalent) with at least 24GB of VRAM. Pruning should take just a few minutes, and the fine-tuning of a single model checkpoint takes about 2 hours on an RTX 4090.

## Acknowledgements
We thank Instituto de Ciência e Tecnologia Itaú (ICTi) for the technical support, resources, and financial aid in the development of the research project. The authors would also like to thank the Programa de Bolsas Itaú (PBI) of the Centro de Ciência de Dados (C2D), supported by Itaú Unibanco S.A.

---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 1
total_pages: 13
date_converted: "2025-11-05"
---

## AcademicEval: Live Long-Context LLM Benchmark
Haozhen Zhang∗
haozhenz@illinois.edu, wazhz14@gmail.com
University of Illinois at Urbana-Champaign
Tao Feng
taofeng2@illinois.edu
University of Illinois at Urbana-Champaign
Pengrui Han
phan12@illinois.edu
University of Illinois at Urbana-Champaign
Jiaxuan You
jiaxuan@illinois.edu
University of Illinois at Urbana-Champaign
Abstract
Large Language Models (LLMs) have recently achieved remarkable performance in long-
context understanding. However, current long-context LLM benchmarks are limited by rigid
context length, labor-intensive annotation, and the pressing challenge of label leakage issues
during LLM training. Therefore, we propose AcademicEval, a live benchmark for evalu-
ating LLMs over long-context generation tasks. AcademicEval adopts papers on arXiv to
introduce several academic writing tasks with long-context inputs, i.e., Title, Abstract,
Introduction, and Related Work, which cover a wide range of abstraction levels and
require no manual labeling. Moreover, AcademicEval integrates high-quality and expert-
curated few-shot demonstrations from a collected co-author graph to enable flexible context
length. Especially, AcademicEval features an efficient live evaluation, ensuring no label
leakage. We conduct a holistic evaluation on AcademicEval, and the results illustrate that
LLMs perform poorly on tasks with hierarchical abstraction levels and tend to struggle with
long few-shot demonstrations, highlighting the challenge of our benchmark. Through exper-
imental analysis, we also reveal some insights for enhancing LLMs’ long-context modeling
capabilities. Code is available at https://github.com/ulab-uiuc/AcademicEval.
1
Introduction
Large Language Models (LLMs) have recently achieved tremendous success in natural language processing
(NLP) tasks (Achiam et al., 2023; AI@Meta, 2024).
## However, when facing long context inputs, LLMs
show a sharp decline in performance, which poses a pressing challenge to LLMs in understanding and
capturing key information in long texts (Li et al., 2024; Liu et al., 2024). Therefore, several long-context LLM
benchmarks are spawned to evaluate LLMs in various settings, including question answering, summarizing,
and reasoning (Shaham et al., 2023; An et al., 2023; Dong et al., 2023; Bai et al., 2023b; Li et al., 2023;
Zhang et al., 2024b). Despite their success, these benchmarks still suffer from concerns of rigid context
length, saturated performance, and being leaked in LLM training.
We envision that the next-generation long-context LLM benchmarks should ideally possess three key features.
(1) Flexible and potentially unlimited context length: existing benchmarks fix the context for each long-
context problem; ideally, the format and length of the context could be flexibly set based on the LLM’s
capability, especially given the release of long-context LLMs (Reid et al., 2024) and their capabilities in
ingesting multi-modal information, e.g., graphs (Dong et al., 2024). (2) High-quality labels derived from
real-world data, minimizing human labeling efforts: existing long-context benchmarks often require human
labeling (Bai et al., 2023b; An et al., 2023; Li et al., 2023; Dong et al., 2023; Zhang et al., 2024b), which
∗Work done as an intern at University of Illinois at Urbana-Champaign
1
arXiv:2510.17725v1  [cs.CL]  20 Oct 2025

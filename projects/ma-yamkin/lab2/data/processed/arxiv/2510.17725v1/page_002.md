---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 2
total_pages: 13
date_converted: "2025-11-05"
---

## Table 1: Comparison with Existing Long-context LLM Benchmarks. Each column indicates the
average input length, whether the annotation is human-assisted, whether there are tasks with hierarchical
abstraction levels, whether it contains few-shot demonstrations, and whether the benchmark is lively updated,
respectively.
## Benchmark
Avg Len
Automatic
Annotation
Hierarchical
Abstraction
Few-shot
Demos
Live
Update
ZeroSCROLLS (Shaham et al., 2023)
∼10K
✓
✗
✗
✗
L-Eval (An et al., 2023)
∼8K
✗
✗
✗
✗
BAMBOO (Dong et al., 2023)
∼16K
✗
✗
✗
✗
LongBench (Bai et al., 2023b)
∼8K
✗
✗
✓
✗
LooGLE (Li et al., 2023)
∼20K
✗
✗
✗
✗
∞Bench (Zhang et al., 2024b)
∼200K
✗
✗
✗
✗
AcademicEval (ours)
Flexible
✓
✓
✓
✓
is costly and limits the size of the benchmarks to about 2000 samples (Xu et al., 2023) (3) Live updates
to mitigate information leakage during LLM pretraining and fine-tuning: benchmark data contamination in
LLM has gradually become a severe issue (Sainz et al., 2023; Ye et al., 2024; Zhu et al., 2024b;a; Xu et al.,
2024); we argue that holding out future data as the val/test set is one of the most effective approaches for
open benchmarks.
Based on these principles, we propose AcademicEval, a live benchmark to evaluate LLMs over long-context
generation tasks. AcademicEval adopts arXiv as its data source and features a suite of academic writing
tasks on each paper without labor-intensive annotation: Title, Abstract, Introduction, and Related
Work, each of which has long-context input and hierarchical abstraction levels. In particular, we construct
a co-author graph via the arXiv API to conveniently obtain co-author papers as high-quality and expert-
curated few-shot demonstrations, which also possess AcademicEval flexible context length. Furthermore,
AcademicEval introduces efficient live evaluation based on the co-author graph, which utilizes the lat-
est papers on arXiv to update the benchmark data periodically and ensures no label leakage. Moreover,
AcademicEval provides in-context few-shot demonstrations for each sample, which is neglected by most ex-
isting long-context LLM benchmarks (Liu et al., 2024; Li et al., 2024). In our experiments, we evaluate three
categories of baselines on AcademicEval: standard LLMs, long-context LLMs, and retrieval-augmented
language models (RALM). Under automatic metrics (BERTScore and ROUGE-L), RALM often attains the
strongest results by concentrating salient evidence into shorter retrieved chunks, while long-context LLMs
and strong standard models remain competitive in several settings. However, an LLM-as-a-Judge evaluation,
which assesses novelty, feasibility, consistency, factuality, and academic style, reveals a more nuanced picture:
retrieval is not always preferred (e.g., for Title/Abstract), whereas it is highly beneficial for Related
Work. Across both evaluations, performance commonly degrades as the input length grows, and correlated
few-shot demonstrations from the co-author graph can provide modest gains for specific model–task pairs.
Overall, the results indicate that AcademicEval is a challenging benchmark that exposes complementary
facets of long-context modeling: overlap-oriented automatic metrics and higher-level judged quality.
We illustrate the comparison with existing long-context LLM benchmarks in Table 1. Our contributions are
summarized as follows:
• We propose a live benchmark, AcademicEval, to evaluate LLMs over long-context generation
tasks. AcademicEval features four academic writing tasks with hierarchical abstraction levels and
requires no manual annotation.
• We construct a co-author graph via the arXiv API and draw on the co-author papers as informative
few-shot demonstrations, making the context length of AcademicEval flexible and scalable. Es-
2

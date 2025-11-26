---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 13
total_pages: 13
date_converted: "2025-11-05"
---

## Table 6: Title-only ablation on Abstract Writing (Abs-9K). The clear degradation indicates genuine
reliance on external context rather than in-weights memorization.
## Setting
Model
BERTScore
ROUGE-L
Default (Abs-9K)
LLaMA
62.4
25.0
Title-only (Abs-9K)
LLaMA
57.4
18.8
Default (Abs-9K)
Hermes
62.2
26.1
Title-only (Abs-9K)
Hermes
56.7
19.3
among which contriever achieves the best results. This is because the summary generation task emphasizes
semantic similarity, which can be well measured by the similarity of dense embeddings. However, the sparse
retrievers perform text chunk recall based on sparse embeddings, and the results are significantly worse than
those of the dense retrievers.
## Understanding the Performance Plateau of AcademicEval. The performance plateau observed at
longer contexts (e.g., 9K→30K) invites further examination of its underlying causes. While our analysis
attributes this plateau partly to the limited ability of current models to utilize ultra-long inputs through
in-context learning (ICL), another plausible factor lies in in-weights learning (IWL) (Chan et al.,
2024). That is, certain academic knowledge may already be internalized during pretraining. In such cases,
adding more context brings diminishing informational returns even when the benchmark itself remains well-
constructed.
To
better
understand
this
phenomenon,
we
analyze
both
structural
and
empirical
evi-
dence.
## Structurally,
AcademicEval
organizes
tasks
across
hierarchical
abstraction
levels
(Title→Abstract→Introduction→Related Work), where deeper contextual reasoning becomes
increasingly essential. Plateaus may thus occur when longer inputs introduce redundancy rather than new
cues, suggesting ICL saturation instead of pure memorization. Empirically, we conduct a Title-only ablation
on the Abstract Writing task under Abs-9K, where most contextual information is removed except for
the paper title. As shown in Table 6, the BERTScore and ROUGE-L of both LLaMA and Hermes drop
sharply (by 5–7 points), confirming that model performance depends strongly on the provided context and
is not solved by IWL alone.
Overall, our evidence indicates that the observed plateau on AcademicEval is primarily driven by imperfect
long-context utilization (ICL limitation), rather than by IWL. This reading is supported by the Title-only
ablation, where removing most contextual information yields substantial drops in both BERTScore and
ROUGE-L, indicating strong dependence on the provided context.
## Moreover, since AcademicEval is
designed as a live-updating benchmark that continuously incorporates newly published arXiv papers via
the co-author graph, the evaluation set evolves over time and reduces the likelihood that performance is
dominated by pre-encoded (in-weights) knowledge. While diminishing informational returns can occur when
additional tokens introduce redundancy, AcademicEval serves as a diagnostic lens showing that the plateau
chiefly reflects current models’ limited ability to exploit ultra-long inputs under realistic, evolving conditions.
This discussion also aims to inspire further reflection in the long-context benchmarking community on how
dataset design and periodic updates can better disentangle ICL and IWL effects, while underscoring the
continued importance of mitigating data leakage in future benchmark construction.
5
Conclusion
In this paper, we propose AcademicEval, a live long-context LLM benchmark for evaluating long-context
generation tasks with hierarchical abstraction levels.
## AcademicEval adopts arXiv as the data source
and introduces several long-context academic writing tasks without manual annotation since the papers on
arXiv can be regarded as original, high-quality, and expert-curated labels. Moreover, we integrate few-shot
demonstrations from a collected co-author graph to make the context length of our benchmark flexible and
scalable. An efficient live evaluation is also designed to make AcademicEval immune to the label leakage
13

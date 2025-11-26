---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 12
total_pages: 13
date_converted: "2025-11-05"
---

BERT Score
Rouge L
50
52
54
56
58
60
62
64
BERT Score
Contriever
DPR
Dragon
BM25
TF-IDF
15
17
19
21
23
25
Rouge L
Gemma Instruct (7B)
BERT Score
Rouge L
50
52
54
56
58
60
62
64
BERT Score
Contriever
DPR
Dragon
BM25
TF-IDF
15
17
19
21
23
25
Rouge L
LLaMA-3 Chat (70B)
Figure 3: Analysis of RALM on Abs-9K. The left figure shows results with Gemma Instruct (7B), while
the right one shows results with LLaMA-3 Chat (70B).
4.4.2
Result Analysis
We report results for the overall preference in Table 5. Compared to BERTScore and ROUGE-L, the LLM-
as-a-Judge evaluation reveals partially different patterns, reflecting that it targets broader qualitative aspects
beyond lexical or semantic overlap.
Title Writing. Under Title-10K, Hermes achieves the highest win rate (72.0), while Mixtral becomes the
top model under the correlated setting Title-31K-G (62.7). This suggests that (1) short, highly abstract
outputs benefit from strong style and concision (favoring Hermes at 10K); and (2) correlated contexts can
help certain models (e.g., Mixtral) at longer lengths when the judge considers qualities beyond surface
similarity. Notably, RALM variants (Gemma†, LLaMA†) are not consistently preferred in the title task,
indicating that aggressive retrieval does not always align with judged title quality.
## Abstract Writing. Qwen attains the highest win rate at Abs-9K (77.0) and remains strong at longer
lengths (Abs-28K: 72.7; Abs-29K-G: 71.0). Mixtral follows closely, while RALM variants trail in preference.
This contrasts with automatic metrics where RALM often ranks highly, suggesting that, for abstracts, the
judge values holistic qualities (coherence, feasibility, academic style) over high overlap with references.
## Introduction Writing. Qwen consistently leads (Intro-8K: 79.3; Intro-28K: 70.3; Intro-28K-G:
70.9), and Hermes improves slightly with correlated contexts (56.9 →59.3). LLaMA† remains competitive
but is not top-ranked. Overall, correlated few-shot demonstrations offer modest gains for some models, sup-
porting that graph-informed contexts can help introductions when evaluated on broader quality dimensions.
## Related Work Writing. In contrast to the above tasks, RALM models (especially LLaMA†) achieve top
preferences at longer lengths (Related-53K: 90.7; Related-53K-G: 90.2), aligning with the intuition that
retrieval is particularly beneficial for Related Work, where judged quality rewards appropriate citations,
prior studies, and domain-specific terminology.
Takeaways. (1) The judge-based preferences are not dominated by RALM across all tasks; instead, pref-
erences depend on task nature and qualitative dimensions. (2) Correlated contexts can yield improvements
in several settings (e.g., Mixtral on Title-31K-G, Hermes on Intro-28K-G), though gains are model-
dependent. (3) The divergence from automatic metrics underscores their complementarity: automatic met-
rics reward overlap, whereas the judge emphasizes higher-level writing quality.
4.5
Discussion
Additional Analysis on RALM. We conduct extensive experiments on RALM on the Abs-9K setting
using standard LLMs Gemma Instruct (7B) and LLaMA-3 Chat (70B), and the results are presented in
Figure 3. We can find that the performance of dense retrievers consistently outperforms sparse retrievers,
12

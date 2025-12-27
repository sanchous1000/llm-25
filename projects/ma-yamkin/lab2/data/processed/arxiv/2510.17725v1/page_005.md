---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 5
total_pages: 13
date_converted: "2025-11-05"
---

## Chronological Split
...
...
...
## Preprocess
Co-author
Co-author Graph
Few-shot Demonstrations Selection
Author Node
Integration
TITLE
ABSTRACT
...
...
MAIN BODY
+
INTRODUCTION
RELATED WORK
CITATION CORPUS
Paper Sample
Figure 1: AcademicEval Benchmark. We construct a co-author graph via arXiv and conduct a chrono-
logical split on all paper samples (training, validation, and test samples are represented by red, orange, and
green, respectively). Each paper sample is preprocessed into separate sections and can be integrated with
few-shot demonstrations from co-author papers.
• Title Writing.
## This task takes a paper’s main body and abstract, along with a specific task
prompt as inputs, and then asks LLMs to output a predicted title.
• Abstract Writing. Similar to the above, this task takes a paper’s main body (with the "Conclu-
sion" section removed) and title, along with a specific task prompt as inputs, and then asks LLMs
to output a predicted abstract.
• Introduction Writing. This task takes a paper’s main body (with the "Introduction" section
removed), title, and abstract, along with a specific task prompt as inputs, and then asks LLMs to
output a predicted introduction.
• Related Work Writing. This task takes a paper’s main body (with the "Related Work" section
removed), title, abstract, and citation corpus (introduced in Section 3.1), along with a specific task
prompt as inputs, and then asks LLMs to output a predicted related work.
Based on the above task descriptions, we can generate four basic benchmark settings with different abstrac-
tion levels, namely Title-10K, Abs-9K, Intro-8K and Related-34K, with suffixes indicating their input
context length4. Intuitively, the paper content itself can be considered as a kind of original, expert-curated,
and high-quality labeled data without manual annotation. Therefore, for evaluation, we directly adopt the
corresponding paper section as the ground truth for each benchmark setting, minimizing human labeling
efforts.
## Integration of Few-shot Demonstrations. Given the rigid context length of current long-context LLM
benchmarks and the general effectiveness of in-context learning in LLMs (Dong et al., 2022; Wei et al.,
2022a;b; Kojima et al., 2022), we propose to integrate long few-shot demonstrations to enable flexible and
scalable context length, and we have two selection options for each sample in the above four basic benchmark
4We use BERT (Devlin et al., 2018) tokenizer by default to count the number of input tokens (output tokens are not
included).
5

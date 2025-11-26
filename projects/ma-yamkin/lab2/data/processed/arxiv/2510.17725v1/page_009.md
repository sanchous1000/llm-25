---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 9
total_pages: 13
date_converted: "2025-11-05"
---

Table 3: Main Results on AcademicEval w.r.t. BERTScore.
## Models
Standard LLMs
Long-context LLMs
RALM
Gemma
LLaMA
Qwen
Mixtral
Hermes
Gemma†
LLaMA†
#Params.
7B
70B
72B
8x7B
8x7B
7B
70B
Context Length
8K
8K
32K
32K
32K
8K
8K
Setting: Title Writing
Title-10K
66.1
74.1
73.9
73.4
74.2
65.8
73.9
Title-30K
-
-
73.0
72.9
73.4
65.7
73.9
Title-31K-G
-
-
72.8
72.8
73.3
65.7
73.8
Setting: Abstract Writing
Abs-9K
59.9
62.4
62.5
61.4
62.2
60.3
61.5
Abs-28K
-
-
61.3
61.2
62.6
60.1
61.4
Abs-29K-G
-
-
61.3
61.4
62.5
60.2
61.3
Setting: Introduction Writing
Intro-8K
54.8
55.8
55.4
54.6
55.2
55.0
55.2
Intro-28K
-
-
54.8
54.0
54.8
55.0
55.2
Intro-28K-G
-
-
54.9
54.1
54.7
55.0
55.3
Setting: Related Work Writing
Related-34K
52.0
56.2
58.5
55.3
57.8
52.4
54.7
Related-53K
-
-
-
-
-
52.4
54.7
Related-53K-G
-
-
-
-
-
52.4
54.8
Bold indicates the highest score in each row.
† denotes augmentation with a retriever (Default: Contriever).
“-” means that the context length is too long to be fed into LLMs.
4.3
Automatic Metric Evaluation
4.3.1
Evaluation Setup
For automatic evaluation metrics, we adopt (1) BERTScore7 (Zhang et al., 2019): This metric leverages
BERT-based embedding to measure semantic similarity between predicted and reference texts. (2) ROUGE-
L (Lin, 2004): This metric evaluates the longest common subsequence between the generated and reference
texts, providing a measure of similarity in terms of sequential matching. For both metrics, higher scores
indicate a better match between the predicted and the reference text.
4.3.2
Result Analysis
We conduct comprehensive experiments on the four academic writing tasks, and the results w.r.t. BERTScore
and RougeL are presented in Table 3 and 4, respectively. Note that we do not conduct experiments on -M
settings because its context length is too long for most of our selected baselines.
## Diverse Task Difficulties and Abstractions. The four tasks we proposed are designed to challenge
LLMs over long-context generation tasks with different abstraction levels. From Table 3 and 4, we can clearly
observe that it provides different difficulties for LLMs to perform well from Title Writing to Related
Work Writing tasks, and the results of all baselines on these four tasks have a relatively obvious trend.
5https://www.together.ai/
6https://www.langchain.com/
7We use deberta-xlarge-mnli (He et al., 2021) instead of the default roberta-large (Liu et al., 2019) as the backbone model
to have the best correlation with human evaluation.
9

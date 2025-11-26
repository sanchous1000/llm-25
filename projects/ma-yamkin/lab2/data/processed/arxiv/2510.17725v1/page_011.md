---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 11
total_pages: 13
date_converted: "2025-11-05"
---

Table 5: Additional results on AcademicEval w.r.t. LLM-as-a-Judge win rate (%).
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
45.7
42.7
63.2
43.1
72.0
50.0
43.9
Title-30K
-
-
54.5
45.6
62.5
47.5
44.9
Title-31K-G
-
-
52.4
62.7
45.4
47.7
43.7
Setting: Abstract Writing
Abs-9K
12.0
55.5
77.0
70.0
61.1
14.3
43.2
Abs-28K
-
-
72.7
66.1
41.2
12.7
42.0
Abs-29K-G
-
-
71.0
65.9
40.7
12.0
43.9
Setting: Introduction Writing
Intro-8K
34.6
63.2
79.3
61.5
58.0
48.8
64.1
Intro-28K
-
-
70.3
60.1
56.9
46.5
62.9
Intro-28K-G
-
-
70.9
61.9
59.3
48.2
63.9
Setting: Related Work Writing
Related-34K
55.9
91.9
91.2
65.6
88.6
71.3
89.8
Related-53K
-
-
-
-
-
72.5
90.7
Related-53K-G
-
-
-
-
-
71.7
90.2
Bold indicates the highest score in each row.
† denotes augmentation with a retriever (Default: Contriever).
“-” means that the context length is too long to be fed into LLMs.
## Impact of Few-shot Demonstrations. From Table 3 and 4, we can observe that the integration of few-
shot demonstrations yields mixed effects: in several settings it is neutral or slightly negative under automatic
metrics, yet correlated demonstrations can produce small but consistent gains for certain model–task pairs.
This shows that current LLMs cannot exploit long few-shot demonstrations to benefit the target tasks well,
emphasizing the importance of evaluating long in-context learning in LLM benchmarks. In addition, we can
also find that few-shot demonstrations from co-author papers generally have a more positive impact on task
performance than randomly selected ones.
4.4
LLM-as-a-Judge Evaluation
4.4.1
Evaluation Setup
To complement automatic metrics, we further incorporate an LLM-as-a-Judge evaluation to capture
higher-level qualitative aspects beyond semantic overlap. Specifically, we employ the open-source Mixtral-
8x22B-Instruct-v0.1 (Jiang et al., 2024) to assess five dimensions of generation quality: (1) Novelty — the
degree to which the content introduces new and meaningful ideas; (2) Feasibility — the plausibility and
practicality of the described methods or claims; (3) Consistency — the internal logical coherence of the out-
put; (4) Factuality — the correctness of factual statements; and (5) Academic Style — the alignment with
conventions of scholarly writing, enabling a more nuanced evaluation of LLM outputs. For each task, we
report the win rate (%), i.e., the percentage of cases where the generated text is preferred over the reference
according to the LLM judge. The detailed LLM-as-a-Judge prompt can be found in Appendix D.
11
